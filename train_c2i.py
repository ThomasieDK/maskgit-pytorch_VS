import argparse
import math
import os
from contextlib import nullcontext
from omegaconf import OmegaConf

import accelerate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import CachedFolder
from models import make_vqmodel, EMA, MaskGITSampler
from utils.data import load_data
from utils.logger import get_logger
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config
from utils.misc import get_time_str, check_freq, get_dataloader_iterator
from utils.tracker import StatusTracker


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
    parser.add_argument('-mp', '--mixed_precision', type=str, default=None, help='Mixed precision training')
    parser.add_argument('-cd', '--cover_dir', action='store_true', default=False, help='Cover the experiment directory if it exists')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), subdirs=['ckpt', 'samples'],
            time_str=args.time_str, exist_ok=args.resume is not None, cover_dir=args.cover_dir,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, 'tensorboard'),
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    assert conf.train.batch_size % accelerator.num_processes == 0
    bspp = conf.train.batch_size // accelerator.num_processes  # batch size per process
    micro_batch_size = conf.train.micro_batch_size or bspp  # actual batch size in each iteration
    train_set = load_data(conf.data, split='all' if conf.data.name.lower() == 'ffhq' else 'train')
    train_loader = DataLoader(train_set, batch_size=bspp, shuffle=True, drop_last=True, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Micro batch size: {micro_batch_size}')
    logger.info(f'Gradient accumulation steps: {math.ceil(bspp / micro_batch_size)}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # LOAD PRETRAINED VQMODEL
    with accelerator.main_process_first():
        vqmodel = make_vqmodel(conf.vqmodel.model_name)
    vqmodel = vqmodel.requires_grad_(False).eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load pretrained vqmodel: {conf.vqmodel.model_name}')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(conf.transformer)
    ema = EMA(model.parameters(), **getattr(conf.train, 'ema', dict()))
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    scheduler = instantiate_from_config(conf.train.sched, optimizer=optimizer)
    logger.info(f'Number of parameters of transformer: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # BUILD SAMPLER
    fm_size = conf.data.img_size // vqmodel.downsample_factor  # feature map size
    sampler = MaskGITSampler(model, sequence_length=fm_size ** 2, sampling_steps=8, device=device)

    # RESUME TRAINING
    step = 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt_model = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load ema
        ckpt_ema = torch.load(os.path.join(resume_path, 'ema.pt'), map_location='cpu', weights_only=True)
        ema.load_state_dict(ckpt_ema['ema'])
        logger.info(f'Successfully load ema from {resume_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(resume_path, 'optimizer.pt'), map_location='cpu', weights_only=True)
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {resume_path}')
        # load scheduler
        ckpt_scheduler = torch.load(os.path.join(resume_path, 'scheduler.pt'), map_location='cpu', weights_only=True)
        scheduler.load_state_dict(ckpt_scheduler['scheduler'])
        logger.info(f'Successfully load scheduler from {resume_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(resume_path, 'meta.pt'), map_location='cpu', weights_only=True)
        step = ckpt_meta['step'] + 1
        logger.info(f'Restart training at step {step}')
        del ckpt_model, ckpt_ema, ckpt_optimizer, ckpt_scheduler, ckpt_meta

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)  # type: ignore
    unwrapped_model = accelerator.unwrap_model(model)
    ema.to(device)
    accelerator.wait_for_everyone()

    # TRAINING FUNCTIONS
    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model and ema model
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model.pt'))
        with ema.scope(model.parameters()):
            accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model_ema.pt'))
        # save ema
        accelerator.save(dict(ema=ema.state_dict()), os.path.join(save_path, 'ema.pt'))
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save scheduler
        accelerator.save(dict(scheduler=scheduler.state_dict()), os.path.join(save_path, 'scheduler.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    def train_micro_batch(micro_batch, loss_scale, no_sync):
        idx, y = micro_batch
        B, L = idx.shape
        with accelerator.no_sync(model) if no_sync else nullcontext():
            with accelerator.autocast():
                # transformer forward
                mask = unwrapped_model.get_random_mask(B, L)                                # (B, L)
                masked_idx = torch.where(mask, unwrapped_model.mask_token_id, idx)          # (B, L)
                preds = model(masked_idx, y=y, cond_drop_prob=conf.train.cond_drop_prob)    # (B, L, C)
                preds = preds.reshape(B * L, -1)                                            # (B * L, C)
                mask = mask.reshape(B * L)                                                  # (B * L)
                # cross-entropy loss
                target = idx.reshape(-1)                                                    # (B * L)
                target = torch.where(mask, target, -100)
                loss = F.cross_entropy(
                    input=preds, target=target, ignore_index=-100,
                    label_smoothing=conf.train.label_smoothing,
                )
            # backward
            loss = loss * loss_scale
            accelerator.backward(loss)
        return loss

    def train_step(batch):
        # get data
        if isinstance(train_set, CachedFolder):
            idx = batch['idx'].long()
            y = batch['y'].long()
            B, L = idx.shape
        else:
            x = batch[0].float()
            y = batch[1].long()
            B, N = x.shape[0], conf.data.img_size // vqmodel.downsample_factor
            L = N * N
            # vqmodel encode
            with torch.no_grad():
                idx = vqmodel.encode(x)['indices'].reshape(B, L)

        # forward and backward with gradient accumulation
        loss = torch.tensor(0., device=device)
        for i in range(0, B, micro_batch_size):
            idx_micro_batch = idx[i:i+micro_batch_size]
            y_micro_batch = y[i:i+micro_batch_size]
            loss_scale = idx_micro_batch.shape[0] / B
            no_sync = (i + micro_batch_size) < B
            loss_micro_batch = train_micro_batch((idx_micro_batch, y_micro_batch), loss_scale, no_sync)
            loss = loss + loss_micro_batch

        # optimize
        optimizer.step()
        scheduler.step()
        ema.update(model.parameters())
        optimizer.zero_grad()
        return dict(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def sample(savepath):
        samples_list = []
        for c in conf.train.sample_class_ids:
            n_samples = math.ceil(conf.train.n_samples_per_class / accelerator.num_processes)
            y = torch.full((n_samples, ), c, dtype=torch.long, device=device)
            with ema.scope(model.parameters()):
                idx = sampler.sample(n_samples=n_samples, y=y)
            samples = vqmodel.decode_indices(idx, shape=(n_samples, fm_size, fm_size, -1)).clamp(-1, 1)
            samples = accelerator.gather(samples)[:conf.train.n_samples_per_class]
            samples_list.append(samples.cpu())
        samples = torch.cat(samples_list, dim=0)
        if accelerator.is_main_process:
            nrow = conf.train.n_samples_per_class
            save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_loader_iterator = get_dataloader_iterator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        _batch = next(train_loader_iterator)
        # run a step
        model.train()
        train_status = train_step(_batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()
        # validate
        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>7d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>7d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>7d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    accelerator.end_training()
    logger.info('End of training')


if __name__ == '__main__':
    main()
