import os
import tqdm
import argparse
import numpy as np
from omegaconf import OmegaConf

import accelerate
import torch
from torch.utils.data import DataLoader
torch.backends.cudnn.benchmark = True

from models import make_vqmodel
from utils.data import load_data
from utils.logger import get_logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Set random seed')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--bspp', type=int, default=128, help='Batch size on each process')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory caching latents')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # INITIALIZE LOGGER
    logger = get_logger(use_tqdm_handler=True, is_main_process=accelerator.is_main_process)

    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    train_set = load_data(conf.data, split='all' if conf.data.name.lower() == 'ffhq' else 'train')
    train_loader = DataLoader(train_set, batch_size=args.bspp, shuffle=False, drop_last=False, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {args.bspp}')
    logger.info(f'Total batch size: {args.bspp * accelerator.num_processes}')

    # LOAD PRETRAINED VQMODEL
    vqmodel = make_vqmodel(conf.vqmodel.model_name)
    vqmodel = vqmodel.requires_grad_(False).eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load pretrained vqmodel: {conf.vqmodel.model_name}')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')

    # PREPARE FOR DISTRIBUTED MODE
    vqmodel, train_loader = accelerator.prepare(vqmodel, train_loader)  # type: ignore
    unwrapped_vqmodel = accelerator.unwrap_model(vqmodel)
    accelerator.wait_for_everyone()

    # START CACHING VQMODEL LATENTS
    logger.info('Start caching vqmodel latents')
    logger.info(f'Cached latents will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    cnt = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader, desc='Caching', disable=not accelerator.is_main_process):
            # get data
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch, None
            B = x.shape[0]
            N = conf.data.img_size // unwrapped_vqmodel.downsample_factor

            # encode image
            enc = unwrapped_vqmodel.encode(x)
            h, quant, idx = enc['h'], enc['quant'], enc['indices'].reshape(B, N * N)

            # encode flipped image
            enc_flip = unwrapped_vqmodel.encode(x.flip(dims=[3]))
            h_flip, quant_flip, idx_flip = enc_flip['h'], enc_flip['quant'], enc_flip['indices'].reshape(B, N * N)

            # gather all processes
            h = accelerator.gather_for_metrics(h)
            quant = accelerator.gather_for_metrics(quant)
            idx = accelerator.gather_for_metrics(idx)
            h_flip = accelerator.gather_for_metrics(h_flip)
            quant_flip = accelerator.gather_for_metrics(quant_flip)
            idx_flip = accelerator.gather_for_metrics(idx_flip)
            if y is not None:
                y = accelerator.gather_for_metrics(y)

            # save to npz on main process
            if accelerator.is_main_process:
                for i in range(len(h)):
                    save_path = os.path.join(args.save_dir, f'{cnt}.npz')
                    data = dict(
                        h=h[i].cpu().numpy(),
                        quant=quant[i].cpu().numpy(),
                        idx=idx[i].cpu().numpy(),
                        h_flip=h_flip[i].cpu().numpy(),
                        quant_flip=quant_flip[i].cpu().numpy(),
                        idx_flip=idx_flip[i].cpu().numpy(),
                    )
                    if y is not None:
                        data.update(y=y[i].cpu().numpy())  # type: ignore
                    np.savez(save_path, **data)
                    cnt += 1
    logger.info(f'Cached latents are saved to {args.save_dir}')
    logger.info('Finished caching vqmodel latents')
    accelerator.end_training()


if __name__ == '__main__':
    main()
