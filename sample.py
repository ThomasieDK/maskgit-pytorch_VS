import os
import math
import tqdm
import argparse
from omegaconf import OmegaConf

import accelerate
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from models import make_vqmodel
from utils.logger import get_logger
from utils.image import image_norm_to_float
from utils.misc import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to inference configuration file')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained transformer weights')
    parser.add_argument('--n_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory saving samples')
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    parser.add_argument('--bspp', type=int, default=100, help='Batch size on each process')
    parser.add_argument('--sampling_steps', type=int, default=8, help='Number of sampling steps')
    parser.add_argument('--topk', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--temp', type=float, default=1.0, help='Softmax temperature for sampling')
    parser.add_argument('--base_choice_temp', type=float, default=4.5, help='Base choice temperature for sampling')
    return parser


class DummyDataset(Dataset):
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.names = list(range(n_samples))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        return self.names[index]


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
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.wait_for_everyone()

    # BUILD DATASET AND DATALOADER
    dataset = DummyDataset(args.n_samples)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.bspp, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )

    # LOAD PRETRAINED VQMODEL
    with accelerator.main_process_first():
        vqmodel = make_vqmodel(conf.vqmodel.model_name)
    vqmodel = vqmodel.eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load pretrained vqmodel: {conf.vqmodel.model_name}')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')

    # BUILD MODEL AND LOAD WEIGHTS
    model = instantiate_from_config(conf.transformer).eval().to(device)
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load transformer from {args.weights}')
    logger.info(f'Number of parameters of transformer: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE
    dataloader = accelerator.prepare(dataloader)  # type: ignore
    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    fm_size = conf.data.img_size // vqmodel.downsample_factor
    for name in tqdm.tqdm(dataloader, desc='Sampling', disable=not accelerator.is_main_process):
        B = name.shape[0]
        *_, idx = model.sample_loop(
            B=B, L=fm_size ** 2, T=args.sampling_steps,
            topk=args.topk, temp=args.temp, base_choice_temp=args.base_choice_temp,
        )
        samples = vqmodel.decode_indices(idx, shape=(B, fm_size, fm_size, -1)).clamp(-1, 1)
        samples = image_norm_to_float(samples).cpu()
        for i, sample in zip(name, samples):
            save_image(sample, os.path.join(args.save_dir, f'{i}.png'))
    logger.info(f'Sampled images are saved to {args.save_dir}')
    accelerator.end_training()
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
