import argparse
import os
import tqdm
from omegaconf import OmegaConf

import accelerate
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from metrics import PSNR, SSIM, LPIPS
from models import make_vqmodel
from utils.data import load_data
from utils.logger import get_logger
from utils.misc import discard_label
from utils.image import image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to imagenet dataset')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--bspp', type=int, default=64, help='Batch size on each process')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to directory saving samples (for rFID)')
    parser.add_argument('--seed', type=int, default=8888, help='Set random seed')
    return parser


def main():
    # PARSE ARGS
    args = get_parser().parse_args()

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

    # BUILD DATASET & DATALOADER
    conf_data = OmegaConf.create(dict(name='imagenet', root=args.dataroot, img_size=args.img_size))
    dataset = load_data(conf_data, split='valid')
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.bspp, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, prefetch_factor=2,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of dataset: {len(dataset)}')
    logger.info(f'Batch size per process: {args.bspp}')
    logger.info(f'Total batch size: {args.bspp * accelerator.num_processes}')

    # BUILD MODEL
    vqmodel = make_vqmodel(args.model_name)
    vqmodel.eval().to(device)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load {args.model_name} vqmodel')
    logger.info(f'Number of parameters of vqmodel: {sum(p.numel() for p in vqmodel.parameters()):,}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    vqmodel, dataloader = accelerator.prepare(vqmodel, dataloader)  # type: ignore
    accelerator.wait_for_everyone()

    # START EVALUATION
    logger.info('Start evaluating...')
    idx = 0
    if args.save_dir is not None:
        os.makedirs(os.path.join(args.save_dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'reconstructed'), exist_ok=True)

    psnr_fn = PSNR(reduction='none')
    ssim_fn = SSIM(reduction='none')
    lpips_fn = LPIPS(reduction='none').to(device)
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for x in tqdm.tqdm(dataloader, desc='Evaluating', disable=not accelerator.is_main_process):
            x = discard_label(x)
            recx = vqmodel(x)
            recx = recx.clamp(-1, 1)

            x = image_norm_to_float(x)
            recx = image_norm_to_float(recx)
            psnr = psnr_fn(recx, x)
            ssim = ssim_fn(recx, x)
            lpips = lpips_fn(recx, x)

            psnr = accelerator.gather_for_metrics(psnr)
            ssim = accelerator.gather_for_metrics(ssim)
            lpips = accelerator.gather_for_metrics(lpips)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips)

            if args.save_dir is not None:
                x = accelerator.gather_for_metrics(x)
                recx = accelerator.gather_for_metrics(recx)
                if accelerator.is_main_process:
                    for ori, dec in zip(x, recx):
                        save_image(ori, os.path.join(args.save_dir, 'original', f'{idx}.png'))
                        save_image(dec, os.path.join(args.save_dir, 'reconstructed', f'{idx}.png'))
                        idx += 1

    psnr = torch.cat(psnr_list, dim=0).mean().item()
    ssim = torch.cat(ssim_list, dim=0).mean().item()
    lpips = torch.cat(lpips_list, dim=0).mean().item()

    logger.info(f'PSNR: {psnr:.4f}')
    logger.info(f'SSIM: {ssim:.4f}')
    logger.info(f'LPIPS: {lpips:.4f}')
    accelerator.end_training()


if __name__ == '__main__':
    main()
