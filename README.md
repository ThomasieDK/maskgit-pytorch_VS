# maskgit-pytorch

Unofficial PyTorch implementation of [MaskGIT](http://arxiv.org/abs/2202.04200). The official Jax implementation can be found [here](https://github.com/google-research/maskgit).

> [!NOTE]
> This repo only focuses on training the second stage of MaskGIT, i.e., the Masked Visual Token Modeling (MVTM) part.
> For the first stage, we support loading the pretrained VQGAN from
> [titok](https://github.com/bytedance/1d-tokenizer),
> [taming-transformers](https://github.com/CompVis/taming-transformers),
> [llamagen](https://github.com/FoundationVision/LlamaGen) or
> [amused](https://huggingface.co/amused/amused-256).
> Note that the titok version is a PyTorch reimplementation of the VQGAN used in the official MaskGIT, with weights converted from the original Jax model.

<br/>



## 🛠️ Installation

> The code is tested with python 3.12, torch 2.4.1 and cuda 12.4.

Clone this repo:

```shell
git clone https://github.com/xyfJASON/maskgit-pytorch.git
cd maskgit-pytorch
```

Create and activate a conda environment:

```shell
conda create -n maskgit python=3.12
conda activate maskgit
```

Install dependencies:

```shell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

<br/>



## 🤖️ Pretrained Models

### 🤖️ Stage-1 Models (VQGAN)

The pretrained VQGAN from [titok](https://github.com/bytedance/1d-tokenizer) can be downloaded by:

```shell
mkdir -p ckpts/titok
wget 'https://huggingface.co/fun-research/TiTok/resolve/main/maskgit-vqgan-imagenet-f16-256.bin' -O 'ckpts/titok/maskgit-vqgan-imagenet-f16-256.bin'
```

The pretrained VQGAN from [taming-transformers](https://github.com/CompVis/taming-transformers) can be downloaded by:

```shell
mkdir -p ckpts/taming
wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_16384.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_16384.yaml'
```

The pretrained VQGAN from [llamagen](https://github.com/FoundationVision/LlamaGen) can be downloaded by:

```shell
mkdir -p ckpts/llamagen
wget 'https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt' -O 'ckpts/llamagen/vq_ds16_c2i.pt'
```

The pretrained VQGAN from [amused](https://huggingface.co/amused/amused-256) will be automatically downloaded when running the training / evaluation script.

<br/>



## 🚀 Evaluate Stage-1 Models (VQGAN)

```shell
accelerate-launch evaluate_vqmodel.py --model_name MODEL_NAME --dataroot IMAGENET_DATAROOT [--bspp BATCH_SIZE_PER_PROCESS]
```

**Quantitative reconstruction results on ImageNet (256x256)**:

|               Model Name               |  PSNR ↑   |  SSIM ↑  | LPIPS ↓  |  rFID ↓  |
|:--------------------------------------:|:---------:|:--------:|:--------:|:--------:|
| `titok/maskgit-vqgan-imagenet-f16-256` |   18.15   |   0.43   |   0.20   | **2.12** |
|   `taming/vqgan_imagenet_f16_16384`    |   20.01   |   0.50   |   0.17   |   5.00   |
|         `llamagen/vq_ds16_c2i`         |   20.79   |   0.56   | **0.14** |   2.19   |
|          `amused/amused-256`           | **21.81** | **0.58** | **0.14** |   4.41   |

**Qualitative reconstruction results (384x384)**:

<table>
<tr>
    <td align="center">original</td>
    <td align="center">titok</td>
    <td align="center">taming</td>
    <td align="center">llamagen</td>
    <td align="center">amused</td>
</tr>
<tr>
    <td width="12%"><img src="/assets/test_img_3.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_3_titok.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_3_taming.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_3_llamagen.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_3_amused.png" alt="" /></td>
</tr>
<tr>
    <td width="12%"><img src="/assets/test_img_2.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_2_titok.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_2_taming.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_2_llamagen.png" alt="" /></td>
    <td width="12%"><img src="./assets/test_img_2_amused.png" alt="" /></td>
</tr>
</table>

The original images are taken from ImageNet and CelebA-HQ respectively.

<br/>



## 🔥 Train Stage-2 Models (MaskGIT)

### 🔥  Step 1: cache the latents (optional but highly recommended)

```shell
accelerate-launch make_cache.py -c CONFIG --save_dir CACHEDIR [--bspp BATCH_SIZE_PER_PROCESS]
```

### 🔥  Step 2: start training

To train an unconditional model (e.g. FFHQ):

```shell
# if not using cached latents
accelerate-launch train.py -c CONFIG -e EXPDIR
# if using cached latents
accelerate-launch train.py -c CONFIG -e EXPDIR --data.name cached --data.root CACHEDIR
```

<br/>



## 🖋️ References

MaskGIT:

```
@inproceedings{chang2022maskgit,
  title={Maskgit: Masked generative image transformer},
  author={Chang, Huiwen and Zhang, Han and Jiang, Lu and Liu, Ce and Freeman, William T},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11315--11325},
  year={2022}
}
```

VQGAN (Taming Transformers):

```
@inproceedings{esser2021taming,
  title={Taming transformers for high-resolution image synthesis},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12873--12883},
  year={2021}
}
```

LlamaGen:

```
@article{sun2024autoregressive,
  title={Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation},
  author={Sun, Peize and Jiang, Yi and Chen, Shoufa and Zhang, Shilong and Peng, Bingyue and Luo, Ping and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2406.06525},
  year={2024}
}
```

aMUSEd:

```
@misc{patil2024amused,
  title={aMUSEd: An Open MUSE Reproduction}, 
  author={Suraj Patil and William Berman and Robin Rombach and Patrick von Platen},
  year={2024},
  eprint={2401.01808},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
