# maskgit-pytorch

Unofficial PyTorch implementation of [MaskGIT](http://arxiv.org/abs/2202.04200). The official Jax implementation is available at [here](https://github.com/google-research/maskgit).

🌟 This repo only focuses on the stage-2 of MaskGIT, i.e., the Masked Visual Token Modeling (MVTM) part.
We use pretrained VQGAN from [taming-transformers](https://github.com/CompVis/taming-transformers) and [amused](https://huggingface.co/amused/amused-256) as the stage-1 model.



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



## 🤖️ Download

### 🤖️ Stage-1 Model

Download VQGAN pretrained on ImageNet from [taming-transformers](https://github.com/CompVis/taming-transformers):

```shell
mkdir -p ckpts/taming
wget 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_1024.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_1024.yaml'
wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_16384.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_16384.yaml'
```



## 🚀 Evaluate Stage-1 Model

```shell
accelerate-launch evaluate_vqmodel.py --model_name MODEL_NAME --dataroot DATAROOT
```

|            Model Name             | PSNR  | SSIM | LPIPS |
|:---------------------------------:|:-----:|:----:|:-----:|
| `taming/vqgan_imagenet_f16_1024`  | 19.52 | 0.49 | 0.19  |
| `taming/vqgan_imagenet_f16_16384` | 19.98 | 0.50 | 0.17  |



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
