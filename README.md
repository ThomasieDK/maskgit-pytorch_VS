# maskgit-pytorch

Unofficial PyTorch implementation of [MaskGIT: Masked Generative Image Transformer](http://arxiv.org/abs/2202.04200). The official Jax implementation can be found [here](https://github.com/google-research/maskgit).

<br/>



## Installation

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



## Stage-1 (VQGAN)

Instead of training a VQGAN from scratch, we directly use the pretrained VQGANs from the community as the image tokenizer.



### Download

We support loading the pretrained VQGAN models from several open-source projects, including:

**VQGAN-MaskGIT**: The pretrained VQGAN used in the original MaskGIT paper is implemented in Jax. The community has converted the model weights to PyTorch, which can be downloaded by:

```shell
mkdir -p ckpts
wget 'https://huggingface.co/fun-research/TiTok/resolve/main/maskgit-vqgan-imagenet-f16-256.bin' -O 'ckpts/maskgit-vqgan-imagenet-f16-256.bin'
```

**VQGAN-Taming**: The pretrained VQGAN from [taming-transformers](https://github.com/CompVis/taming-transformers) can be downloaded by:

```shell
mkdir -p ckpts/taming
wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_16384.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'ckpts/taming/vqgan_imagenet_f16_16384.yaml'
```

**VQGAN-LlamaGen**: The pretrained VQGAN from [llamagen](https://github.com/FoundationVision/LlamaGen) can be downloaded by:

```shell
mkdir -p ckpts/llamagen
wget 'https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt' -O 'ckpts/llamagen/vq_ds16_c2i.pt'
```

**VQGAN-aMUSEd**: The pretrained VQGAN from [amused](https://huggingface.co/amused/amused-256) will be automatically downloaded when running the training / evaluation script.

<br/>



### Evaluation

```shell
accelerate-launch evaluate_vqmodel.py --model_name MODEL_NAME \
                                      --dataroot IMAGENET_DATAROOT \ 
                                      [--save_dir SAVE_DIR] \
                                      [--bspp BATCH_SIZE_PER_PROCESS]
```

- `--model_name`: name of the pretrained VQGAN model. Options:
  - `maskgit-vqgan-imagenet-f16-256`
  - `taming/vqgan_imagenet_f16_16384`
  - `llamagen/vq_ds16_c2i`
  - `amused/amused-256`
- `--dataroot`: the root directory of the ImageNet dataset.
- `--save_dir`: the directory to save the reconstructed images.
- `--bspp`: batch size per process.

**Quantitative reconstruction results on ImageNet (256x256) validation set**:

|              Model Name               | Codebook Size | Codebook Dim |  PSNR ↑   |  SSIM ↑  | LPIPS ↓  |  rFID ↓  |
|:-------------------------------------:|:-------------:|:------------:|:---------:|:--------:|:--------:|:--------:|
|   `maskgit-vqgan-imagenet-f16-256`    |     1024      |     256      |   18.15   |   0.43   |   0.20   | **2.12** |
|   `taming/vqgan_imagenet_f16_16384`   |     16384     |     256      |   20.01   |   0.50   |   0.17   |   5.00   |
|        `llamagen/vq_ds16_c2i`         |     16384     |      8       |   20.79   |   0.56   | **0.14** |   2.19   |
|          `amused/amused-256`          |     8192      |      64      | **21.81** | **0.58** | **0.14** |   4.41   |

**Qualitative reconstruction results (384x384)**:

<table>
<tr>
    <td align="center">original</td>
    <td align="center">maskgit</td>
    <td align="center">taming</td>
    <td align="center">llamagen</td>
    <td align="center">amused</td>
</tr>
<tr>
    <td width="12%"><img src="/assets/stage1/test_img_3.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_3_maskgit.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_3_taming.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_3_llamagen.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_3_amused.png" alt="" /></td>
</tr>
<tr>
    <td width="12%"><img src="/assets/stage1/test_img_2.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_2_maskgit.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_2_taming.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_2_llamagen.png" alt="" /></td>
    <td width="12%"><img src="./assets/stage1/test_img_2_amused.png" alt="" /></td>
</tr>
</table>

The original images are taken from ImageNet and CelebA-HQ respectively.
It can be observed that better rFID doesn't necessarily mean better visual quality.

<br/>



## Stage-2 (Transformer)

### Training

**Step 1 (optional): cache the latents**.
Caching the latents encoded by VQGAN can greatly accelerate the training and decrease the memory usage in the stage-2 training.
However, make sure you have enough disk space to store the cached latents.

|         Dataset         |   VQGAN type   | Disk space required |
|:-----------------------:|:--------------:|:-------------------:|
|          FFHQ           |  VQGAN-aMUSEd  |       \> 18G        |
| ImageNet (training set) | VQGAN-MaskGIT  |       \> 1.3T       |

```shell
accelerate-launch make_cache.py -c CONFIG --save_dir CACHEDIR [--bspp BATCH_SIZE_PER_PROCESS]
```

- `-c`: path to the config file, e.g., `./configs/imagenet256.yaml`.
- `--save_dir`: the directory to save the cached latents.
- `--bspp`: batch size per process.

**Step 2: start training**.
To train an **unconditional** model (e.g. FFHQ), run the following command:

```shell
# if not using cached latents
accelerate-launch train.py -c CONFIG [-e EXPDIR] [-mp MIXED_PRECISION]
# if using cached latents
accelerate-launch train.py -c CONFIG [-e EXPDIR] [-mp MIXED_PRECISION] --data.name cached --data.root CACHEDIR
```

To train a **class-conditional** model (e.g. ImageNet), run the following command:

```shell
# if not using cached latents
accelerate-launch train_c2i.py -c CONFIG [-e EXPDIR] [-mp MIXED_PRECISION]
# if using cached latents
accelerate-launch train_c2i.py -c CONFIG [-e EXPDIR] [-mp MIXED_PRECISION] --data.name cached --data.root CACHEDIR
```

- `-c`: path to the config file, e.g., `./configs/imagenet256.yaml`.
- `-e`: the directory to save the experiment logs. Default: `./runs/<current time>`.
- `-mp`: mixed precision training. Options: `no`, `fp16`, `bf16`.

<br/>



### Sampling

To sample from the trained **unconditional** model (e.g., FFHQ), run the following command:

```shell
accelerate-launch sample.py -c CONFIG \
                            --weights WEIGHTS \
                            --n_samples N_SAMPLES \
                            --save_dir SAVEDIR \
                            [--seed SEED] \
                            [--bspp BATCH_SIZE_PER_PROCESS] \
                            [--sampling_steps SAMPLING_STEPS] \
                            [--topk TOPK] \
                            [--softmax_temp SOFTMAX_TEMP] \
                            [--base_gumbel_temp BASE_GUMBEL_TEMP]
```

- `-c`: path to the config file, e.g., `./configs/ffhq256.yaml`.
- `--weights`: path to the trained model weights.
- `--n_samples`: number of samples to generate.
- `--save_dir`: the directory to save the generated samples.
- `--seed`: random seed. Default: 8888.
- `--bspp`: batch size per process. Default: 100.
- `--sampling_steps`: number of sampling steps. Default: 8.
- `--topk`: only select from the top-k tokens in each sampling step. Default: None.
- `--softmax_temp`: softmax temperature. Default: 1.0.
- `--base_gumbel_temp`: temperature for gumbel noise. Default: 4.5.

To sample from the trained **class-conditional** model (e.g., ImageNet), run the following command:

```shell
accelerate-launch sample_c2i.py -c CONFIG \
                                --weights WEIGHTS \
                                --n_samples N_SAMPLES \
                                --save_dir SAVEDIR \
                                [--cfg CFG] \
                                [--cfg_schedule CFG_SCHEDULE] \
                                [--seed SEED] \
                                [--bspp BATCH_SIZE_PER_PROCESS] \
                                [--sampling_steps SAMPLING_STEPS] \
                                [--topk TOPK] \
                                [--softmax_temp SOFTMAX_TEMP] \
                                [--base_gumbel_temp BASE_GUMBEL_TEMP]
```

- `--cfg`: classifier free guidance. Default: 1.0.
- `--cfg_schedule`: schedule for classifier free guidance. Options: "constant", "linear", "power-cosine-[num]". Default: "linear".

<br/>



### Evaluation

We use [OpenAI's ADM Evaluations](https://github.com/openai/guided-diffusion/tree/main/evaluations) to evaluate the image quality.
Please follow their instructions.

You may need to make a `.npz` file from the samples for evaluation:

```shell
python make_npz.py --sample_dir SAMPLE_DIR
```

The script will recursively search for all the images in `SAMPLE_DIR` and save them in `SAMPLE_DIR.npz`.

<br/>



### Results (class-conditional ImageNet 256x256)

Below we show the quantitative and qualitative results of class-conditional ImageNet (256x256).
As a reference, the original MaskGIT paper reports FID=6.18 and IS=182.1 with 8 sampling steps without classifier-free guidance (CFG=1).

**Quantitative results**:

| EMA Model | Sampling Steps |     CFG     | FID ↓ |  IS ↑  | Precision ↑ | Recall ↑ |
|:---------:|:--------------:|:-----------:|:-----:|:------:|:-----------:|:--------:|
|    Yes    |       8        |     1.0     | 8.92  | 124.76 |    0.86     |   0.42   |
|    Yes    |       8        | linear(2.0) | 7.15  | 186.18 |    0.91     |   0.36   |
|    Yes    |       8        | linear(3.0) | 8.18  | 233.28 |    0.94     |   0.31   |

**Uncurated samples**:

<table>
<tr>
    <td align="center">8 steps, cfg=1.0</td>
    <td align="center">8 steps, cfg=linear(2.0)</td>
    <td align="center">8 steps, cfg=linear(3.0)</td>
</tr>
<tr>
    <td width="30%"><img src="./assets/stage2/imagenet256-maskgit-ema-8steps-topall-temp1-gumbel4_5-cfg1.png" alt="" /></td>
    <td width="30%"><img src="./assets/stage2/imagenet256-maskgit-ema-8steps-topall-temp1-gumbel4_5-cfglinear2.png" alt="" /></td>
    <td width="30%"><img src="./assets/stage2/imagenet256-maskgit-ema-8steps-topall-temp1-gumbel4_5-cfglinear3.png" alt="" /></td>
</tr>
</table>

We further improve the results by increasing the training batch size to 2048 and using the arccos mask schedule.
See detailed differences by `diff configs/imagenet256.yaml configs/imagenet256-improved.yaml`.

**Quantitative results**:

| EMA Model | Sampling Steps |     CFG     | FID ↓ |  IS ↑  | Precision ↑ | Recall ↑ |
|:---------:|:--------------:|:-----------:|:-----:|:------:|:-----------:|:--------:|
|    Yes    |       8        |     1.0     | 7.21  | 152.98 |    0.89     |   0.42   |
|    Yes    |       8        | linear(1.5) | 6.21  | 192.62 |    0.91     |   0.39   |
|    Yes    |       8        | linear(2.0) | 6.09  | 226.52 |    0.92     |   0.37   |
|    Yes    |       8        | linear(2.5) | 6.54  | 257.97 |    0.93     |   0.35   |
|    Yes    |       8        | linear(3.0) | 7.19  | 282.99 |    0.94     |   0.32   |

**Uncurated samples**:

<table>
<tr>
    <td align="center">8 steps, cfg=1.0</td>
    <td align="center">8 steps, cfg=linear(2.0)</td>
    <td align="center">8 steps, cfg=linear(3.0)</td>
</tr>
<tr>
    <td width="30%"><img src="./assets/stage2/imagenet256-improved-maskgit-ema-8steps-topall-temp1-gumbel4_5-cfg1.png" alt="" /></td>
    <td width="30%"><img src="./assets/stage2/imagenet256-improved-maskgit-ema-8steps-topall-temp1-gumbel4_5-cfglinear2.png" alt="" /></td>
    <td width="30%"><img src="./assets/stage2/imagenet256-improved-maskgit-ema-8steps-topall-temp1-gumbel4_5-cfglinear3.png" alt="" /></td>
</tr>
</table>

<br/>



## References

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

A Pytorch Reproduction of Masked Generative Image Transformer:

```
@article{besnier2023pytorch,
  title={A Pytorch Reproduction of Masked Generative Image Transformer},
  author={Besnier, Victor and Chen, Mickael},
  journal={arXiv preprint arXiv:2310.14400},
  year={2023}
}
```

TiTok:

```
@inproceedings{yu2024an,
  title={An Image is Worth 32 Tokens for Reconstruction and Generation},
  author={Qihang Yu and Mark Weber and Xueqing Deng and Xiaohui Shen and Daniel Cremers and Liang-Chieh Chen},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=tOXoQPRzPL}
}
```
