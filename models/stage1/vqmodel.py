from typing import Tuple, Union
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch import Tensor, FloatTensor

from diffusers import VQModel as aMUSEdVQModel
from .taming.vqmodel import VQModel as TamingVQModel


AVAILABLE_MODEL_NAMES = (
    'taming/vqgan_imagenet_f16_1024',
    'taming/vqgan_imagenet_f16_16384',
    'amused/amused-256',
    'amused/amused-512',
)


def make_vqmodel(name: str):
    assert name in AVAILABLE_MODEL_NAMES, f"Model {name} not available"

    if 'taming' in name:
        from pytorch_lightning.callbacks import ModelCheckpoint
        torch.serialization.add_safe_globals([ModelCheckpoint])
        # load config & build model
        config_path = f'ckpts/{name}.yaml'
        conf = OmegaConf.load(config_path)
        model_params = OmegaConf.to_container(conf.model.params)
        model_params.pop('lossconfig')
        vqmodel = TamingVQModel(**model_params)
        # load weights
        ckpt_path = f'ckpts/{name}.ckpt'
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        vqmodel.load_state_dict(weights['state_dict'], strict=False)
        del weights
        # wrap model
        model = TamingVQModelWrapper(vqmodel)
        model.eval()
        model.requires_grad_(False)
        return model

    elif 'amused' in name:
        # build model & load weights
        vq_model = aMUSEdVQModel.from_pretrained(name, subfolder='vqvae')
        # wrap model
        model = aMUSEdVQModelWrapper(vq_model)
        model.eval()
        model.requires_grad_(False)
        return model

    else:
        raise NotImplementedError(f"Model {name} not implemented")


class TamingVQModelWrapper(nn.Module):
    def __init__(self, vqmodel: TamingVQModel):
        super().__init__()
        self.vqmodel = vqmodel

    def forward(self, x: Tensor):
        recx = self.vqmodel(x)[0]
        return recx

    def encode(self, x: Tensor):
        quant, emb_loss, info = self.vqmodel.encode(x)
        indices = info[-1]
        return dict(quant=quant, indices=indices)

    def decode(self, z: Tensor):
        return self.vqmodel.decode(z)

    def decode_indices(self, indices: Tensor, shape: Tuple[int, ...]):
        quant = self.vqmodel.quantize.get_codebook_entry(indices, shape)
        return self.decode(quant)


class aMUSEdVQModelWrapper(nn.Module):
    def __init__(self, vqmodel: aMUSEdVQModel):
        super().__init__()
        self.vqmodel = vqmodel

    def forward(self, x: Union[Tensor, FloatTensor]):
        recx = self.vqmodel(x).sample
        return recx

    def encode(self, x: Union[Tensor, FloatTensor]):
        z = self.vqmodel.encode(x).latents
        quant, emb_loss, info = self.vqmodel.quantize(z)
        indices = info[-1]
        return dict(quant=quant, indices=indices)

    def decode(self, z: Union[Tensor, FloatTensor]):
        return self.vqmodel.decode(z).sample

    def decode_indices(self, indices: Tensor, shape: Tuple[int, ...]):
        quant = self.vqmodel.quantize.get_codebook_entry(indices, shape)
        return self.decode(quant)
