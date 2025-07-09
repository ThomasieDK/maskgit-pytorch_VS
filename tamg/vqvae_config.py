"""Configuration for pathology-trained VQ-VAE."""

from dataclasses import dataclass


@dataclass
class VQVAEConfig:
    encoder_ch_mult = (1, 2, 4, 4)
    decoder_ch_mult = (4, 2, 1, 1)
    codebook_size: int = 1024
    embed_dim: int = 64
    beta: float = 0.25
