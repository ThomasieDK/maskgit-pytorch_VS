import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mask_schedule import get_mask_scheduling_fn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        """ (B, L, D) -> (B, L, D) """
        B, L, D = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)                              # (B, L, D)
        q = q.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)   # (B, H, L, D/H)
        k = k.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)   # (B, H, L, D/H)
        v = v.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)   # (B, H, L, D/H)

        x = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=None, is_causal=False,
            dropout_p=self.dropout if self.training else 0,
        )

        x = x.transpose(1, 2).contiguous().view(B, L, D)                    # (B, L, D)
        x = self.out_dropout(self.out(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        """ (B, L, D) -> (B, L, D) """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MaskTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            n_heads: int,
            n_layers: int,
            n_tokens: int,
            n_classes: int = 0,
            dropout: float = 0.0,
            mask_schedule_type: str = 'cosine',
    ):
        super().__init__()

        # get mask scheduling function
        self.gamma = get_mask_scheduling_fn(mask_schedule_type)

        # token embedding (mask token is the last token)
        self.mask_token_id = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros((1, n_tokens, embed_dim)))
        self.drop_emb = nn.Dropout(dropout)

        # class embedding
        if n_classes > 0:
            # uncond token is the last token
            self.uncond_token_id = n_classes
            self.class_emb = nn.Embedding(n_classes + 1, embed_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

        # weights initialization
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # helper distribution for sampling
        self.gumbel = torch.distributions.Gumbel(0, 1)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor, y: Tensor = None, cond_drop_prob: float = 0.0):
        """ idx (B, L), y (B, 1) -> logits (B, L, C) """
        B, L = idx.shape
        # token embedding
        x = self.token_emb(idx)
        # add position embedding
        x = x + self.pos_emb[:, :L, :]
        x = self.drop_emb(x)
        # prepend class embedding
        if y is not None:
            y = y[:, None] if y.ndim == 1 else y
            cond_drop_mask = torch.lt(torch.rand_like(y, dtype=torch.float), cond_drop_prob)
            y = torch.where(cond_drop_mask, torch.full_like(y, self.uncond_token_id), y)
            class_embed = self.class_emb(y)
            x = torch.cat((class_embed, x), dim=1)
        # forward
        for block in self.blocks:
            x = block(x)
        if y is not None:
            x = x[:, 1:, :]
        # classifier
        logits = self.classifier(x)
        return logits

    def get_random_mask(self, B: int, L: int):
        device = self.pos_emb.device
        n = math.ceil(self.gamma(np.random.random()) * L)
        index = torch.rand((B, L), device=device).topk(n, dim=1).indices
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=index, src=torch.ones_like(mask, dtype=torch.bool))
        return mask

    @torch.no_grad()
    def sample_one_step(
            self, n: int, idx: Tensor, y: Tensor = None, cfg: float = 1.0,
            temp: float = 1.0, topk: int = None, choice_temp: float = 0.0,
    ):
        """ idx (B, L), y (B, 1) -> sampled_idx (B, L) """
        B, L = idx.shape
        mask = torch.eq(idx, self.mask_token_id)
        # get probabilities
        logits = self(idx, y) / temp
        if y is not None and cfg != 1.0:
            logits_uncond = self(idx, y, cond_drop_prob=1.0) / temp
            logits = cfg * logits + (1 - cfg) * logits_uncond
        if topk is not None:
            v, _ = torch.topk(logits, min(topk, L), largest=True, sorted=True)
            logits[logits < v[..., [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        # sample all positions
        sampled_idx = torch.multinomial(probs.reshape(B * L, -1), num_samples=1).reshape(B, L)
        sampled_probs = torch.gather(probs, dim=-1, index=sampled_idx[:, :, None]).reshape(B, L)
        # restore unmasked positions
        sampled_idx = torch.where(mask, sampled_idx, idx)
        sampled_probs = torch.where(mask, sampled_probs, torch.full_like(sampled_probs, torch.inf))
        # unmask top L-n positions (with gumbel noise added for randomness)
        randomness = self.gumbel.sample(sampled_probs.shape).to(sampled_probs.device)
        confidence = torch.log(sampled_probs) + choice_temp * randomness
        index = confidence.topk(L - n, dim=1).indices
        mask = mask.scatter(dim=1, index=index, src=torch.zeros_like(mask, dtype=torch.bool))
        sampled_idx = torch.where(mask, self.mask_token_id, sampled_idx)
        return sampled_idx

    @torch.no_grad()
    def sample_loop(
            self, B: int, L: int, T: int, y: Tensor = None, cfg: float = 1.0, cfg_schedule: str = 'linear',
            temp: float = 1.0, topk: int = None, base_choice_temp: float = 4.5,
    ):
        assert T <= L, f'The number of steps T should <= the sequence length L, but got T={T} and L={L}'
        assert cfg_schedule in ['constant', 'linear']
        device = self.pos_emb.device
        idx = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=device)
        for t in range(T):
            # after this iteration, n positions remain masked
            n = math.floor(self.gamma((t + 1) / T) * L)
            choice_temp = base_choice_temp * (1 - (t + 1) / T)
            cfg_t = cfg
            if cfg_schedule == 'linear':
                cfg_t = 1 + (cfg - 1) * (L - n) / L
            idx = self.sample_one_step(n, idx, y, cfg_t, temp, topk, choice_temp)
            yield idx
