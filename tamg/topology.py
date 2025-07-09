import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor

try:
    import gudhi
except Exception:  # pragma: no cover - if gudhi isn't available
    gudhi = None


class TopologicalLoss(nn.Module):
    """Compute differentiable topology loss via persistent homology."""

    def __init__(self, lambda_topo: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.lambda_topo = lambda_topo
        self.eps = eps

    def _betti_signature(self, binary_map: Tensor) -> Tensor:
        if gudhi is None:
            return torch.zeros(2, device=binary_map.device)
        img = binary_map.detach().cpu().numpy().astype(float)
        cc = gudhi.CubicalComplex(top_dimensional_cells=img)
        cc.persistence()
        bettis = cc.betti_numbers()
        b0 = bettis[0] if len(bettis) > 0 else 0
        b1 = bettis[1] if len(bettis) > 1 else 0
        return torch.tensor([b0, b1], device=binary_map.device, dtype=torch.float)

    def forward(
        self,
        logits: Tensor,
        mask: BoolTensor,
        target_signature: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return topology loss and its finite-difference gradient."""
        B, L, C = logits.shape
        size = int(math.sqrt(L))
        probs = torch.softmax(logits, dim=-1)
        tokens = probs.argmax(dim=-1)
        signatures = []
        for b in range(B):
            seg = tokens[b].view(size, size).float()
            signatures.append(self._betti_signature(seg))
        signatures = torch.stack(signatures)
        target = target_signature if target_signature is not None else torch.zeros_like(signatures)
        loss = F.mse_loss(signatures, target)

        grad = torch.zeros_like(logits)
        eps = self.eps
        for b in range(B):
            masked_pos = torch.nonzero(mask[b], as_tuple=False).flatten()
            for l in masked_pos:
                for c in range(C):
                    logits_p = logits.clone()
                    logits_p[b, l, c] += eps
                    probs_p = torch.softmax(logits_p, dim=-1)
                    tokens_p = probs_p.argmax(dim=-1)
                    seg_p = tokens_p[b].view(size, size).float()
                    sig_p = self._betti_signature(seg_p)
                    loss_p = F.mse_loss(sig_p, target[b])
                    grad[b, l, c] = (loss_p - loss) / eps
        return loss, grad
