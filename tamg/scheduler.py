import torch
from torch import Tensor, BoolTensor

from .topology import TopologicalLoss

_topo_helper = TopologicalLoss(lambda_topo=1.0)


def compute_topo_uncertainty(logits: Tensor, mask: BoolTensor) -> Tensor:
    """Estimate uncertainty from topology gradients."""
    with torch.no_grad():
        _, grad = _topo_helper(logits, mask)
        uncertainty = grad.norm(dim=-1)
    return uncertainty


def select_high_uncertainty(uncertainties: Tensor, k: int) -> BoolTensor:
    """Select k positions with the highest uncertainty."""
    index = uncertainties.topk(k, dim=1).indices
    new_mask = torch.zeros_like(uncertainties, dtype=torch.bool)
    new_mask.scatter_(dim=1, index=index, src=torch.ones_like(new_mask, dtype=torch.bool))
    return new_mask
