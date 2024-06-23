import torch
from torch import nn


class MSE_w(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target, weights) -> torch.Tensor:
        intermediate = torch.mean((logits - target) ** 2, dim=1)
        return torch.mean(intermediate * torch.Tensor(weights))


def MSE_weighted(logits: torch.Tensor, target: torch.Tensor, weights: list):
    """weighted MSE loss"""
    intermediate = torch.mean((logits - target) ** 2, dim=1)
    return torch.mean(intermediate * torch.Tensor(weights))

