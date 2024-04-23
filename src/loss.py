import torch


def MSE_weighted(logits: torch.Tensor, target: torch.Tensor, weights: list):
    """weighted MSE loss"""
    intermediate = torch.mean((logits - target) ** 2, dim=1)
    return torch.mean(intermediate * torch.Tensor(weights))

