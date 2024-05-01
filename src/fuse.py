import torch


def fuse_fts(ft1: torch.Tensor, ft2: torch.Tensor, method: int) -> torch.Tensor:
    if method == 0:
        return ft1 * ft2
    elif method == 1:
        return torch.abs(ft1 - ft2)
    elif method == 2:
        return torch.sqrt(ft1 ** 2 + ft2 ** 2)
    elif method == 3:
        return torch.matmul(ft1, ft2)
    elif method == 4:
        return ft1 + ft2
    elif method == 5:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.mean(fts, dim=1)
    elif method == 6:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.max(fts, dim=1).values
    elif method == 7:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.min(fts, dim=1).values
    elif method == 8:
        return torch.abs(ft1 * ft2)
    elif method == 9:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.prod(fts, dim=1)
    elif method == 10:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.std(fts, dim=1)
    elif method == 11:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.norm(fts, dim=1)
    elif method == 12:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.var(fts, dim=1)
    else:
        raise NotImplementedError
