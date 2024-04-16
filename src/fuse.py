import torch


def fuse_fts(ft1: torch.Tensor, ft2: torch.Tensor, method: int) -> torch.Tensor:
    if method == 1:
        return ft1 * ft2
    elif method == 2:
        return torch.abs(ft1 - ft2)
    elif method == 3:
        return torch.sqrt(ft1 ** 2 + ft2 ** 2)
    elif method == 4:
        return torch.tensordot(ft1.unsqueeze(1), ft2.unsqueeze(1))
    elif method == 5:
        return ft1 + ft2
    elif method == 6:
        return torch.vstack([torch.cartesian_prod(ft1[x, :], ft2[x, :])[:, 0] *
                             torch.cartesian_prod(ft1[x, :], ft2[x, :])[:, 1]
                             for x in range(ft1.shape[0])])
    elif method == 7:
        return torch.concatenate([ft1, ft2], dim=1)
    elif method == 8:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.mean(fts, dim=1)
    elif method == 9:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.max(fts, dim=1).values
    elif method == 10:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.median(fts, dim=1).values
    elif method == 11:
        return torch.abs(ft1 * ft2)
    elif method == 12:
        ft1 = torch.unsqueeze(ft1, dim=1)
        ft2 = torch.unsqueeze(ft2, dim=1)
        fts = torch.concatenate([ft1, ft2], dim=1)
        return torch.prod(fts, dim=1)
    else:
        raise NotImplementedError