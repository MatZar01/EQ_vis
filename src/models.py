import torch
from torch import nn
import copy


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
    else:
        return NotImplementedError


class Init_net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.embedder_first = nn.Sequential(
            nn.BatchNorm2d(input_size[-1]),
            nn.Conv2d(input_size[-1], 16, (3, 3), 1, 1),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.Conv2d(256, 256, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten()
        )
        self.embedder_second = copy.deepcopy(self.embedder_first)

        self.classifier = nn.Sequential(
            nn.Linear(12544, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb_1 = self.embedder_first(ft1)
        emb_2 = self.embedder_second(ft2)

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_method)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)

class Small_net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.embedder_first = nn.Sequential(
            nn.BatchNorm2d(input_size[-1]),
            nn.Conv2d(input_size[-1], 16, (3, 3), 1, 1),
            nn.Conv2d(16, 16, (3, 3), 1, 1),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(128),
            nn.Flatten()
        )
        self.embedder_second = copy.deepcopy(self.embedder_first)

        self.classifier = nn.Sequential(
            nn.Linear(6272, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb_1 = self.embedder_first(ft1)
        emb_2 = self.embedder_second(ft2)

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_method)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)