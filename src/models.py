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


class Small_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.embedder_first = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 16),
            nn.Tanh(),
            nn.Linear(16, 32)
        )
        self.embedder_second = copy.deepcopy(self.embedder_first)

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb_1 = self.embedder_first(ft1)
        emb_2 = self.embedder_second(ft2)

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_method)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)


class UNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.embedder_first = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.embedder_second = copy.deepcopy(self.embedder_first)

        self.classifier = nn.Sequential(
            #nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb_1 = self.embedder_first(ft1)
        emb_2 = self.embedder_second(ft2)

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_method)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)


class ConvNet_1d(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.embedder_first = nn.Sequential(
            Conv1D_block(input_size, 16),
            nn.ReLU(),
            Conv1D_block(16, 32),
            nn.ReLU()
        )
        self.embedder_second = copy.deepcopy(self.embedder_first)

        self.classifier = nn.Sequential(
            Conv1D_block(32, 16),
            nn.ReLU(),
            Conv1D_block(16, 8),
            nn.Linear(8, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb_1 = self.embedder_first(ft1)
        emb_2 = self.embedder_second(ft2)

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_method)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)


class Conv1D_block(nn.Module):
    def __init__(self, in_fts, out_fts):
        super().__init__()
        self.in_fts = in_fts
        self.out_fts = out_fts
        self.layer = nn.Conv1d(1, self.out_fts, self.in_fts)

    def forward(self, fts):
        fts = fts.unsqueeze(dim=1)
        out = self.layer(fts)
        return out.squeeze()


class ConvNet_2d(nn.Module):
    def __init__(self, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 4, 5),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        fused = self.fuse_fts(ft1, ft2)
        logits = self.classifier(fused)

        return torch.sigmoid(logits)

    def fuse_fts(self, ft1, ft2):
        bsize = ft1.shape[0]
        ft1, ft2 = ft1.unsqueeze(dim=2), ft2.unsqueeze(dim=1)
        out = []
        for i in range(bsize):
            a = ft1[i, :, :]
            b = ft2[i, :, :]
            res = torch.mm(a, b)
            out.append(res)

        return torch.tensor(torch.stack(out)).unsqueeze(dim=1)


class Encoded_ConvNet_2d(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method

        self.encoder_first = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64)
        )
        self.encoder_second = copy.deepcopy(self.encoder_first)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),

            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb1 = self.encoder_first(ft1)
        emb2 = self.encoder_second(ft2)

        fused = self.fuse_fts(emb1, emb2)
        logits = self.classifier(fused)

        return torch.sigmoid(logits)

    def fuse_fts(self, ft1, ft2) -> torch.Tensor:
        bsize = ft1.shape[0]
        ft1, ft2 = ft1.unsqueeze(dim=2), ft2.unsqueeze(dim=1)
        out = []
        for i in range(bsize):
            a = ft1[i, :, :]
            b = ft2[i, :, :]
            res = a * b
            #a1 = torch.triu(torch.ones_like(b) * a)
            #b1 = torch.triu(torch.ones_like(a) * torch.ones_like(b)).T * b
            a1 = a + b
            b1 = a - b
            out.append(torch.stack([a1, res, b1]))

        return torch.stack(out)
        #return torch.tensor(torch.stack(out)).unsqueeze(dim=1)


class ConvNet_2d_slim(nn.Module):
    def __init__(self, output_size: int,):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 4, (3, 12), padding=(1, 6)),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, (3, 13), padding=(1, 6)),
            nn.LeakyReLU(),
            nn.Conv2d(4, 4, (3, 13), padding=(1, 6)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(104, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        fused = self.fuse_fts(ft1, ft2)
        logits = self.classifier(fused)

        return torch.sigmoid(logits)

    def fuse_fts(self, ft1, ft2) -> torch.Tensor:
        return torch.hstack([ft1.unsqueeze(dim=1), ft2.unsqueeze(dim=1)]).unsqueeze(dim=1)
