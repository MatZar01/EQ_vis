import torch
import torchvision.models.resnet
from torch import nn
import copy
from .fuse import fuse_fts

##### Imports for baseline experiments
from torchvision.models import vgg16, resnet18, resnet50
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16, vit_l_16
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_L_16_Weights

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


class Fuse_Mk2_0(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_methods = fuse_methods['H']
        # Stage 1
        self.stage_1_A = nn.Sequential(
            nn.BatchNorm2d(input_size[-1]),
            nn.Conv2d(input_size[-1], 16, (3, 3), 1, 1),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.ReLU(),
        )
        self.stage_1_B = copy.deepcopy(self.stage_1_A)

        # Stage 2
        self.stage_2_A = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_2_B = copy.deepcopy(self.stage_2_A)

        # Stage 3
        self.stage_3_A = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # Stage 4
        self.stage_4_A = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 32, (3, 3), 1, 1),
            nn.ReLU(),
        )
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # Stage 5
        self.stage_5 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # Stage 5
        self.stage_6 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((4, 4))
        )

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_1 = nn.functional.max_pool2d(fused_1, (4, 4))

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_2 = nn.functional.max_pool2d(fused_2, (2, 2))

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Concatenate stages
        fused_stages = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        # Stage 5
        emb = self.stage_5(fused_stages)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class Stage_Module(nn.Module):
    def __init__(self, mod_type: str = 'conv', input_filters: int = 3,  conv_filters: int = 32, pool_size: int = 2):
        super().__init__()
        match mod_type:
            case 'conv':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.GELU()
                )
            case 'conv_pool':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.GELU(),
                    nn.MaxPool2d((pool_size, pool_size))
                )
            case 'conv_upsample':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(input_filters),
                    nn.Conv2d(input_filters, conv_filters//2, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters//2, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.GELU()
                )
            case 'conv_upsample_pool':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(input_filters),
                    nn.Conv2d(input_filters, conv_filters // 2, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters // 2, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.GELU(),
                    nn.MaxPool2d((pool_size, pool_size))
                )
            case 'conv_downsize':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), stride=(pool_size, pool_size), padding=(1, 1)),
                    nn.GELU()
                )

    def forward(self, x):
        return self.module(x)


class Fuse_HV(nn.Module):  # Former Fuse_Mk4
    def __init__(self, input_size: list, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_v = fuse_methods['V']
        self.fuse_h = fuse_methods['H']
        filter_count = 96
        # Stage 1
        self.stage_1_A = Stage_Module(mod_type='conv_upsample', input_filters=input_size[-1], conv_filters=filter_count)
        self.stage_1_B = copy.deepcopy(self.stage_1_A)
        self.stage_1_downsize = Stage_Module(mod_type='conv_downsize', conv_filters=filter_count, pool_size=4)

        # Stage 2
        self.stage_2_A = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=2)
        self.stage_2_B = copy.deepcopy(self.stage_2_A)
        self.stage_2_downsize = Stage_Module(mod_type='conv_downsize', conv_filters=filter_count, pool_size=2)

        # Stage 3
        self.stage_3_A = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=2)
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # Stage 4
        self.stage_4_A = Stage_Module(conv_filters=filter_count)
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # Stage 5
        self.stage_5 = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=2)

        # Stage 6
        self.stage_6 = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=4)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*filter_count, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_1 = self.stage_1_downsize(fused_1)

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_2 = self.stage_2_downsize(fused_2)
        fused_2 = fuse_fts(fused_1, fused_2, method=self.fuse_v)

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_3 = fuse_fts(fused_2, fused_3, method=self.fuse_v)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_4 = fuse_fts(fused_3, fused_4, method=self.fuse_v)

        # Stage 5
        emb = self.stage_5(fused_4)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class Fuse_Mk6(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_methods = fuse_methods['H']
        # Stage 1
        self.stage_1_A = Stage_Module(mod_type='conv_upsample', input_filters=input_size[-1], conv_filters=16)
        self.stage_1_B = copy.deepcopy(self.stage_1_A)
        self.stage_1_downsize = Stage_Module(mod_type='conv_downsize', conv_filters=16, pool_size=4)

        # Stage 2
        self.stage_2_A = Stage_Module(mod_type='conv_upsample_pool', input_filters=16, conv_filters=32, pool_size=2)
        self.stage_2_B = copy.deepcopy(self.stage_2_A)
        self.stage_2_downsize = Stage_Module(mod_type='conv_downsize', conv_filters=32, pool_size=2)

        # Stage 3
        self.stage_3_A = Stage_Module(mod_type='conv_upsample_pool', input_filters=32, conv_filters=64, pool_size=2)
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # Stage 4
        self.stage_4_A = Stage_Module(mod_type='conv_upsample', input_filters=64, conv_filters=128)
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # Stage 5
        self.stage_5 = Stage_Module(mod_type='conv_pool', conv_filters=240, pool_size=2)

        # Stage 6
        self.stage_6 = Stage_Module(mod_type='conv_pool', conv_filters=240, pool_size=4)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*240, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_1 = self.stage_1_downsize(fused_1)

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_2 = self.stage_2_downsize(fused_2)

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Concatenate stages
        fused_stages = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        # Stage 5
        emb = self.stage_5(fused_stages)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class Fuse_Mk5(Fuse_HV):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__(input_size, output_size, fuse_methods)
        self.fuse_v = fuse_methods['V']
        self.fuse_h = fuse_methods['H']
        # New stage after S4
        self.stage_4_1_A = Stage_Module(conv_filters=64)
        self.stage_4_1_B = copy.deepcopy(self.stage_4_1_A)

        # New stage after S6
        self.stage_6_1 = Stage_Module(conv_filters=64)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_1 = self.stage_1_downsize(fused_1)

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_2 = self.stage_2_downsize(fused_2)
        fused_2 = fuse_fts(fused_1, fused_2, method=self.fuse_v)

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_3 = fuse_fts(fused_2, fused_3, method=self.fuse_v)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_4 = fuse_fts(fused_3, fused_4, method=self.fuse_v)

        # Stage 4.1
        emb_A = self.stage_4_1_A(emb_A)
        emb_B = self.stage_4_1_B(emb_B)
        fused_41 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_41 = fuse_fts(fused_4, fused_41, method=self.fuse_v)

        # Stage 5
        emb = self.stage_5(fused_41)

        # Stage 6
        emb = self.stage_6(emb)

        # Stage 6.1
        emb = self.stage_6_1(emb)

        logits = self.classifier(emb)
        return logits


class Fuse_Mk3(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_methods = fuse_methods['H']
        # Stage 1
        self.stage_1_A = Stage_Module(mod_type='conv_upsample', input_filters=input_size[-1], conv_filters=32)
        self.stage_1_B = copy.deepcopy(self.stage_1_A)
        self.stage_1_downsize = Stage_Module(mod_type='conv_downsize', conv_filters=32, pool_size=4)

        # Stage 2
        self.stage_2_A = Stage_Module(mod_type='conv_pool', conv_filters=32, pool_size=2)
        self.stage_2_B = copy.deepcopy(self.stage_2_A)
        self.stage_2_downsize = Stage_Module(mod_type='conv_downsize', conv_filters=32, pool_size=2)

        # Stage 3
        self.stage_3_A = Stage_Module(mod_type='conv_pool', conv_filters=32, pool_size=2)
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # Stage 4
        self.stage_4_A = Stage_Module(conv_filters=32)
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # Stage 5
        self.stage_5 = Stage_Module(mod_type='conv_pool', conv_filters=128, pool_size=2)

        # Stage 5
        self.stage_6 = Stage_Module(mod_type='conv_pool', conv_filters=128, pool_size=4)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_1 = self.stage_1_downsize(fused_1)

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_2 = self.stage_2_downsize(fused_2)

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Concatenate stages
        fused_stages = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        # Stage 5
        emb = self.stage_5(fused_stages)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class Fuse_H(nn.Module):
    def __init__(self, input_size: list, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_methods = fuse_methods['H']
        # Stage 1
        self.stage_1_A = Stage_Module(mod_type='conv_upsample', input_filters=input_size[-1], conv_filters=32)
        self.stage_1_B = copy.deepcopy(self.stage_1_A)

        # Stage 2
        self.stage_2_A = Stage_Module(mod_type='conv_pool', conv_filters=32, pool_size=2)
        self.stage_2_B = copy.deepcopy(self.stage_2_A)

        # Stage 3
        self.stage_3_A = Stage_Module(mod_type='conv_pool', conv_filters=32, pool_size=2)
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # Stage 4
        self.stage_4_A = Stage_Module(conv_filters=32)
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # Stage 5
        self.stage_5 = Stage_Module(mod_type='conv_pool', conv_filters=128, pool_size=2)

        # Stage 5
        self.stage_6 = Stage_Module(mod_type='conv_pool', conv_filters=128, pool_size=4)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_1 = nn.functional.max_pool2d(fused_1, (4, 4))

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_2 = nn.functional.max_pool2d(fused_2, (2, 2))

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        # Concatenate stages
        fused_stages = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        # Stage 5
        emb = self.stage_5(fused_stages)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class Fuse_V(nn.Module):
    def __init__(self, input_size: list, output_size: int, fuse_methods: dict):
        super().__init__()
        filter_count = 64
        self.fuse_methods = fuse_methods['V']
        # Stage 1
        self.stage_1_A = Stage_Module(mod_type='conv_upsample', input_filters=input_size[-1], conv_filters=filter_count)
        self.stage_1_B = copy.deepcopy(self.stage_1_A)

        # Stage 2
        self.stage_2_A = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=2)
        self.stage_2_B = copy.deepcopy(self.stage_2_A)

        # Stage 3
        self.stage_3_A = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=2)
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # Stage 4
        self.stage_4_A = Stage_Module(conv_filters=filter_count)
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # Stage 5
        self.stage_5 = Stage_Module(mod_type='conv_pool', conv_filters=filter_count*2, pool_size=2)

        # Stage 5
        self.stage_6 = Stage_Module(mod_type='conv_pool', conv_filters=filter_count*2, pool_size=4)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*filter_count*2, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # Stage 1
        emb_A = self.stage_1_A(ft1)
        emb_B = self.stage_1_B(ft2)
        f_A = nn.functional.max_pool2d(emb_A, (2, 2))
        f_B = nn.functional.max_pool2d(emb_B, (2, 2))

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        f_A = fuse_fts(f_A, emb_A, method=self.fuse_methods)
        f_B = fuse_fts(f_B, emb_B, method=self.fuse_methods)
        f_A = nn.functional.max_pool2d(f_A, (2, 2))
        f_B = nn.functional.max_pool2d(f_B, (2, 2))

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        f_A = fuse_fts(f_A, emb_A, method=self.fuse_methods)
        f_B = fuse_fts(f_B, emb_B, method=self.fuse_methods)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        f_A = fuse_fts(f_A, emb_A, method=self.fuse_methods)
        f_B = fuse_fts(f_B, emb_B, method=self.fuse_methods)

        # Concatenate stages
        fused_stages = torch.concatenate([f_A, f_B], dim=1)

        # Stage 5
        emb = self.stage_5(fused_stages)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class VGG(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        model = vgg16()
        self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()
        self.feature_extractor = model.features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088*2, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # preprocess image
        ft1 = self.preprocess(ft1)
        ft1 = self.feature_extractor(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.feature_extractor(ft2)

        fts = torch.concatenate([ft1, ft2], dim=1)
        logits = self.classifier(fts)

        return logits


class ResNet_18(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = resnet18()
        self.model.fc = nn.Identity()
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.clf = nn.Linear(512*2, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model(ft2)

        fts = torch.concatenate([ft1, ft2], dim=1)
        logits = self.clf(fts)

        return logits


class ResNet_50_HS(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_method = fuse_methods['H']
        self.model = resnet50()
        self.model.fc = nn.Identity()
        self.preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.clf = nn.Linear(2048, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model(ft2)

        fts = fuse_fts(ft1, ft2, self.fuse_method)
        logits = self.clf(fts)

        return logits

class ResNet_50_HD(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_method = fuse_methods['H']
        self.model_f1 = resnet50()
        self.model_f1.fc = nn.Identity()
        self.model_f2 = resnet50()
        self.model_f2.fc = nn.Identity()
        self.preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.clf = nn.Linear(2048, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model_f1(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model_f2(ft2)

        fts = fuse_fts(ft1, ft2, self.fuse_method)
        logits = self.clf(fts)

        return logits


class VIT_B(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = vit_b_16()
        self.model.heads = nn.Identity()
        self.preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        self.clf = nn.Linear(768*2, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model(ft2)

        fts = torch.concatenate([ft1, ft2], dim=1)
        logits = self.clf(fts)

        return logits


class VIT_L(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = vit_l_16()
        self.model.heads = nn.Identity()
        self.preprocess = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        self.clf = nn.Linear(1024*2, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model(ft2)

        fts = torch.concatenate([ft1, ft2], dim=1)
        logits = self.clf(fts)

        return logits


class Fuse_Mk7(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.patch_size = 16
        self.hidden_dim = 1024
        self.conv_prep_layer_A = nn.Conv2d(3, self.hidden_dim, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        self.conv_prep_layer_B = copy.deepcopy(self.conv_prep_layer_A)
        self.class_token_A = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.class_token_B = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.encoder = Encoder(self.hidden_dim, 0, 0, fuse_method=fuse_methods['H'])

        self.encs = [copy.deepcopy(self.encoder).to('cuda') for x in range(24)]

        self.head = nn.Linear(self.hidden_dim, output_size)


    def forward(self, ft_1, ft_2):
        n, c, h, w = ft_1.shape
        n_h = h // self.patch_size
        n_w = w // self.patch_size
        # preprocess data initially
        prep_A = self.conv_prep_layer_A(ft_1)
        prep_B = self.conv_prep_layer_B(ft_2)

        # make sequences
        prep_A = prep_A.reshape(n, self.hidden_dim, n_h * n_w).permute(0, 2, 1)
        prep_B = prep_B.reshape(n, self.hidden_dim, n_h * n_w).permute(0, 2, 1)
        batch_class_token_A = self.class_token_A.expand(n, -1, -1)
        batch_class_token_B = self.class_token_B.expand(n, -1, -1)

        prep_A = torch.cat([batch_class_token_A, prep_A], dim=1)
        prep_B = torch.cat([batch_class_token_B, prep_B], dim=1)

        out_attn = self.encoder(prep_A, prep_B)
        for e in self.encs[:]:
            out_attn = e(out_attn, out_attn)
        out_attn = out_attn[:, 0]
        out_attn = self.head(out_attn)

        return out_attn


class Encoder(nn.Module):
    def __init__(self, hidden_dim, dropout_p, attn_dropout, fuse_method):
        super().__init__()
        self.fuse_method = fuse_method
        self.init_dropout = nn.Dropout(p=dropout_p)
        self.ln_1 = nn.LayerNorm(1024)
        self.ln_2 = nn.LayerNorm(1024)
        self.multi_head_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=16, dropout=attn_dropout, batch_first=True)
        self.mlp = MLP(hidden_dim)

    def forward(self, ft1, ft2):
        ft1 = self.ln_1(ft1)
        ft2 = self.ln_1(ft2)

        ft = fuse_fts(ft1, ft2, self.fuse_method)

        out, _ = self.multi_head_att(ft, ft, ft, need_weights=False)

        out = out + ft

        out_y = self.ln_2(out)
        out_y = self.mlp(out_y)

        return out + out_y


class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim)
        )

    def forward(self, fts):
        return self.layers(fts)


##### RESNET BASED FUSION MODELS

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicFuseBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_method: int = 1
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.fuse_method = fuse_method

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = fuse_fts(out, identity, self.fuse_method)
        out = self.relu(out)

        return out


class BottleneckFused(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_method: int = 1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.fuse_method = fuse_method

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = fuse_fts(out, identity, self.fuse_method)
        out = self.relu(out)

        return out


class resnet_f(nn.Module):
    def __init__(
        self,
        block: BasicFuseBlock,
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        fuse_method: int = 1
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], fuse_method=fuse_method)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], fuse_method=fuse_method)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], fuse_method=fuse_method)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], fuse_method=fuse_method)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: BasicFuseBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        fuse_method: int = 1
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNet_18_F(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_v = fuse_methods['V']
        self.fuse_h = fuse_methods['H']
        self.model = resnet_f(BasicFuseBlock, [2, 2, 2, 2], fuse_method=self.fuse_v)
        self.model.fc = nn.Identity()
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.clf = nn.Linear(512, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model(ft2)

        fts = fuse_fts(ft1, ft2, self.fuse_h)
        logits = self.clf(fts)

        return logits


class ResNet_50_F(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_v = fuse_methods['V']
        self.fuse_h = fuse_methods['H']
        self.model = resnet_f(BottleneckFused, [3, 4, 6, 3], fuse_method=self.fuse_v)
        self.model.fc = nn.Identity()
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.clf = nn.Linear(2048, output_size)

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft1 = self.preprocess(ft1)
        ft1 = self.model(ft1)

        ft2 = self.preprocess(ft2)
        ft2 = self.model(ft2)

        fts = fuse_fts(ft1, ft2, self.fuse_h)
        logits = self.clf(fts)

        return logits