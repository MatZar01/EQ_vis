import torch
from torch import nn
import copy
from .fuse import fuse_fts

##### Imports for baseline experiments
from torchvision.models import vgg16, resnet18, resnet50
from torchvision.models import VGG16_Weights, ResNet18_Weights, ResNet50_Weights
from torchvision.models.vision_transformer import vit_b_16, vit_l_16
from torchvision.models.vision_transformer import ViT_B_16_Weights, ViT_L_16_Weights


class Init_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_methods = fuse_methods['H']
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

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_methods)

        logits = self.classifier(fused)
        return logits


class Comb_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(input_size[-1]*2),
            nn.Conv2d(input_size[-1]*2, 16, (3, 3), 1, 1),
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
        x = torch.concatenate([ft1, ft2], dim=1)
        emb = self.feature_extractor(x)
        logits = self.classifier(emb)

        return logits


class Fuse_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_methods = fuse_methods['H']
        # initial embedding
        self.initial_embedder_A = nn.Sequential(
            nn.BatchNorm2d(input_size[-1]),
            nn.Conv2d(input_size[-1], 16, (3, 3), 1, 1),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.initial_embedder_B = copy.deepcopy(self.initial_embedder_A)

        # 1st stage embedding
        self.stage_1_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_1_B = copy.deepcopy(self.stage_1_A)

        # 2nd stage embedding
        self.stage_2_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_2_B = copy.deepcopy(self.stage_2_A)

        # 3rd stage embedding
        self.stage_3_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # 4th stage embedding
        self.stage_4_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_4_B = copy.deepcopy(self.stage_4_A)

        # final classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12544, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # initial embedding
        emb_A = self.initial_embedder_A(ft1)
        emb_B = self.initial_embedder_B(ft2)

        # 1st stage
        emb_A = self.stage_1_A(emb_A)
        emb_B = self.stage_1_B(emb_B)
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_1 = nn.functional.max_pool2d(fused_1, (8, 8))

        # 2nd stage
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_2 = nn.functional.max_pool2d(fused_2, (4, 4))

        # 3rd stage
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)
        fused_3 = nn.functional.max_pool2d(fused_3, (2, 2))

        # 4th stage
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_methods)

        fused_final = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        logits = self.classifier(fused_final)
        return logits


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
                    nn.ReLU()
                )
            case 'conv_pool':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d((pool_size, pool_size))
                )
            case 'conv_upsample':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(input_filters),
                    nn.Conv2d(input_filters, conv_filters//2, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters//2, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.ReLU()
                )
            case 'conv_upsample_pool':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(input_filters),
                    nn.Conv2d(input_filters, conv_filters // 2, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters // 2, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d((pool_size, pool_size))
                )
            case 'conv_downsize':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), stride=(pool_size, pool_size), padding=(1, 1)),
                    nn.ReLU()
                )

    def forward(self, x):
        return self.module(x)


class Fuse_HV(nn.Module):  # Former Fuse_Mk4
    def __init__(self, input_size: list, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_v = fuse_methods['V']
        self.fuse_h = fuse_methods['H']
        filter_count = 64
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


class Fuse_HV_2(nn.Module):  # Former Fuse_Mk4
    def __init__(self, input_size: list, output_size: int, fuse_methods: dict):
        super().__init__()
        self.fuse_v = fuse_methods['V']
        self.fuse_h = fuse_methods['H']
        filter_count = 64
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
        self.stage_5_A = Stage_Module(conv_filters=filter_count)
        self.stage_5_B = copy.deepcopy(self.stage_5_A)

        # Stage 6
        self.stage_6 = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=2)

        # Stage 7
        self.stage_7 = Stage_Module(mod_type='conv_pool', conv_filters=filter_count, pool_size=4)

        # Stage 8
        self.stage_8 = Stage_Module(conv_filters=filter_count)

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
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_5 = fuse_fts(emb_A, emb_B, method=self.fuse_h)
        fused_5 = fuse_fts(fused_4, fused_5, method=self.fuse_v)

        # Stage 6
        emb = self.stage_6(fused_5)

        # Stage 6
        emb = self.stage_7(emb)

        # Stage 7
        emb = self.stage_8(emb)

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
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()
        self.feature_extractor = model.features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # preprocess image
        ft = self.preprocess(ft2)
        ft = self.feature_extractor(ft)
        logits = self.classifier(ft)

        return logits


class ResNet_18(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, output_size)
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft = self.preprocess(ft2)
        logits = self.model(ft)

        return logits


class ResNet_50(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(2048, output_size)
        self.preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft = self.preprocess(ft2)
        logits = self.model(ft)

        return logits


class VIT_B(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Linear(768, output_size)
        self.preprocess = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft = self.preprocess(ft2)
        logits = self.model(ft)

        return logits


class VIT_L(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_methods: dict):
        super().__init__()
        self.model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        self.model.heads = nn.Linear(1024, output_size)
        self.preprocess = ViT_L_16_Weights.IMAGENET1K_V1.transforms()

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        ft = self.preprocess(ft2)
        logits = self.model(ft)

        return logits

