import torch
from torch import nn
import copy
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from .fuse import fuse_fts


class Init_Net(nn.Module):
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
        return logits


class Comb_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
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


class Init_Net_NF(Init_Net):
    """Small_Net without feature fusion - only second branch for post disaster images"""
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__(input_size=input_size, output_size=output_size, fuse_method=fuse_method)

        self.embedder_second = None

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb = self.embedder_first(ft2)
        logits = self.classifier(emb)
        return logits


class Fuse_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
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
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_1 = nn.functional.max_pool2d(fused_1, (8, 8))

        # 2nd stage
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_2 = nn.functional.max_pool2d(fused_2, (4, 4))

        # 3rd stage
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_3 = nn.functional.max_pool2d(fused_3, (2, 2))

        # 4th stage
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_method)

        fused_final = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        logits = self.classifier(fused_final)
        return logits


class Fuse_Mk2_0(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
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
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_1 = nn.functional.max_pool2d(fused_1, (4, 4))

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_2 = nn.functional.max_pool2d(fused_2, (2, 2))

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_method)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_method)

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
                    nn.ReLU(),
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
                    nn.ReLU(),
                )
            case 'conv_downsize':
                self.module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), stride=(pool_size, pool_size), padding=(1, 1)),
                    nn.ReLU(),
                )

    def forward(self, x):
        return self.module(x)


class Fuse_Mk2(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
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
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_1 = nn.functional.max_pool2d(fused_1, (4, 4))

        # Stage 2
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_2 = nn.functional.max_pool2d(fused_2, (2, 2))

        # Stage 3
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_method)

        # Stage 4
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_method)

        # Concatenate stages
        fused_stages = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        # Stage 5
        emb = self.stage_5(fused_stages)

        # Stage 6
        emb = self.stage_6(emb)

        logits = self.classifier(emb)
        return logits


class Fuse_Net_ELU(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        # initial embedding
        self.initial_embedder_A = nn.Sequential(
            nn.BatchNorm2d(input_size[-1]),
            nn.Conv2d(input_size[-1], 16, (3, 3), 1, 1),
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.Conv2d(32, 64, (3, 3), 1, 1),
            nn.ELU(),
            nn.MaxPool2d((2, 2))
        )
        self.initial_embedder_B = copy.deepcopy(self.initial_embedder_A)

        # 1st stage embedding
        self.stage_1_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ELU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_1_B = copy.deepcopy(self.stage_1_A)

        # 2nd stage embedding
        self.stage_2_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ELU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_2_B = copy.deepcopy(self.stage_2_A)

        # 3rd stage embedding
        self.stage_3_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ELU(),
            nn.MaxPool2d((2, 2))
        )
        self.stage_3_B = copy.deepcopy(self.stage_3_A)

        # 4th stage embedding
        self.stage_4_A = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ELU(),
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
        fused_1 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_1 = nn.functional.max_pool2d(fused_1, (8, 8))

        # 2nd stage
        emb_A = self.stage_2_A(emb_A)
        emb_B = self.stage_2_B(emb_B)
        fused_2 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_2 = nn.functional.max_pool2d(fused_2, (4, 4))

        # 3rd stage
        emb_A = self.stage_3_A(emb_A)
        emb_B = self.stage_3_B(emb_B)
        fused_3 = fuse_fts(emb_A, emb_B, method=self.fuse_method)
        fused_3 = nn.functional.max_pool2d(fused_3, (2, 2))

        # 4th stage
        emb_A = self.stage_4_A(emb_A)
        emb_B = self.stage_4_B(emb_B)
        fused_4 = fuse_fts(emb_A, emb_B, method=self.fuse_method)

        fused_final = torch.concatenate([fused_1, fused_2, fused_3, fused_4], dim=1)

        logits = self.classifier(fused_final)
        return logits


class VGG_Fuse_net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.preprocess = VGG16_Weights.IMAGENET1K_FEATURES.transforms()

        self.feature_extractor = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            #nn.ReLU(),
            #nn.Linear(512, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # preprocess image
        ft1 = self.preprocess(ft1)
        ft2 = self.preprocess(ft2)

        #with torch.no_grad():
        emb_1 = self.feature_extractor(ft1)
        emb_2 = self.feature_extractor(ft2)

        fused = fuse_fts(emb_1, emb_2, self.fuse_method)
        logits = self.classifier(emb_2)

        return logits


class ResNet50_Fuse_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

        self.feature_extractor = ResNet50_extractor()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            #nn.ReLU(),
            #nn.Linear(512, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # preprocess image
        ft1 = self.preprocess(ft1)
        ft2 = self.preprocess(ft2)

        #with torch.no_grad():
        emb_1 = self.feature_extractor.extract_fts(ft1)
        emb_2 = self.feature_extractor.extract_fts(ft2)

        fused = fuse_fts(emb_1, emb_2, self.fuse_method)
        logits = self.classifier(fused)

        return logits


class ResNet18_Fuse_Net(ResNet50_Fuse_Net):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__(input_size=input_size, output_size=output_size, fuse_method=fuse_method)
        self.preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.feature_extractor = ResNet18_extractor()


class ResNet50_extractor:
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        self.init_model = resnet50(weights=weights).to('cuda')
        self.init_model.fc = nn.Identity()

    def extract_fts(self, x: torch.Tensor) -> torch.Tensor:
        return self.init_model(x)


class ResNet18_extractor:
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        self.init_model = resnet18(weights=weights).to('cuda')
        self.init_model.fc = nn.Identity()

    def extract_fts(self, x: torch.Tensor) -> torch.Tensor:
        return self.init_model(x)
