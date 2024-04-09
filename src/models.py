import torch
from torch import nn
import copy
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights


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
    else:
        raise NotImplementedError


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
        #self.embedder_second = copy.deepcopy(self.embedder_first)

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
        emb_2 = self.embedder_first(ft2)

        fused = fuse_fts(emb_1, emb_2, method=self.fuse_method)

        logits = self.classifier(fused)
        return torch.sigmoid(logits)


class Init_Net_NF(Init_Net):
    """Small_Net without feature fusion - only second branch for post disaster images"""
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__(input_size=input_size, output_size=output_size, fuse_method=fuse_method)

        self.embedder_second = None

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        emb = self.embedder_first(ft2)
        logits = self.classifier(emb)
        return torch.sigmoid(logits)


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

        logits = self.classifier(fused_4)
        return torch.sigmoid(logits)


class VGG_Fuse_net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.preprocess = VGG16_Weights.IMAGENET1K_FEATURES.transforms()

        self.feature_extractor = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, output_size),
            #nn.ReLU(),
            #nn.Linear(4096, 512),
            #nn.ReLU(),
            #nn.Linear(512, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # preprocess image
        #ft1 = self.preprocess(ft1)
        ft2 = self.preprocess(ft2)

        with torch.no_grad():
            #emb_1 = self.feature_extractor(ft1)
            emb_2 = self.feature_extractor(ft2)

        #fused = fuse_fts(emb_1, emb_2, self.fuse_method)
        logits = self.classifier(emb_2)

        return torch.sigmoid(logits)


class ResNet_Fuse_Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, fuse_method: int):
        super().__init__()
        self.fuse_method = fuse_method
        self.preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()

        self.feature_extractor = Res_Net_extractor()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048*7*7, output_size),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, ft1: torch.Tensor, ft2: torch.Tensor) -> torch.Tensor:
        # preprocess image
        ft1 = self.preprocess(ft1)
        ft2 = self.preprocess(ft2)

        with torch.no_grad():
            emb_1 = self.feature_extractor.extract_fts(ft1)
            emb_2 = self.feature_extractor.extract_fts(ft2)

        fused = fuse_fts(emb_1, emb_2, self.fuse_method)
        logits = self.classifier(fused)

        return torch.sigmoid(logits)


class Res_Net_extractor:
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        self.init_model = resnet50(weights=weights).to('cuda')
        self.conv1 = self.init_model.conv1
        self.bn1 = self.init_model.bn1
        self.relu = self.init_model.relu
        self.maxpool = self.init_model.maxpool
        self.layer1 = self.init_model.layer1
        self.layer2 = self.init_model.layer2
        self.layer3 = self.init_model.layer3
        self.layer4 = self.init_model.layer4
        self.avgpool = self.init_model.avgpool

    def extract_fts(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
