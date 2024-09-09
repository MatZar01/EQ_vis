import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from typing import Any, Dict
import torch


class Augmenter:
    def __init__(self, aug_dict: dict, shape, train):
        self.aug_dict = aug_dict
        self.shape = shape
        self.train = train
        self.augments_train = self.compose_augments_train()
        self.augments_test = self.compose_augments_test()

    def apply_augs(self, image):
        image = torch.Tensor(image).permute(2, 0, 1)

        if self.train:
            out = self.augments_train(image)
        else:
            out = self.augments_test(image)
        return out

    def compose_augments_train(self):
        transforms = v2.Compose([
            v2.RandomResizedCrop(size=self.shape, antialias=True),
            #v2.Resize(size=self.shape),
            v2.RandomHorizontalFlip(p=self.aug_dict['FLIP']),
            v2.RandomRotation(self.aug_dict['ROTATION']),
            v2.ToDtype(torch.float32, scale=True),
            Gauss(self.aug_dict['GAUSS']),
            v2.GaussianBlur((5, 5), (0.1, 1.2)),
            v2.ColorJitter(0.3, 0.3, 0.3, 0.3),
            v2.RandomPerspective(),
            v2.RandomAffine(0.2),
            v2.Normalize(mean=self.aug_dict['MEAN'], std=self.aug_dict['STD']),
        ])
        return transforms

    def compose_augments_test(self):
        transforms = v2.Compose([
            v2.Resize(size=(self.shape[0] + 32, self.shape[1] + 32), antialias=True),
            v2.CenterCrop(size=self.shape),                             # inference resize aug
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.aug_dict['MEAN'], std=self.aug_dict['STD']),
        ])
        return transforms


class Gauss(Transform):
    def __init__(self, params):
        super().__init__()
        self.s_min = params[0]
        self.s_max = params[1]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        assert isinstance(inpt, torch.Tensor)
        dtype = inpt.dtype
        if not inpt.is_floating_point():
            inpt = inpt.to(torch.float32)

        sigma = np.random.uniform(self.s_min, self.s_max)

        inpt = inpt + sigma * torch.randn_like(inpt)

        if inpt.dtype != dtype:
            inpt = inpt.to(dtype)

        return inpt


class CutMix_with_prob:
    def __init__(self, num_classes: int, p: float = 0.25):
        self.p = p
        self.cutmix = v2.CutMix(num_classes=num_classes)

    def transform(self, batch):
        im, lbl, meta = batch
        random_idx = torch.Tensor(np.random.choice(list(range(im.shape[0])), size=int(im.shape[0] * self.p))).to(int)
        ims = im[random_idx]
        labs = lbl[random_idx]
        labs_int = torch.argmax(labs, dim=1)
        new_ims, new_labs = self.cutmix(ims, labs_int)
        im[random_idx] = new_ims
        lbl[random_idx] = new_labs
        return im, lbl, meta
