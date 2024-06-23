import torchvision

from torch.utils.data import DataLoader
from src import EQ_Data
import torch
import matplotlib.pyplot as plt

B_pt = 'DS/IDA-BD/i_B'

data_train = EQ_Data(B_pt, train=True, train_size=0.8, onehot=True, seed=42)

a, b, s, meta = next(iter(data_train))
plt.imshow(meta['A']['im'])
plt.show()
plt.imshow(meta['B']['im'])
plt.show()
#%%
train_dataloader = DataLoader(data_train, batch_size=4, shuffle=False)
data = next(iter(train_dataloader))
#%%
import numpy as np
bsize = a.shape[0]
out = []
for i in range(bsize):
    a_b = a[i, :, :]
    b_b = b[i, :, :]
    res = torch.mm(a_b, b_b)
    out.append(res)

out = torch.tensor(torch.stack(out)).unsqueeze(dim=1)
#%%
import numpy as np
import cv2

im_1 = cv2.imread('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/PRJ-3563/masks/AOI3-tile_6-5_post_disaster.png', -1)
im_2 = cv2.imread('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/PRJ-3563/masks/AOI3-tile_6-5_pre_disaster.png', -1)//255

im_1 = np.where(im_1 != 0, 1, 0)

i_ou = im_2 - im_1
o = np.unique(i_ou)
print(o)
#%%
def iii(a, b):
    print(a)
    if a != b:
        print(0)
    else:
        pass
    print('end')

iii(1, 1)
#%%
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
from torchsummary import summary
from torch import Tensor
preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
from typing import Any, Callable, List, Optional


class Res_Net_extractor:
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        self.init_model = resnet50(weights=weights).to('cuda')
        self.conv1 = None

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

        return x


feature_extractor = Res_Net_extractor(weights=ResNet50_Weights.IMAGENET1K_V2).to('cuda')
feature_extractor.fc = None
summary(feature_extractor, (3, 244, 244))
#%%
import torch
from torch import nn

input = torch.randn((3, 224, 224))
input2 = torch.randn((1, 3, 224, 224))
l = nn.Conv2d(3, 3, (3, 3), stride=(4, 4), padding=(1, 1))
out = l(input)
conv_filters = 3
pool_size = 4
module = nn.Sequential(
                    nn.BatchNorm2d(conv_filters),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), 1, 1),
                    nn.Conv2d(conv_filters, conv_filters, (3, 3), stride=(pool_size, pool_size), padding=(1, 1)),
                    nn.ReLU(),
                )
out2 = module(input2)
#%%
import json
from imutils import paths

ftypes = []
dmgs = []
pts = list(paths.list_files('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/xView2/geotiffs/hold/labels'))
pts = [x for x in pts if 'post' in x]
for p in pts:
    d = json.load(open(p, 'r'))
    for o in d['features']['xy']:
        ftypes.append(o['properties']['feature_type'])
        dmgs.append(o['properties']['subtype'])
#%%
import numpy as np
f = np.array(ftypes)
d = np.array(dmgs)
ds = np.unique(d, return_counts=True)
fs = np.unique(f, return_counts=True)
#%%
import cv2
import json
import numpy as np
from imutils import paths
from tqdm import tqdm

im_pts = list(paths.list_files('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/xV/geotiffs/images'))
im_pts = [x for x in im_pts if '_post' in x]

for i in tqdm(range(len(im_pts))):
    p = im_pts[i]
    lb_p = p.replace('images', 'masks').replace('post', 'pre').replace('.tif', '.png')
    im = cv2.imread(p, -1)
    lbl = cv2.imread(lb_p, -1)
    if lbl is None:
        continue
    unqs = np.unique(lbl)
    if len(unqs) > 2:
        print(unqs)
        cv2.imwrite('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/xV/geotiffs/image.tif', im.astype('uint8'))
        cv2.imwrite('/home/mateusz/Desktop/EQ_vis/data/IDA_BD/xV/geotiffs/mask.tif', lbl * 60)
        break
#%%
import yaml
from imutils import paths
DIR = '/home/mateusz/Desktop/EQ_vis/results/results_xView'
pts = list(paths.list_files(DIR))

max_tests = []
fuses = []
for p in pts:
    data = yaml.load(open(p, 'r'), Loader=yaml.Loader)
    max_t = max(data['test']['acc'])
    fuse = data['FUSE_METHODS']
    max_tests.append(max_t)
    fuses.append(fuse)
#%%
Z = [x for _, x in sorted(zip(max_tests, fuses), key=lambda pair: pair[0])][::-1]
i = fuses.index(Z[0])
print(Z[:3])
