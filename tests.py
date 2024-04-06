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