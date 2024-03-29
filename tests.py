from torch.utils.data import DataLoader
from src import EQ_Data
import torch
import matplotlib.pyplot as plt

A_pt = 'DS/IDA-BD/i_A'
B_pt = 'DS/IDA-BD/i_B'

data_train = EQ_Data(B_pt, train=True, train_size=0.8, onehot=True)

a, b, s, meta = next(iter(data_train))
plt.imshow(meta['A']['im'])
plt.show()
plt.imshow(meta['B']['im'])
plt.show()
#%%
train_dataloader = DataLoader(data_train, batch_size=4, shuffle=True)
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
