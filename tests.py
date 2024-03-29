from torch.utils.data import DataLoader
from src import BanditData
import torch

A_pt = './ds/A.csv'
B_pt = './ds/B.csv'
S_pt = './ds/S.csv'

data_train = BanditData(A_pt, B_pt, S_pt, train=True, train_size=0.8, skip_name=False)

a = torch.tensor(data_train.A_tensor[0:4, :]).unsqueeze(dim=2)
b = torch.tensor(data_train.B_tensor[0:4, :]).unsqueeze(dim=1)
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
