import yaml
import numpy as np
import torch

DIR = './cfgs/Res50_HS'

model_info = {
    'DATA_PATH': 'DS/xView/i_B',
    'ONEHOT_DATA': True,
    'DEVICE': 'cuda',
    'EPOCHS': 90,
    'LR': 1e-4,
    'TRAIN_SIZE': 0.8,
    'FUSE_METHODS': 8,
    'DATA_SEED': 24,
    'BATCH_SIZE': 32,
    'NORMALIZE_INPUT': True,
    'OPT': 'Adam',
    'SCHEDULER': {'NAME': 'ROP', 'PATIENCE': 2, 'FACTOR': 0.7, 'STEP': 5, 'GAMMA': 0.5},
    'MODEL_NAME': 'ResNet_50_HS',
    'LOG': True,

    'CLASS_W': [0.7298184429172223, 2.1619568520206625, 6.197735191637631, 169.4047619047619]
}
#%%
fuse_met = list(range(13))
l1 = torch.Tensor(np.array(fuse_met))
prod = torch.cartesian_prod(l1, l1).numpy().astype(int).tolist()

ex_num = 0
for m in prod:
    model_info['FUSE_METHODS'] = {'V': m[0], 'H': m[1]}
    yaml.dump(model_info, open(f'{DIR}/ex_{ex_num}.yml', 'w'))
    ex_num += 1

#%%
fuse_met = list(range(13))

ex_num = 13
for m in fuse_met:
    model_info['FUSE_METHODS'] = {'V': m, 'H': m}
    yaml.dump(model_info, open(f'{DIR}/ex_{ex_num}.yml', 'w'))
    ex_num += 1
