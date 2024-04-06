import numpy as np
import torch
from src import EQ_Data
from src import train, test
from src import Grapher
from src import verbose
from torch.utils.data import DataLoader
import importlib

# set models module path for auto imports
nets_module = importlib.import_module('src.models')

B_pt = 'DS/IDA-BD/i_B'

model_info = {
    'DEVICE': 'cuda',
    'EPOCHS': 30,
    'LR': 1e-4,
    'TRAIN_SIZE': 0.75,
    'FUSE_METHOD': 2,
    'DATA_SEED': None,
    'BATCH_SIZE': 64,
    'OPT': 'Adam',
    'ROP': {'on': False, 'pat': 5, 'fac': 0.8},
    'MODEL_NAME': 'Small_net'
}

grapher = Grapher(base_pt='./result_graphs', model_info=model_info)

# prepare dataset
data_train = EQ_Data(B_pt, train=True, train_size=model_info['TRAIN_SIZE'], onehot=True, seed=model_info['DATA_SEED'])
data_test = EQ_Data(B_pt, train=False, train_size=model_info['TRAIN_SIZE'], onehot=True, seed=model_info['DATA_SEED'])

train_dataloader = DataLoader(data_train, batch_size=model_info['BATCH_SIZE'], shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=model_info['BATCH_SIZE'], shuffle=True)

# initialize model
net_class = getattr(nets_module, model_info['MODEL_NAME'])
model = net_class(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1],
                  fuse_method=model_info['FUSE_METHOD']).to(model_info['DEVICE'])

# initialize loss fn and optimizer
#loss = torch.nn.MSELoss()
#loss = torch.nn.HuberLoss()
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_info['LR'], weight_decay=model_info['LR']/model_info['EPOCHS'], amsgrad=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=LR/EPOCHS)

# train
for e in range(model_info['EPOCHS']):
    train_res = train(train_dataloader, model, loss, optimizer, device=model_info['DEVICE'])
    test_res = test(test_dataloader, model, loss, device=model_info['DEVICE'])
    verbose(e, train_res, test_res, freq=1)
    grapher.add_data(train_data=train_res, test_data=test_res, lr=optimizer.param_groups[0]['lr'])

grapher.make_graph()
print('[INFO] Done!')
