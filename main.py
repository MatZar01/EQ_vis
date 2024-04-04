import numpy as np
import torch
from src import EQ_Data
from src import Init_net, Small_net, Small_net_NF
from src import train, test
from src import Grapher
from src import verbose
from torch.utils.data import DataLoader
import importlib

B_pt = 'DS/IDA-BD/i_B'

model_info = {
    'DEVICE': 'cuda',
    'EPOCHS': 1,
    'LR': 1e-4,
    'TRAIN_SIZE': 0.75,
    'FUSE_METHOD': 2,
    'MODEL_NAME': 'Small_net_NF'
}

grapher = Grapher(base_pt='./result_graphs')

# prepare dataset
data_train = EQ_Data(B_pt, train=True, train_size=model_info['TRAIN_SIZE'], onehot=True)
data_test = EQ_Data(B_pt, train=False, train_size=model_info['TRAIN_SIZE'], onehot=True)

train_dataloader = DataLoader(data_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=64, shuffle=True)

# import models and initialize network
nets_module = importlib.import_module('src.models')
net_class = getattr(nets_module, model_info['MODEL_NAME'])
# load network
model = net_class(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1],
                  fuse_method=model_info['FUSE_METHOD']).to(model_info['DEVICE'])

#model = Init_net(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1], fuse_method=5).to(DEVICE)
#model = Small_net(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1], fuse_method=model_info['FUSE_METHOD']).to(model_info['DEVICE'])
#model = Small_net_NF(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1], fuse_method=model_info['FUSE_METHOD']).to(model_info['DEVICE'])

# initialize loss fn and optimizer
#loss = torch.nn.MSELoss()
#loss = torch.nn.HuberLoss()
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_info['LR'],
                             weight_decay=model_info['LR']/model_info['EPOCHS'], amsgrad=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=LR/EPOCHS)

# train
for e in range(model_info['EPOCHS']):
    train_res = train(train_dataloader, model, loss, optimizer, device=model_info['DEVICE'])
    test_res = test(test_dataloader, model, loss, device=model_info['DEVICE'])
    verbose(e, train_res, test_res, freq=1)
    grapher.add_data(train_data=train_res, test_data=test_res)

grapher.make_graph()
print('[INFO] Done!')
