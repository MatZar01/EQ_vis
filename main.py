import numpy as np
import torch
from src import EQ_Data
from src import Small_Net
from src import train, test
from src import Grapher
from src import verbose
from torch.utils.data import DataLoader
from src import get_train_test_inds

B_pt = 'DS/IDA-BD/i_B'

DEVICE = 'cuda'
EPOCHS = 800
LR = 1e-4
TRAIN_SIZE = 0.75
grapher = Grapher(base_pt='./result_graphs')

# prepare dataset
data_train = EQ_Data(B_pt, train=True, train_size=TRAIN_SIZE, onehot=True)
data_test = EQ_Data(B_pt, train=False, train_size=TRAIN_SIZE, onehot=True)

train_dataloader = DataLoader(data_train, batch_size=4, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=4, shuffle=True)

# load network
model = Small_Net(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1], fuse_method=5).to(DEVICE)
#model = ConvNet_2d(output_size=data_train.data_shape[1], fuse_method=5).to(DEVICE)
#model = Encoded_ConvNet_2d(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1], fuse_method=5).to(DEVICE)
#model = ConvNet_2d_slim(output_size=data_train.data_shape[1]).to(DEVICE)

# initialize loss fn and optimizer
#loss = torch.nn.MSELoss()
#loss = torch.nn.HuberLoss()
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=LR/EPOCHS, amsgrad=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=LR/EPOCHS)

# train
for e in range(EPOCHS):
    train_res = train(train_dataloader, model, loss, optimizer, device=DEVICE)
    test_res = test(test_dataloader, model, loss, device=DEVICE)
    verbose(e, train_res, test_res)
    grapher.add_data(train_data=train_res, test_data=test_res)

grapher.make_graph()
print('[INFO] Done!')
