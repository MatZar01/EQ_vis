import numpy as np
import torch
from src import BanditData
from src import Small_Net, UNet, ConvNet_2d, Encoded_ConvNet_2d, ConvNet_2d_slim
from src import train, test
from src import Grapher
from src import verbose
from torch.utils.data import DataLoader
from src import get_train_test_inds

A_pt = './ds/A.csv'
B_pt = './ds/B.csv'
S_pt = './ds/S.csv'

DEVICE = 'cuda'
EPOCHS = 800
LR = 1e-4
TRAIN_SIZE = 0.8
grapher = Grapher(base_pt='./result_graphs')


train_inds, test_inds = get_train_test_inds(TRAIN_SIZE, np.genfromtxt(A_pt, delimiter=',', skip_header=1).shape[0])

# prepare dataset
data_train = BanditData(A_pt, B_pt, S_pt, train=True, train_inds=train_inds, test_inds=test_inds,
                        skip_name=True, encode_name=True, onehot_S=True, up_result=True)
data_test = BanditData(A_pt, B_pt, S_pt, train=False, train_inds=train_inds, test_inds=test_inds,
                       skip_name=True, encode_name=True, onehot_S=True, up_result=True)

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
