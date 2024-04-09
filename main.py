import torch
from src import EQ_Data, get_train_test_idx
from src import train, test
from src import Grapher
from src import Scheduler_manager
from src import verbose, update_writer, get_args
from torch.utils.data import DataLoader
import importlib
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # get args for current  run
    model_info = get_args()
    # set models module path for auto imports
    nets_module = importlib.import_module('src.models')

    grapher = Grapher(base_pt='./result_graphs', model_info=model_info)
    writer = SummaryWriter()
    writer.add_text('model_info', str(model_info))

    # prepare dataset
    ds_idx = get_train_test_idx(model_info['TRAIN_SIZE'], model_info['DATA_PATH'], model_info['DATA_SEED'])
    data_train = EQ_Data(model_info['DATA_PATH'], train=True, onehot=True, ds_idx=ds_idx,
                         normalize=model_info['NORMALIZE_INPUT'])
    data_test = EQ_Data(model_info['DATA_PATH'], train=False, onehot=True, ds_idx=ds_idx,
                        normalize=model_info['NORMALIZE_INPUT'])

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

    if model_info['OPT'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_info['LR'],
                                     weight_decay=model_info['LR']/model_info['EPOCHS'], amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=model_info['LR'],
                                    weight_decay=model_info['LR']/model_info['EPOCHS'])

    # initialize scheduler
    scheduler = Scheduler_manager(optimizer=optimizer, scheduler_options=model_info['SCHEDULER'])

    # train
    for e in range(model_info['EPOCHS']):
        train_res = train(train_dataloader, model, loss, optimizer, device=model_info['DEVICE'])
        test_res = test(test_dataloader, model, loss, device=model_info['DEVICE'])
        verbose(e, train_res, test_res, freq=1)
        update_writer(writer=writer, epoch=e, train_res=train_res, test_res=test_res, lr=optimizer.param_groups[0]['lr'])
        grapher.add_data(train_data=train_res, test_data=test_res, lr=optimizer.param_groups[0]['lr'])

        # perform scheduler update
        scheduler.update(test_loss=test_res[0])

    grapher.make_graph()
    writer.flush()
    print('[INFO] Done!')
