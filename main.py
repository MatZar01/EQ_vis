import torch
from src import EQ_Data, get_train_test_idx
from src import Light_Net
from src import Grapher
from src import Scheduler_manager
from src import verbose, get_args
from src import MSE_weighted
from src import MSE_w
from torch.utils.data import DataLoader
import importlib
import lightning as L


if __name__ == '__main__':
    # get args for current  run
    model_info = get_args()
    grapher = Grapher(base_pt='./result_graphs', model_info=model_info)
    # set models module path for auto imports
    nets_module = importlib.import_module('src.models')

    # prepare dataset
    ds_idx = get_train_test_idx(model_info['TRAIN_SIZE'], model_info['DATA_PATH'], model_info['DATA_SEED'])
    data_train = EQ_Data(model_info['DATA_PATH'], train=True, onehot=model_info['ONEHOT_DATA'], ds_idx=ds_idx,
                         normalize=model_info['NORMALIZE_INPUT'], weights=model_info['CLASS_W'])
    data_val = EQ_Data(model_info['DATA_PATH'], train=False, onehot=model_info['ONEHOT_DATA'], ds_idx=ds_idx,
                       normalize=model_info['NORMALIZE_INPUT'], weights=model_info['CLASS_W'])

    train_dataloader = DataLoader(data_train, batch_size=model_info['BATCH_SIZE'], shuffle=True, num_workers=15)
    val_dataloader = DataLoader(data_val, batch_size=model_info['BATCH_SIZE'], shuffle=False, num_workers=15)

    # initialize model
    net_class = getattr(nets_module, model_info['MODEL_NAME'])
    model = net_class(input_size=data_train.data_shape[0], output_size=data_train.data_shape[1],
                      fuse_methods=model_info['FUSE_METHODS']).to(model_info['DEVICE'])

    # initialize loss fn and optimizer
    loss = torch.nn.MSELoss()
    #loss = MSE_w()
    #loss = torch.nn.NLLLoss(weight=torch.Tensor(model_info['CLASS_W']))
    #loss = torch.nn.HuberLoss()
    #loss = torch.nn.BCELoss()


    if model_info['OPT'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_info['LR'],
                                     weight_decay=model_info['LR']/model_info['EPOCHS'], amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=model_info['LR'],
                                    weight_decay=model_info['LR']/model_info['EPOCHS'])

    # initialize scheduler
    scheduler_manager = Scheduler_manager(optimizer=optimizer, model_info=model_info)

    light_model = Light_Net(network=model, loss_fn=loss, optimizer=optimizer, scheduler=scheduler_manager.scheduler,
                            model_info=model_info, grapher=grapher)

    lightning_trainer = L.Trainer(accelerator=model_info['DEVICE'], max_epochs=model_info['EPOCHS'],
                                  limit_train_batches=100, limit_val_batches=100,
                                  check_val_every_n_epoch=1, log_every_n_steps=20,
                                  enable_progress_bar=True, logger=model_info['LOG'])

    # train
    lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    grapher.save_data()
    #grapher.make_graph()
    print('[INFO] Done!')
