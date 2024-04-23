import torch
import lightning as L
import torchmetrics
import yaml


class Light_Net(L.LightningModule):
    def __init__(self, network, loss_fn, optimizer, scheduler, model_info, grapher):
        super().__init__()
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grapher = grapher
        self.model_info = model_info
        self.scheduler = scheduler

        self.loss_train = torch.Tensor([0])
        self.loss_test = torch.Tensor([0])
        self.acc_train = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4)
        self.acc_test = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4)

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "monitor": 'Loss/test', "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        if self.model_info['SCHEDULER']['NAME'] == 'ROP':
            scheduler.step(metrics=self.loss_test)
        else:
            scheduler.step()

    def on_train_start(self):
        """save model info in pkl file"""
        try:
            yaml.dump(self.model_info, open(f'{self.logger.log_dir}/model_info.yml', 'w'))
        except AttributeError:
            print("[INFO] No logging - model_info not saved")

    def network_step(self, batch):
        ft1, ft2, flag, weights, meta = batch
        logits = self.network(ft1, ft2)
        loss = self.loss_fn(logits, flag)
        preds = torch.nn.functional.one_hot(torch.argmax(torch.nn.functional.sigmoid(logits), dim=1), num_classes=4)
        return loss, preds, flag

    def training_step(self, batch, batch_idx):
        loss, preds, flag = self.network_step(batch)
        self.loss_train = loss
        self.acc_train(preds, flag)

        self.log('Acc/train/step', self.acc_train)
        self.log("Loss/train/step", self.loss_train)
        return self.loss_train

    def validation_step(self,  batch, batch_idx):
        loss, preds, flag = self.network_step(batch)
        self.loss_test = loss
        self.acc_test(preds, flag)

        return self.loss_test

    def on_train_epoch_end(self):
        self.grapher.add_data(train_data=[self.loss_train.item(), self.acc_train.compute().item()],
                              test_data=[self.loss_test.item(), self.acc_test.compute().item()],
                              lr=self.optimizer.param_groups[0]['lr'])
        self.log('Acc/train', self.acc_train.compute().item())
        self.log('Loss/train', self.loss_train)
        print(f'\n[INFO] Training -- Loss: {self.loss_train} Accuracy: {self.acc_train.compute().item()}')

    def on_validation_epoch_end(self):
        self.log('Acc/test', self.acc_test.compute().item())
        self.log('Loss/test', self.loss_test)
        self.log('LR', self.optimizer.param_groups[0]['lr'])
        print(f'\n[INFO] Testing -- Loss: {self.loss_test}, Accuracy: {self.acc_test.compute().item()}')

