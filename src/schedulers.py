from torch.optim.lr_scheduler import ReduceLROnPlateau


class Scheduler_manager:
    """Manage different schedulers"""
    def __init__(self, optimizer, scheduler_options: dict):
        self.name = scheduler_options['NAME']
        self.optimizer = optimizer

        if scheduler_options['NAME'] is None:
            print('[INFO] No scheduler selected')
            self.scheduler = None
        elif scheduler_options['NAME'] == 'ROP':
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=scheduler_options['PAT'],
                                               factor=scheduler_options['FAC'])
        else:
            print('[INFO] Unknown scheduler selected, omitting')
            self.scheduler = None

    def update(self, epoch, train_loss):
        """Update scheduler and communicate LR update"""
        init_lr = self.optimizer.param_groups[0]['lr']

        if self.name == 'ROP':
            self.scheduler.step(train_loss)
        else:
            pass

        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr != init_lr:
            print(f'[INFO] LR updated to {current_lr}')
