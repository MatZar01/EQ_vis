from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR


class Scheduler_manager:
    """Manage different schedulers"""
    def __init__(self, optimizer, scheduler_options: dict):
        self.name = scheduler_options['NAME']
        self.optimizer = optimizer

        if self.name is None:
            print('[INFO] No scheduler selected')
            self.scheduler = None
        elif self.name == 'ROP':
            self.scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=scheduler_options['PATIENCE'],
                                               factor=scheduler_options['FACTOR'])
        elif self.name == 'STEPLR':
            self.scheduler = StepLR(optimizer=optimizer, step_size=scheduler_options['STEP'],
                                    gamma=scheduler_options['GAMMA'])
        else:
            print('[INFO] Unknown scheduler selected, omitting')
            self.scheduler = None

    def update(self, train_loss):
        """Update scheduler and communicate LR update"""
        init_lr = self.optimizer.param_groups[0]['lr']

        if self.name == 'ROP':
            self.scheduler.step(train_loss)
        elif self.name == 'STEPLR':
            self.scheduler.step()
        else:
            pass

        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr != init_lr:
            print(f'[INFO] LR updated to {current_lr}')
