from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR


class Scheduler_manager:
    """Manage different schedulers"""
    def __init__(self, optimizer, model_info: dict):
        scheduler_options = model_info['SCHEDULER']
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
        elif self.name == 'ONECYCLE':
            self.scheduler = OneCycleLR(optimizer=optimizer, max_lr=model_info['LR'], total_steps=model_info['EPOCHS'])
        else:
            print('[INFO] Unknown scheduler selected, omitting')
            self.scheduler = None
