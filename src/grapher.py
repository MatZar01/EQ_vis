import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
import yaml


class Grapher:
    def __init__(self, base_pt: str, save_pickle: bool = True, model_info: dict = None):
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.lr = []
        self.model_info = model_info
        self.save_pickle = save_pickle
        self.path = self.__set_path(base_pt)

    def __set_path(self, base_pt: str):
        date = datetime.now()
        return f'{base_pt}/{date.year}-{date.month}-{date.day}+{date.hour}:{date.minute}.png'

    def add_data(self, train_data, test_data, lr):
        self.train_loss.append(train_data[0])
        self.train_acc.append(train_data[-1])
        self.test_loss.append(test_data[0])
        self.test_acc.append(test_data[-1])
        self.lr.append(lr)

    def smooth_data(self, data):
        return savgol_filter(data, 11, 3, mode='nearest')

    def make_out_dict(self):
        out_dict = {'epoch_num': len(self.test_acc),
                    'train': {'loss': self.train_loss, 'acc': self.train_acc,
                              'loss_s': self.smooth_data(self.train_loss), 'acc_s': self.smooth_data(self.train_acc)},
                    'test': {'loss': self.test_loss, 'acc': self.test_acc,
                             'loss_s': self.smooth_data(self.test_loss), 'acc_s': self.smooth_data(self.test_acc)}}

        if self.model_info is not None:
            self.model_info.update(out_dict)
            return self.model_info
        else:
            return out_dict

    def make_graph(self):
        epochs = list(range(len(self.test_acc)))
        colors = ['steelblue', 'limegreen']
        fig, ax = plt.subplots(1, 3, figsize=(22, 4), gridspec_kw={'wspace': 0.1, 'hspace': 0.1}, sharey=False)
        ax[0].plot(epochs, self.train_loss, color=colors[0], alpha=0.2)
        ax[0].plot(epochs, self.smooth_data(self.train_loss), color=colors[0], label='train')
        ax[0].plot(epochs, self.test_loss, color=colors[1], alpha=0.2)
        ax[0].plot(epochs, self.smooth_data(self.test_loss), color=colors[1], label='test')
        ax[0].set_xlabel('Epoch')
        ax[0].grid()
        ax[0].set_title('Train/test loss')
        ax[0].legend()
        ax[1].plot(epochs, self.train_acc, color=colors[0], alpha=0.2)
        ax[1].plot(epochs, self.smooth_data(self.train_acc), color=colors[0], label='train')
        ax[1].plot(epochs, self.test_acc, color=colors[1], alpha=0.2)
        ax[1].plot(epochs, self.smooth_data(self.test_acc), color=colors[1], label='test')
        ax[1].set_xlabel('Epoch')
        ax[1].grid()
        ax[1].set_title('Train/test accuracy')
        ax[1].legend()
        ax[2].plot(epochs, self.lr, color=colors[0], label='LR')
        ax[2].grid()
        ax[2].set_xlabel('Epoch')
        ax[2].set_title('LR for Epoch')
        ax[2].legend()
        plt.savefig(self.path, bbox_inches='tight')

        if self.save_pickle:
            yaml.dump(self.make_out_dict(), open(self.path.replace('png', 'yml'), 'w'))
