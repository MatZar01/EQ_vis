from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import random


class BanditData(Dataset):
    def __init__(self, A_pt: str, B_pt: str, S_pt: str, train: bool, train_inds: list, test_inds: list,
                 skip_name: bool = False, encode_name: bool = False, onehot_S: bool = False,
                 up_result: bool = False):
        self.train = train
        self.onehot = onehot_S

        A_tensor = np.genfromtxt(A_pt, delimiter=',', skip_header=1)
        B_tensor = np.genfromtxt(B_pt, delimiter=',', skip_header=1)
        S_tensor = np.genfromtxt(S_pt, delimiter=',', skip_header=1)

        if skip_name:
            A_tensor = A_tensor[:, 1:]
            B_tensor = B_tensor[:, 1:]
        else:
            if encode_name:
                a_names = np.array([[int(y) for y in np.binary_repr(x, width=5)] for x in A_tensor[:, 0].astype(int)])
                b_names = np.array([[int(y) for y in np.binary_repr(x, width=5)] for x in B_tensor[:, 0].astype(int)])
                A_tensor = np.hstack([a_names, A_tensor[:, 1:]])
                B_tensor = np.hstack([b_names, B_tensor[:, 1:]])

        if up_result: A_tensor, B_tensor, S_tensor = self.recode_results(A_tensor, B_tensor, S_tensor)

        if onehot_S: S_tensor = self.onehot_score(S_tensor)

        self.data_shape = A_tensor.shape[-1], S_tensor.shape[-1]

        if train:
            self.A_tensor = torch.Tensor(A_tensor[train_inds])
            self.B_tensor = torch.Tensor(B_tensor[train_inds])
            self.S_tensor = torch.Tensor(S_tensor[train_inds])
        else:
            self.A_tensor = torch.Tensor(A_tensor[test_inds])
            self.B_tensor = torch.Tensor(B_tensor[test_inds])
            self.S_tensor = torch.Tensor(S_tensor[test_inds])

    def __len__(self) -> int:
        return self.S_tensor.shape[0]

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if self.onehot:
            return self.A_tensor[idx], self.B_tensor[idx], self.S_tensor[idx]
        else:
            return self.A_tensor[idx], self.B_tensor[idx], self.S_tensor[idx].unsqueeze(0)

    def recode_results(self, A, B, S) -> tuple:
        A[:, -5:] += 1
        B[:, -5:] += 1
        S += 1
        return A, B, S

    def onehot_score(self, S_tensor) -> np.array:
        enc = OneHotEncoder()
        enc.fit(S_tensor.reshape(-1, 1))
        return enc.transform(S_tensor.reshape(-1, 1)).toarray().astype(int)


def get_train_test_inds(TRAIN_SIZE, ds_length):
    ind_list = list(range(ds_length))
    train_inds = random.sample(ind_list, int(TRAIN_SIZE*ds_length))
    test_inds = list(set(ind_list) - set(train_inds))
    return train_inds, test_inds
