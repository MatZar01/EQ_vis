from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import random
from imutils import paths
import cv2


class EQ_Data(Dataset):
    def __init__(self, B_pt: str, train: bool, onehot: bool, ds_idx: dict):
        self.train = train
        self.onehot = onehot
        train_idx, test_idx = ds_idx['train'], ds_idx['test']

        B_pts = list(paths.list_images(B_pt))
        A_pts = [f'{pt[:-6].replace("i_B", "i_A")}.png' for pt in B_pts]
        S = np.array([int(x.split('_')[-1].split('.')[0]) for x in B_pts])

        if onehot: S = self.onehot_score(S)

        self.data_shape = cv2.imread(A_pts[0], -1).shape, S.shape[-1]

        if self.train:
            self.A_pts = [A_pts[i] for i in train_idx]
            self.B_pts = [B_pts[i] for i in train_idx]
            self.S = [S[i] for i in train_idx]
        else:
            self.A_pts = [A_pts[i] for i in test_idx]
            self.B_pts = [B_pts[i] for i in test_idx]
            self.S = [S[i] for i in test_idx]

    def __len__(self) -> int:
        return len(self.A_pts)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor, dict):
        A = torch.Tensor(cv2.imread(self.A_pts[idx], -1)).permute(2, 0, 1)
        B = torch.Tensor(cv2.imread(self.B_pts[idx], -1)).permute(2, 0, 1)
        S = torch.Tensor(self.S[idx])

        meta = {'A': {'im': cv2.imread(self.A_pts[idx], -1), 'pt': self.A_pts[idx]},
                'B': {'im': cv2.imread(self.B_pts[idx], -1), 'pt': self.B_pts[idx]},
                'S': {'val': self.S[idx]}
                }

        if self.onehot:
            return A, B, S, meta
        else:
            return A, B, S[idx].unsqueeze(0), meta

    def onehot_score(self, S_tensor) -> np.array:
        enc = OneHotEncoder()
        enc.fit(S_tensor.reshape(-1, 1))
        return enc.transform(S_tensor.reshape(-1, 1)).toarray().astype(int)


def get_train_test_idx(TRAIN_SIZE: float, ds_path: str, seed: int = None) -> dict:
    # setup seed
    if seed is not None:
        random.seed(seed)

    # get DS idx
    ind_list = list(range(len(list(paths.list_images(ds_path)))))
    train_idx = random.sample(ind_list, int(TRAIN_SIZE * len(ind_list)))
    test_idx = list(set(ind_list) - set(train_idx))
    return {'train': train_idx, 'test': test_idx}
