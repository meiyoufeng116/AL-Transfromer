from re import L
import torch
from torch.utils.data import Dataset


class SepsisDataset(Dataset):

    def __init__(self, dataset, mask, valid_data, valid_mask, test_data, test_mask):
        # 'Initialization'
        #self.X = torch.load(path)
        #self.X = pickle.load(open(path, 'rb'))
        self.X = dataset
        self.M = mask
        self.vx = valid_data
        self.vm = valid_mask
        self.tx = test_data
        self.tm = test_mask

    def __len__(self):
        #'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        #'Generates one sample of data'
        # Select sample
        return self.X[index], self.M[index], self.vx[index], self.vm[index], self.tx[index], self.tm[index]


class TrainDataset(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]
    
class VaildDataset(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]
    
class TestDataset(Dataset):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.mask[index]