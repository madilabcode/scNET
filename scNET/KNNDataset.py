import torch
from torch.utils.data import Dataset
import numpy as np

class KNNDataset(Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index.T

    def __len__(self):
        return self.edge_index.shape[0]

    def __getitem__(self, idx):
        return self.edge_index[idx,:]

            
class CellDataset(Dataset):
    def __init__(self, x, knn):
        self.x = x
        self.knn = knn


    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        return self.x[:,idx] , idx

    
class CustomDataset(Dataset):
    def __init__(self, x):
        self.data = x
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index])