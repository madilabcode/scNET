
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
    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index


    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        isin_result  = torch.isin(self.edge_index, idx)
        columns_to_select = isin_result.all(dim=0)

        return self.x[:,idx] , idx