import torch
import numpy as np
from torch.utils.data import Dataset


class BuildDataset(Dataset):
    def __init__(self, parameter, x, overlap):
        """ 
        Make a dataset to train or encode

        Args:
            parameter (dict): The dictionary contain 
            x (array): The input time series 
            overlap (bool): Whether to maker overlaped time windows

        """

        self.x = np.concatenate(x.transpose(0, 2, 1)) # seq_len x input_dim

        self.overlap = overlap
        self.window_size = parameter['window_size']
        self.forecast_step = parameter['forecast_step']

        self.start_point = range(0, self.x.shape[0]-self.window_size-self.forecast_step)
        self.nonoverlap_start_point = range(0, self.x.shape[0]-self.window_size, self.window_size)
        
    def __len__(self):
        if self.overlap:
            return len(self.start_point)
        else:
            return len(self.nonoverlap_start_point)
            
    def __getitem__(self, idx):
        if self.overlap:
            return torch.FloatTensor(self.x[self.start_point[idx]:self.start_point[idx]+self.window_size, :]), torch.FloatTensor(self.x[self.start_point[idx]+self.forecast_step:self.start_point[idx]+self.forecast_step+self.window_size, :])
        else:
            return torch.FloatTensor(self.x[self.nonoverlap_start_point[idx]:self.nonoverlap_start_point[idx]+self.window_size, :])