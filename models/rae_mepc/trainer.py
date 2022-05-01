import os
import time
import copy
import random
import argparse
import warnings
import pandas as pd

import torch
import torch.nn.functional as F

import sys
sys.path.append('./models/rae_mepc')

from model import *

warnings.filterwarnings("ignore")


class Trainer_RAE_MEPC:
    def __init__(self, args):
        self.args = args
        self.device = args['device']

        self.decoding_mask()

        # Load Model
        self.model = RAE_MEPC(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=1e-6)

    def fit(self, train_loader, valid_loader):
        # Start training
        best_model = copy.deepcopy(self.model)
        best_valid_loss = 1000000000
        
        for epoch in range(self.args['num_epoch']):
            # Train
            train_loss = self.model_train(train_loader)
            print(f"Epoch #{epoch + 1}: train loss={train_loss}")
            
            # Validate
            valid_loss = self.model_evaluate(valid_loader)
            print(f"Epoch #{epoch + 1}: validation loss={valid_loss}\n")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)
        return best_model

    def encode(self, test_loader):
        """ 
        Compute representations using the model.
        
        Args:
            test_loader (DataLoader): test dataset loader.

        Returns:
            output (dataframe): Output representation vector.
        """

        self.model.eval()

        with torch.no_grad():
            output = []
            for data, pred_data in test_loader:
                data = data.transpose(1, 2).to(self.device)
                pred_data = pred_data.transpose(1, 2).to(self.device)

                _, _, _, out = self.model(data, pred_data, mode="test")
                output.append(out[0][0])

            output = torch.cat(output, dim=0)

        output = pd.DataFrame(output.cpu().numpy())
        output.columns = [f'V{i + 1}' for i in range(output.shape[-1])]
        return output
            
    def model_train(self, train_loader):
        total_loss = []
        self.model.train()

        for batch_idx, (data, pred_data) in enumerate(train_loader):
            data = data.transpose(1, 2).to(self.device)
            pred_data = pred_data.transpose(1, 2).to(self.device)

            loss, _, _, _ = self.model(data, pred_data)
            total_loss.append(loss.item())

            # optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def model_evaluate(self, valid_loader):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, pred_data) in enumerate(valid_loader):
                data = data.transpose(1, 2).to(self.device)
                pred_data = pred_data.transpose(1, 2).to(self.device)

                loss, _, _, _ = self.model(data, pred_data, mode='test')
                total_loss.append(loss.item())

        total_loss = torch.tensor(total_loss).mean()
        return total_loss
    
    def decoding_mask(self):
        self.args['all_decode_masks'] = []
        for i in range(self.args['dec_nlayers']):
            mask_i = np.random.randint(1, high=4, size=self.args['window_length'], dtype='l')
            temp_mask = []
            for j in mask_i:
                temp_mask.append([1, 0])
            self.args['all_decode_masks'].append(temp_mask)

        ratios = [1 / (self.args['tau'] ** (self.args['dec_nlayers'] - i - 1)) for i in range(self.args['dec_nlayers'])]
        self.args['dec_Ls'] = np.random.randint(1, high=10 + 1, size=self.args['dec_nlayers'], dtype='l')

        self.args['dec_lengths'] = []
        for i in range(self.args['dec_nlayers']):
            self.args['dec_lengths'].append(int(self.args['window_length'] * ratios[i]))
