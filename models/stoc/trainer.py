import os
from tabnanny import verbose
import copy
import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn

import sys
sys.path.append('./models/stoc')

from stoc import STOC


class Trainer_STOC():
    """
    Initialize a trainer.

    Args:
        input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
        output_dim (int): The representation dimension.
        feature_size (int): The dimension of the Transformer encoder layer used to encode input time series.
        device (int): The gpu used for training and inference.
        lr (int): The learning rate.
        batch_size (int): The batch size.
        patience (int): The number of epochs with no improvement after which training will be stopped.
    """
    def __init__(self, config):
        self.lr = config['lr']
        self.patience = config['patience']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']

        self.input_dim = config['input_dim']
        self.feature_size = config['feature_size']
        self.output_dim = config['output_dim']
        
        # GPU 설정
        self.device = config['device']
        
        # model, optimier, scheduler, criterion 초기화
        self.init_model()


    def init_model(self):
        """
        Initialize mode, optimizer, scheduler, criterion

        """
        self.model = STOC(self.input_dim, self.feature_size, self.output_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0.00001)
        self.criterion = nn.MSELoss()


    def fit(self, train_loader, valid_loader):
        """
        Train the STOC model to fit trainset. Retrun the best model which has the minimum valid loss.

        Args:
            train_loader (DataLoader): train dataloader.
            valid_loader (DataLoader): valid dataloader.
            
        Returns:
            best_model (model): trained STOC Model.

        """
        min_loss = 10e15

        # for early stopping
        self.trigger_times = 0
        self.early_stopped = False


        for epoch in range(self.num_epochs):
            min_loss = self._train_epoch(epoch, min_loss, train_loader, valid_loader)

        return self.best_model 


    def _train_epoch(self, epoch, min_loss, train_loader, valid_loader):
        """
        Train the STOC model during 1 epoch. 
        
        Args:
            epoch (int): 
            min_loss (int): The minimum epoch loss until previous epoch
            train_loader (DataLoader): train dataloader.
            valid_loader (DataLoader): valid dataloader.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            min_loss (int): The minimum epoch loss after this epoch

        """
        self.model.train()
        epoch_loss = 0

        for x, targets in train_loader:
            x = x.transpose(1, 2).to(self.device)  # batch, window_size, input_dim 
            targets = targets.transpose(1, 2).to(self.device)  # batch, forecast_step, input_dim 

            self.optimizer.zero_grad()
            output = self.model(True, x)  # batch, window_size, input_dim 
            loss = self.criterion(output, targets)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()

        print(f"Epoch #{epoch + 1}: loss={epoch_loss}")

        val_loss = self.evaluate(valid_loader)
        print(f"Epoch #{epoch + 1}: validation loss={val_loss}\n")

        if val_loss >= min_loss:
            self.trigger_times += 1
            if self.trigger_times >= self.patience:
                self.early_stopped = True
                return min_loss

        elif val_loss < min_loss:
            self.trigger_times = 0
            self.best_model = copy.deepcopy(self.model)
            min_loss = val_loss

        return min_loss


    def evaluate(self, valid_loader):
        """
        Evaluate the trained model about valid dataset

        Args:
            valid_loader (DataLoader): valid dataloader.

        Returns:
            eval_loss (int): The evaluation loss about valid dataset
        """

        self.model.eval()

        eval_loss = 0.
        with torch.no_grad():
            for x, targets in valid_loader:
                x = x.transpose(1, 2).to(self.device)  
                targets = targets.transpose(1, 2).to(self.device)  

                output = self.model(True, x)
                eval_loss += self.criterion(output, targets).item()

        return eval_loss

    def encode(self, test_loader):
        """ 
        Compute representations using the model.
        
        Args:
            test_loader (DataLoader): test dataset loader.

        Returns:
            output (dataframe): Output representation vector.
        """

        self.model.eval()

        final_output = torch.Tensor().to(self.device) 
        with torch.no_grad():
            for x, _ in test_loader:
                if len(x.shape) != 3:
                    x = x[0]
                x = x.transpose(1, 2).to(self.device)
                
                output = self.model(False, x)
                final_output = torch.cat([final_output, output], dim=0)
        
        final_output = F.max_pool1d(
            final_output.transpose(1, 2),
            kernel_size=final_output.size(1),
        ).transpose(1, 2)
        final_output = final_output.squeeze(1)

        final_output_df = pd.DataFrame(final_output.cpu().numpy())
        final_output_df.columns = [f'V{i+1}' for i in range(final_output.shape[-1])]
        return final_output_df
