import os
from tabnanny import verbose
import copy
import tqdm
import pandas as pd
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models.stoc.modules.stoc import STOC


class Trainer():
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
    def __init__(
        self,
        input_dim,
        output_dim=128,
        feature_size=256,
        device='cuda',
        lr=0.0001,
        batch_size=16, 
        patience = 50
        ):

        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size

        self.input_dim = input_dim
        self.feature_size = feature_size
        self.output_dim = output_dim    

        self.manualSeed = 42

        # seed 고정  
        torch.manual_seed(self.manualSeed)
        cudnn.benchmark = True

        # GPU 설정
        self.device = torch.device("cuda" if device == "cuda" else "cpu")  # select the device
        
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


    def fit(self, train_loader, valid_loader, num_epochs, verbose=True):
        """
        Train the STOC model to fit trainset. Retrun the best model which has the minimum valid loss.

        Args:
            train_loader (DataLoader): train dataloader.
            valid_loader (DataLoader): valid dataloader.
            num_epochs (int): The number of epochs. When this reaches, the training stops.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            best_model (model): trained STOC Model.

        """
        min_loss = 10e15

        # for early stopping
        self.trigger_times = 0
        self.early_stopped = False


        for epoch in range(num_epochs):
            min_loss = self._train_epoch(epoch, min_loss, train_loader, valid_loader, verbose)


        return self.best_model 


    def _train_epoch(self, epoch, min_loss, train_loader, valid_loader, verbose):
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
            x = x.to(self.device)  # batch, window_size, input_dim 
            targets = targets.to(self.device)  # batch, window_size, input_dim 

            self.optimizer.zero_grad()
            output = self.model(True, x)  # batch, window_size, input_dim 
            loss = self.criterion(output, targets)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()

        if verbose:
            print(f"Epoch #{epoch+1}: loss={epoch_loss}")

        val_loss = self.evaluate(valid_loader)

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
                x = x.to(self.device)  
                targets = targets.to(self.device)  

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
            for x in test_loader:
                x = x.to(self.device) 
                output = self.model(False, x)
                final_output = torch.cat([final_output, output], dim=0)

        final_output = np.concatenate(final_output.cpu().numpy())
        final_output_df = pd.DataFrame(final_output)
        final_output_df.columns = [f'V{i+1}' for i in range(output.shape[-1])]

        return final_output_df

                
            
