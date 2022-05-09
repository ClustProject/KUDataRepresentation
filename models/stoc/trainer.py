import copy
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

import sys
sys.path.append('./models/stoc')

from stoc import STOC


class Trainer_STOC():
    def __init__(self, config):
        """
        Initialize Trainer_RAE_MEPC class and prepare model and optimizer for training.

        :param config: configuration
        :type config: dictionary
        """

        self.lr = config['lr']
        self.patience = config['patience']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']

        self.input_dim = config['input_dim']
        self.feature_size = config['feature_size']
        self.output_dim = config['output_dim']
        self.window_size = config['window_size']
        
        # GPU 설정
        self.device = config['device']
        
        # model, optimizer, scheduler, criterion 초기화
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
        Train the STOC model

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: trained STOC model
        :rtype: model
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
        Train STOC model for one epoch

        :param epoch: current epoch
        :type epoch: int

        :param min_loss: minimum epoch loss until previous epoch
        :type min_loss: float

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: the minimum epoch loss after this epoch
        :rtype: float
        """

        self.model.train()
        epoch_loss = 0

        for x, targets in train_loader:
            x = x.transpose(1, 2).to(self.device)  # shape=(batch, window_size, input_dim)
            targets = targets.transpose(1, 2).to(self.device)  # shape=(batch, forecast_step, input_dim)

            self.optimizer.zero_grad()

            # get loss
            output = self.model(True, x)  # shape=(batch, window_size, input_dim)
            loss = self.criterion(output, targets)

            # backward propagation
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()

        print(f"Epoch #{epoch + 1}: loss={epoch_loss / len(train_loader)}")

        # validation
        val_loss = self.evaluate(valid_loader)
        print(f"Epoch #{epoch + 1}: validation loss={val_loss / len(valid_loader)}\n")

        # update minimum loss
        if val_loss >= min_loss:
            self.trigger_times += 1
            if self.trigger_times >= self.patience:
                self.early_stopped = True
                return min_loss

        # update best model
        elif val_loss < min_loss:
            self.trigger_times = 0
            self.best_model = copy.deepcopy(self.model)
            min_loss = val_loss
        return min_loss

    def evaluate(self, valid_loader):
        """
        Evaluate the STOC model

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: validation loss
        :rtype: float
        """

        self.model.eval()

        eval_loss = 0.
        with torch.no_grad():
            for x, targets in valid_loader:
                x = x.transpose(1, 2).to(self.device)  
                targets = targets.transpose(1, 2).to(self.device)  

                # get loss
                output = self.model(True, x)
                eval_loss += self.criterion(output, targets).item()
        return eval_loss

    def encode(self, test_loader):
        """
        Encode raw data to representation using trained STOC model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: representation vectors
        :rtype: dataFrame
        """

        self.model.eval()
        with torch.no_grad():
            output = []
            for data in test_loader:
                data = data[0]

                # split input into set of time windows without overlapping
                # windows: shape=(T // window_size, window_size, input_dim)
                windows = np.split(data[:, :, :self.window_size * (data.shape[-1] // self.window_size)],
                                   (data.shape[-1] // self.window_size), -1)
                windows = np.concatenate(windows, 0)
                windows = torch.from_numpy(windows).transpose(1, 2).to(self.device)

                # out: shape=(1, T, repr_dim)
                out = self.model(False, windows)
                out = out.reshape(-1, out.shape[-1]).unsqueeze(0)
                output.append(out.cpu())

            output = torch.cat(output, dim=0)

        # 각 관측치의 전체 데이터를 time window로 분할하여 모델링하므로 time window마다 representation vector 도출
        # 이때 STOC는 time window의 모든 시점에 대하여 representation vector가 도출되므로 최종적으로 (# observations, window_size * (T // window_size), repr_dim) 차원의 representation 도출
        # 시점을 기준으로 1D max pooling을 적용하여 최종 representation 도출
        output = F.max_pool1d(
            output.transpose(1, 2),
            kernel_size=output.size(1),
        ).transpose(1, 2)
        output = output.squeeze(1)

        output_df = pd.DataFrame(output.numpy())
        output_df.columns = [f'V{i+1}' for i in range(output.shape[-1])]
        return output_df
