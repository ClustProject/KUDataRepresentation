import sys
sys.path.append('./models/ts_tcc')

import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.TC import TC
from modules.model import base_Model
from modules.loss import NTXentLoss
from augmentations import DataTransform

import warnings
warnings.filterwarnings(action='ignore')


class Trainer_TS_TCC:
    def __init__(self, config):
        """
        Initialize Trainer_TS_TCC class and prepare models and optimizers for training.

        :param config: configuration
        :type config: dictionary
        """

        self.config = config
        self.device = config['device']

        # build model
        self.model = base_Model(self.config).to(self.device)
        self.temporal_contr_model = TC(config, self.device).to(self.device)

        # build optimizer
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], betas=(0.9, 0.99), weight_decay=3e-4)
        self.temp_cont_optimizer = torch.optim.Adam(self.temporal_contr_model.parameters(), lr=config['lr'], betas=(0.9, 0.99), weight_decay=3e-4)

    def fit(self, train_loader, valid_loader):
        """
        Train the TS-TCC model

        :param train_loader: train dataloader.
        :type train_loader: DataLoader

        :param valid_loader: validation dataloader.
        :type valid_loader: DataLoader

        :return: trained TS-TCC model
        :rtype: model
        """

        best_model = copy.deepcopy(self.model)
        best_valid_loss = 1000000000

        for epoch in range(1, self.config['num_epoch'] + 1):
            # training
            train_loss = self.model_train(train_loader)
            print(f"Epoch #{epoch}: train loss={train_loss}")
            
            # validation
            valid_loss = self.model_evaluate(valid_loader)
            print(f"Epoch #{epoch + 1}: validation loss={valid_loss}\n")

            # update best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)
        return best_model
            
    def model_train(self, train_loader):
        """
        Train the TS-TCC model for one epoch

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :return: training loss
        :rtype: float
        """

        total_loss = []
        self.model.train()
        self.temporal_contr_model.train()

        for batch_idx, data in enumerate(train_loader):
            self.model_optimizer.zero_grad()
            self.temp_cont_optimizer.zero_grad()

            data = data[0]

            # get loss
            loss = self.get_loss(self.model, self.temporal_contr_model, data)
            total_loss.append(loss.item())

            # backward propagation
            loss.backward()
            self.model_optimizer.step()
            self.temp_cont_optimizer.step()

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def model_evaluate(self, valid_loader):
        """
        Evaluate the TS-TCC model

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: validation loss
        :rtype: float
        """

        total_loss = []
        self.model.eval()
        self.temporal_contr_model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                data = data[0]

                # get loss
                loss = self.get_loss(self.model, self.temporal_contr_model, data)
                total_loss.append(loss.item())

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def get_loss(self, model, temporal_contr_model, data):
        """
        Compute the loss for temporal and contextual contrasting

        :param model: encoder in TS-TCC
        :type model: model

        :param temporal_contr_model: temporal contrasting model in TS-TCC
        :type temporal_contr_model: model

        :param data: input data
        :type data: Tensor

        :return: loss
        :rtype: float
        """

        # get augmented views by weak and strong augmentation
        aug1, aug2 = DataTransform(data, self.config)

        # send to device
        aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)

        # get features from encoder
        features1 = model(aug1)
        features2 = model(aug2)

        # normalize projection feature vectors
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # get loss and feature from temporal contrasting model
        temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
        temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

        # normalize projection feature vectors
        zis = temp_cont_lstm_feat1 
        zjs = temp_cont_lstm_feat2

        # compute loss
        lambda1 = 1
        lambda2 = 0.7
        nt_xent_criterion = NTXentLoss(self.device, self.config['batch_size'], 0.2, True)
        loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2
        return loss

    def encode(self, test_loader):
        """
        Encode raw data to representation using trained TS-TCC model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: representation vectors
        :rtype: dataFrame
        """

        self.model.eval()
        with torch.no_grad():
            output = []
            for data in test_loader:
                # send to device
                data = data[0].float().to(self.device)

                # out: shape=(batch_size, repr_dim, output dim from Conv1D)
                out = self.model(data)
                output.append(out.cpu())

            output = torch.cat(output, dim=0)

        # Conv1D 사용으로 인해 각 관측치에서 encoder의 output dim에 대하여 representation vector가 도출되므로 output dim을 기준으로 1D max pooling을 적용하여 최종 representation 도출
        output = F.max_pool1d(
            output,
            kernel_size=output.size(2),
        )
        output = output.squeeze(2)

        output = pd.DataFrame(output.numpy())
        output.columns = [f'V{i + 1}' for i in range(output.shape[-1])]
        return output
