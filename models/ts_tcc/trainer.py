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
        self.config = config
        self.device = config['device']

        # Load Model
        self.model = base_Model(self.config).to(self.device)
        self.temporal_contr_model = TC(config, self.device).to(self.device)

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], betas=(0.9, 0.99), weight_decay=3e-4)
        self.temp_cont_optimizer = torch.optim.Adam(self.temporal_contr_model.parameters(), lr=config['lr'], betas=(0.9, 0.99), weight_decay=3e-4)

    def fit(self, train_loader, valid_loader):
        # Start training
        best_model = copy.deepcopy(self.model)
        best_valid_loss = 1000000000

        for epoch in range(1, self.config['num_epoch'] + 1):
            # Train
            train_loss = self.model_train(train_loader)
            print(f"Epoch #{epoch}: train loss={train_loss}")
            
            # Validate
            valid_loss = self.model_evaluate(valid_loader)
            print(f"Epoch #{epoch + 1}: validation loss={valid_loss}\n")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)
        return best_model
            
    def model_train(self, train_loader):
        total_loss = []
        self.model.train()
        self.temporal_contr_model.train()

        for batch_idx, data in enumerate(train_loader):
            # optimizer
            self.model_optimizer.zero_grad()
            self.temp_cont_optimizer.zero_grad()

            data = data[0]
            loss = self.get_loss(self.model, self.temporal_contr_model, data)
            total_loss.append(loss.item())

            loss.backward()
            self.model_optimizer.step()
            self.temp_cont_optimizer.step()

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def model_evaluate(self, valid_loader):
        total_loss = []
        self.model.eval()
        self.temporal_contr_model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                data = data[0]
                loss = self.get_loss(self.model, self.temporal_contr_model, data)
                total_loss.append(loss.item())

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def get_loss(self, model, temporal_contr_model, data):
        aug1, aug2 = DataTransform(data, self.config)

        # send to device
        aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)

        features1 = model(aug1)
        features2 = model(aug2)

        # normalize projection feature vectors
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

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
        self.model.eval()

        with torch.no_grad():
            output = []
            for data in test_loader:
                # send to device
                data = data[0].float().to(self.device)

                out = self.model(data)
                out = out.reshape(out.shape[0], -1).cpu()
                output.append(out)

            output = torch.cat(output, dim=0)

        output = pd.DataFrame(output.numpy())
        output.columns = [f'V{i + 1}' for i in range(output.shape[-1])]
        return output
