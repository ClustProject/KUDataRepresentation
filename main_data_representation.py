import os 
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from models.rae_mepc.main import train_RAE_MEPC, encode_RAE_MEPC

from models.stoc.main import train_STOC, encode_STOC
from models.stoc.modules.dataset import BuildDataset

from models.ts_tcc.main import train_TS_TCC, encode_TS_TCC
from models.ts2vec.main import train_TS2Vec, encode_TS2Vec


class Encode():
    def __init__(self, config, train_data, test_data):
        """
        Initialize Alignment class and prepare OverlapData based on min-max index.

        :param config: config 
        :type config: dictionary

        :param train_data: train data whose shape is (batch_size x input_size x seq_len)
        :type train_data: numpy array
        
        :param test_data: test data whose shape is (batch_size x input_size x seq_len)
        :type test_data: numpy array

        example
            >>> config = { 
                    "window_size": 64, # 모델의 input sequence 길이, int(default: 64, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
                    "input_dim": 561, # 데이터의 변수 개수, int
                    "repr_dim": 128, # data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
                    "num_epochs": 100, # 학습 epoch 횟수, int(default: 100, 범위: 1 이상)
                    "batch_size": 64, # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
                    "learning_rate": 0.0001, # learning rate, float(default: 0.0001, 범위: 0.1 이하)
                    "device": "cuda" # 학습 환경, ["cuda", "cpu"] 중 선택
                }
            >>> data_representation = mdr.Encode(config, train_data, test_data)
            >>> output = data_representation.getResult()
        """

        self.model = config['model']
        self.parameter = config['parameter']

        self.train_loader, self.valid_loader, self.test_loader = self.get_loaders(train_data,
                                                                                test_data,
                                                                                self.parameter["window_size"],
                                                                                self.parameter["batch_size"], 
                                                                                self.model)

    def getResult(self) :
        """
        getResult by model and its parameter
        :return: data representation
        :rtype: dataFrame
        """
        
        if self.model == 'rae_mepc':
            result = self.RAE_MEPC()
        elif self.model == 'stoc':
            result = self.STOC()
        elif self.model == 'ts_tcc':
            result = self.TS_TCC()
        elif self.model == 'ts2vec':
            result = self.TS2Vec()
        return result
        
    def RAE_MEPC(self):
        model = train_RAE_MEPC(self.parameter, self.train_loader, self.valid_loader)
        result_repr = encode_RAE_MEPC(self.parameter, self.test_loader, model)
        return result_repr
    
    def STOC(self):
        model = train_STOC(self.parameter, self.train_loader, self.valid_loader)
        result_repr = encode_STOC(self.parameter, self.test_loader, model)
        return result_repr
    
    def TS_TCC(self):
        model = train_TS_TCC(self.parameter, self.train_loader, self.valid_loader)
        result_repr = encode_TS_TCC(self.parameter, self.test_loader, model)
        return result_repr
    
    def TS2Vec(self):
        model = train_TS2Vec(self.parameter, self.train_loader, self.valid_loader)
        result_repr = encode_TS2Vec(self.parameter, self.test_loader, model)
        return result_repr
            
    def get_loaders(self, train_data, test_data, window_size, batch_size, model):
        # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
        n_train = int(0.8 * len(train_data))
        n_valid = len(train_data) - n_train
        n_test = len(test_data)

        # train/validation set의 개수에 맞게 데이터 분할
        x_train = train_data[:n_train]
        x_valid = train_data[n_train:]
        x_test = test_data

        if model == 'stoc':
            # 예측 모델 학습을 위한 dataset 생성
            # shape=(batch_size x window_size x input_dims)
            trainset = BuildDataset(self.parameter, x_train, overlap=True)
            validset = BuildDataset(self.parameter, x_valid, overlap=True)
            testset = BuildDataset(self.parameter, x_test, overlap=False)

        else:
            # train/validation/test 데이터를 window_size 시점 길이로 분할
            datasets = []
            for set in [(x_train, n_train), (x_valid, n_valid), (x_test, n_test)]:
                # 전체 시간 길이 설정
                T = set[0].shape[-1]
                # 전체 X 데이터를 window_size 크기의 time window로 분할
                windows = np.split(set[0][:, :, :window_size * (T // window_size)], (T // window_size), -1)
                windows = np.concatenate(windows, 0)
                # 분할된 time window 단위의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.Tensor(windows)))

            # train/validation/test DataLoader 구축: shape=(batch_size x input_dims x window_size)
            trainset, validset, testset = datasets[0], datasets[1], datasets[2]

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader
        
