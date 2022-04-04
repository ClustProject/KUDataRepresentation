import torch
import pickle
import random
import numpy as np
import pandas as pd
import main_data_representation as mdr


# seed 고정
random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

config1 = {
    "model": 'rae_mepc',
    "parameter": {
        "window_size": 64, # 모델의 input sequence 길이, int(default: 64, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
        "input_dim": 561, # 데이터의 변수 개수, int
        "repr_dim": 128, # data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        "num_epochs": 100, # 학습 epoch 횟수, int(default: 100, 범위: 1 이상)
        "batch_size": 64, # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
        "lr": 0.0001, # learning rate, float(default: 0.0001, 범위: 0.1 이하)
        "device": "cuda" # 학습 환경, ["cuda", "cpu"] 중 선택
    }
}

# Dataset
dataset_dir = {
    "train": './data/x_train.pkl',
    "test": './data/x_test.pkl'
}

# train/test 데이터 불러오기 (pickle 형태)
train_data = pd.read_table('./data/X_train.txt', sep='\t')  # shape: (9, 561, 288)
test_data = pd.read_table('./data/X_test.txt', sep='\t') # shape: (21, 561, 281)

# Case 1. model = rae_mepc
config = config1
data_representation = mdr.Encode(config, train_data, test_data)
output = data_representation.getResult()

trainX = pd.read_csv('./data/X_train.txt', delim_whitespace=True,header=None)
trainy = pd.read_csv('./data/y_train.txt',delim_whitespace=True,header=None)
testX = pd.read_csv('./data/X_test.txt',delim_whitespace=True,header=None)
trainy.shape