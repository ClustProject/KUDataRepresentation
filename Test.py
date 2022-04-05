import torch
import pickle
import random
import numpy as np
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
        "window_size": 64,  # 모델의 input sequence 길이, int(default: 64, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
        "input_dim": 561,  # 데이터의 변수 개수, int
        "repr_dim": 128,  # data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        "num_epochs": 100,  # 학습 epoch 횟수, int(default: 100, 범위: 1 이상)
        "batch_size": 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
        "lr": 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
        "device": "cuda"  # 학습 환경, ["cuda", "cpu"] 중 선택
    }
}

# Dataset
dataset_dir = {
    "train": './data/X_train.pkl',
    "test": './data/X_test.pkl'
}

# train/test 데이터 불러오기 (pickle 형태)
# shape=(# observations, # features, # time steps)
train_data = pickle.load(open(dataset_dir["train"], 'rb'))  # shape=(7352, 9, 128)
test_data = pickle.load(open(dataset_dir["test"], 'rb'))  # shape=(2947, 9, 128)

# Case 1. model = rae_mepc
config = config1
data_representation = mdr.Encode(config, train_data, test_data)
output = data_representation.getResult()
