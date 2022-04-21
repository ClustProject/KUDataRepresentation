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
    "model": 'ts2vec',
    "training": False,
    "best_model_path": './ckpt/ts2vec.pt',
    "parameter": {
        "input_dim": 9, # 데이터의 변수 개수, int
        "repr_dim": 128, # data representation 차원, int(default: 320, 범위: 1 이상, 2의 지수로 설정 권장)
        "num_epochs": 20, # 학습 epoch 횟수, int(default: 20, 범위: 1 이상)
        "batch_size": 1024, # batch 크기, int(default: 8, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
        "lr": 0.001, # learning rate, float(default: 0.0001, 범위: 0.1 이하)
        "device": "cuda", # 학습 환경, ["cuda", "cpu"] 중 선택
    }
}

config2 = {
    "model": 'ts_tcc',
    "parameter": {
        "window_size": 64,  # 모델의 input sequence 길이, int(default: 64, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
        "input_dim": 9,  # 데이터의 변수 개수, int
        "repr_dim": 128,  # data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
        "hidden_dim": 100,
        "num_epochs": 100,  # 학습 epoch 횟수, int(default: 100, 범위: 1 이상)
        "batch_size": 64,  # batch 크기, int(default: 64, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
        "lr": 0.0001,  # learning rate, float(default: 0.0001, 범위: 0.1 이하)
        "device": "cuda",  # 학습 환경, ["cuda", "cpu"] 중 선택
        "verbose": True,
        "jitter_scale_ratio": 1.1,
        "jitter_ratio": 0.8,
        "max_seg": 8
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
data_repr = mdr.Encode(config, train_data, test_data)
model = data_repr.build_model()
if config["training"]:
    best_model = data_repr.train_model(model)
    data_repr.save_model(best_model, best_model_path=config["best_model_path"])
train_repr, test_repr = data_repr.encode_data(model, best_model_path=config["best_model_path"])
# train_repr.to_csv('../ts2vec_repr_train.csv', index=False)
# test_repr.to_csv('../ts2vec_repr_test.csv', index=False)