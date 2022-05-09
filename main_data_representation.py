import torch
import copy
import numpy as np
from sklearn.model_selection import train_test_split

from models.ts2vec.trainer import Trainer_TS2Vec
from models.ts_tcc.trainer import Trainer_TS_TCC
from models.rae_mepc.trainer import Trainer_RAE_MEPC
from models.stoc.trainer import Trainer_STOC


class Encode():
    def __init__(self, config, train_data, test_data):
        """
        Initialize Encode class and prepare dataloaders for training and testing.

        :param config: config
        :type config: dictionary

        :param train_data: train data whose shape is (# observations, # features, # time steps)
        :type train_data: numpy array

        :param test_data: test data whose shape is (# observations, # features, # time steps)
        :type test_data: numpy array

        example
            >>> config = {
                    "model": 'ts2vec',
                    "training": True,  # 학습 여부, 저장된 학습 완료 모델 존재시 False로 설정
                    "best_model_path": './ckpt/ts2vec.pt',  # 학습 완료 모델 저장 경로
                    "parameter": {
                        "input_dim": 9,  # 데이터의 변수 개수, int
                        "repr_dim": 64,  # data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
                        "num_epochs": 30,  # 학습 epoch 횟수, int(default: 30, 범위: 1 이상)
                        "batch_size": 512,  # batch 크기, int(default: 512, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
                        "lr": 0.001,  # learning rate, float(default: 0.001, 범위: 0.1 이하)
                        "device": "cuda",  # 학습 환경, ["cuda", "cpu"] 중 선택
                    }
                }
            >>> data_repr = mdr.Encode(config, train_data, test_data)
            >>> model = data_repr.build_model()  # 모델 구축
            >>> if config["training"]:
            >>>     best_model = data_repr.train_model(model)  # 모델 학습
            >>>     data_repr.save_model(best_model, best_model_path=config["best_model_path"])  # 모델 저장
            >>> train_repr, test_repr = data_repr.encode_data(model, best_model_path=config["best_model_path"])  # representation 도출
        """

        self.model_name = config['model']
        self.parameter = config['parameter']

        self.model_config = self.get_model_config(self.parameter)
        self.train_loader, self.valid_loader = self.get_train_loaders(train_data)
        self.inference_train_loader, self.test_loader = self.get_test_loaders(train_data, test_data)

    def build_model(self):
        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model_name == 'ts2vec':
            model = Trainer_TS2Vec(**self.model_config)
        elif self.model_name == 'ts_tcc':
            model = Trainer_TS_TCC(self.model_config)
        elif self.model_name == 'rae_mepc':
            model = Trainer_RAE_MEPC(self.model_config)
        elif self.model_name == 'stoc':
            model = Trainer_STOC(self.model_config)
        return model

    def train_model(self, model):
        """
        Train model and return best model

        :param model: initialized model
        :type model: model

        :return: best trained model
        :rtype: model
        """

        print("Start training model\n")

        # train model
        best_model = model.fit(self.train_loader, self.valid_loader)
        return best_model

    def save_model(self, best_model, best_model_path):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        :param best_model_path: path for saving model
        :type best_model_path: str
        """

        # save model
        torch.save(best_model.state_dict(), best_model_path)

    def encode_data(self, model, best_model_path):
        """
        Encode raw data to representations based on the best trained model

        :param model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: representation vectors for train and test dataset
        :rtype: numpy array
        """

        print("Start encoding data\n")

        # load best model
        if self.model_name == 'ts2vec':
            model.net.load_state_dict(torch.load(best_model_path))
        else:
            model.model.load_state_dict(torch.load(best_model_path))

        # get representation
        train_repr = model.encode(self.inference_train_loader)
        test_repr = model.encode(self.test_loader)
        return train_repr, test_repr

    def get_model_config(self, config):
        """
        Get model configuration for selected model by replacing input configuration with one used in selected model

        :param config: input config
        :type config: config

        :return: configuration whose keys are suitable for selected model
        :rtype: dictionary
        """

        # copy input configuration
        model_config = copy.deepcopy(config)

        # set key to be replaced
        if self.model_name == 'ts2vec':
            replaced_key_dict = {
                'input_dim': 'input_dims',
                'repr_dim': 'output_dims',
                'hidden_dim': 'hidden_dims',
                'num_epochs': 'n_epochs'
            }
        elif self.model_name == 'ts_tcc':
            replaced_key_dict = {
                'input_dim': 'input_channels',
                'repr_dim': 'final_out_channels',
                'num_epochs': 'num_epoch',
            }
        elif self.model_name == 'rae_mepc':
            replaced_key_dict = {
                'input_dim': 'ninp',
                'repr_dim': 'hidden_size',
                'window_size': 'window_length',
                'num_epochs': 'num_epoch'
            }
        elif self.model_name == 'stoc':
            replaced_key_dict = {
                'repr_dim': 'output_dim',
                'hidden_dim': 'feature_size'
            }

        # replace input configuration with one used in selected model
        for config_key in replaced_key_dict:
            model_config_key = replaced_key_dict[config_key]
            model_config[model_config_key] = model_config.pop(config_key)
        return model_config

    def get_train_loaders(self, x_train):
        """
        train dataset을 기반으로 모델 학습을 위한 train 및 validation loader를 생성하는 함수
        TS2Vec & TS-TCC는 한 관측치에 대한 전체 시점의 데이터를 input으로 사용
        RAE-MEPC & STOC는 한 관측치에 대한 전체 시점의 데이터를 기반으로 생성한 window_size 크기의 time window와 예측을 위한 time window를 input으로 사용

        :param x_train: train data whose shape is (# observations, # features, # time steps)
        :type x_train: numpy array

        :return: dataloaders for training and validation
        :rtype: DataLoader
        """

        batch_size = self.parameter['batch_size']

        # train data를 8:2의 비율로 train/validation set으로 분할
        x_train, x_valid = train_test_split(x_train, test_size=0.2, random_state=42)

        datasets = []
        for dataset in [x_train, x_valid]:
            # 전체 시간 길이 설정
            T = dataset.shape[-1]

            # TS2Vec & TS-TCC train/validation 데이터셋 생성: shape = (batch_size, input_dims, T)
            if self.model_name in ['ts2vec', 'ts_tcc']:
                # 각 관측치의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(dataset)))

            # RAE-MEPC & STOC 모델을 위한 train/validation 데이터셋 생성: input time window와 예측 time window로 구성
            # input time window: shape = (batch_size, input_dims, window_size)
            # 예측 time window는 input time window의 각 시점에서 forecast_step만큼 이동한 미래 데이터: shape = (batch_size, input_dims, window_size)
            else:
                window_size = self.parameter['window_size']
                if self.model_name == 'stoc':
                    forecast_step = self.parameter['forecast_step']
                else:
                    forecast_step = window_size // 2

                # 전체 데이터를 겹치는 데이터 없이 window_size 크기의 time window로 분할하여 input 생성
                windows = np.split(dataset[:, :, :-1 * forecast_step][:, :, :window_size * ((T - forecast_step) // window_size)],
                                   ((T - forecast_step) // window_size), -1)
                windows = np.concatenate(windows, 0)

                # input time window에 대하여 forecast_step 시점 만큼 이동한 후 이를 window_size 크기로 분할하여 예측 time window 생성
                targets = np.roll(dataset, -1 * forecast_step, axis=2)
                targets = np.split(targets[:, :, :-1 * forecast_step][:, :, :window_size * ((T - forecast_step) // window_size)],
                                   ((T - forecast_step) // window_size), -1)
                targets = np.concatenate(targets, 0)

                # 분할된 time window 단위의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets)))

        # train/validation DataLoader 구축
        train_set, valid_set = datasets[0], datasets[1]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader, valid_loader

    def get_test_loaders(self, x_train, x_test):
        """
        train 및 test dataset을 기반으로 representation vector 도출을 위한 train 및 test loader를 생성하는 함수
        모든 모델이 한 관측치에 대한 전체 시점의 데이터를 input으로 사용
        단, RAE-MEPC & STOC는 time window 단위로 모델링하므로 batch_size를 1로 설정한 후, 모델의 encoding 단계에서 input을 window로 분할하여 사용

        :param x_train: train data whose shape is (# observations, # features, # time steps)
        :type x_train: numpy array

        :param x_test: test data whose shape is (# observations, # features, # time steps)
        :type x_test: numpy array

        :return: dataloaders for training and testing
        :rtype: DataLoader
        """

        # set batch size
        if self.model_name in ['ts2vec', 'ts_tcc']:
            batch_size = self.parameter['batch_size']
        else:  # RAE-MEPC & STOC는 batch_size를 1로 설정
            batch_size = 1

        # train/test 데이터셋 생성: shape = (batch_size, input_dims, T)
        datasets = []
        for dataset in [x_train, x_test]:
            # 각 관측치의 데이터를 tensor 형태로 축적
            datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(dataset)))
        
        # train/test DataLoader 구축 (encoding 단계이므로 shuffle=False로 설정)
        inference_train_set, test_set = datasets[0], datasets[1]
        inference_train_loader = torch.utils.data.DataLoader(inference_train_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return inference_train_loader, test_loader
        
