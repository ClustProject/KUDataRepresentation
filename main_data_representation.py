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

        self.model_name = config['model']
        self.parameter = config['parameter']

        self.model_config = self.get_model_config(self.parameter)
        self.train_loader, self.valid_loader, self.test_loader, self.inference_train_loader = self.get_loaders(train_data, test_data)

    def build_model(self):
        """
        Train model and return best model

        :return: best trained model
        :rtype: model
        """
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

        :return: best trained model
        :rtype: model
        """

        print("Start training model\n")
        best_model = model.fit(self.train_loader, self.valid_loader)
        return best_model

    def save_model(self, best_model, best_model_path):
        torch.save(best_model.state_dict(), best_model_path)

    def encode_data(self, model, best_model_path):
        """
        Encode representations from trained model

        :param best_model: best trained model
        :type best_model: model

        :return: representation vector
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
        # set configuration for training model
        model_config = copy.deepcopy(config)
        if self.model_name == 'ts2vec':
            replaced_key_dict = {
                'input_dim': 'input_dims',
                'repr_dim': 'output_dims',
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

        for config_key in replaced_key_dict:
            model_config_key = replaced_key_dict[config_key]
            model_config[model_config_key] = model_config.pop(config_key)
        return model_config

    def get_loaders(self, x_train, x_test):
        batch_size = self.parameter['batch_size']

        # train data를 8:2의 비율로 train/validation set으로 분할
        inference_x_train = copy.deepcopy(x_train)
        x_train, x_valid = train_test_split(x_train, test_size=0.2, random_state=42)

        datasets = []
        for dataset in [x_train, x_valid, x_test, inference_x_train]:
            # 전체 시간 길이 설정
            T = dataset.shape[-1]

            # TS2Vec & TS-TCC train/validation/test 데이터셋 생성: shape = (batch_size, input_dims, time_steps)
            if self.model_name in ['ts2vec', 'ts_tcc']:
                # 각 관측치의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(dataset)))

            # RAE-MEPC 모델을 위한 train/validation/test 데이터셋 생성: shape = (batch_size, input_dims, window_size)
            elif self.model_name == 'rae_mepc':
                window_size = self.parameter['window_size']
                pred_size = window_size // 2

                # 전체 데이터를 window_size 크기의 time window로 분할하여 input을 생성함
                windows = np.split(dataset[:, :, :-1 * pred_size][:, :, :window_size * ((T - pred_size) // window_size)],
                                   ((T - pred_size) // window_size), -1)
                windows = np.concatenate(windows, 0)

                targets = np.roll(dataset, -1 * pred_size, axis=2)
                targets = np.split(targets[:, :, :-1 * pred_size][:, :, :window_size * ((T - pred_size) // window_size)],
                                   ((T - pred_size) // window_size), -1)
                targets = np.concatenate(targets, 0)

                # 분할된 time window 단위의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets)))

            # STOC 모델을 위한 train/validation/test 데이터셋 생성: shape = (batch_size, input_dims, window_size)
            elif self.model_name == 'stoc':
                window_size = self.parameter['window_size']
                forecast_step = self.parameter['forecast_step']

                # 전체 데이터를 window_size 크기의 time window로 분할하여 input을 생성함
                windows = np.split(dataset[:, :, :-1 * forecast_step][:, :, :window_size * ((T - forecast_step) // window_size)],
                                   ((T - forecast_step) // window_size), -1)
                windows = np.concatenate(windows, 0)

                # input에 대하여 forecast_step 시점만큼의 미래 데이터를 target으로 사용함
                targets = np.roll(dataset, -1 * forecast_step, axis=2)
                targets = np.split(targets[:, :, :-1 * forecast_step][:, :, :window_size * ((T - forecast_step) // window_size)],
                                   ((T - forecast_step) // window_size), -1)
                targets = np.concatenate(targets, 0)
                targets = targets[:, :, :forecast_step]

                # 분할된 time window 단위의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets)))

        # train/validation/test DataLoader 구축
        train_set, valid_set, test_set, inference_train_set = datasets[0], datasets[1], datasets[2], datasets[3]
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        inference_train_loader = torch.utils.data.DataLoader(inference_train_set, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader, inference_train_loader