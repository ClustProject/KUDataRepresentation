import torch
import numpy as np
from sklearn.model_selection import train_test_split

from models.stoc.modules.dataset import BuildDataset
from models.rae_mepc.main import train_RAE_MEPC, encode_RAE_MEPC
from models.stoc.main import train_STOC, encode_STOC
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

        self.train_loader, self.valid_loader, self.test_loader = self.get_loaders(train_data, test_data)

    def getResult(self) :
        """
        getResult by model and its parameter
        :return: data representation
        :rtype: dataFrame
        """
        if self.model == 'rae_mepc':
            trained_model = train_RAE_MEPC(self.parameter, self.train_loader, self.valid_loader)
            result = encode_RAE_MEPC(self.parameter, self.test_loader, trained_model)
        elif self.model == 'stoc':
            trained_model = train_STOC(self.parameter, self.train_loader, self.valid_loader)
            result = encode_STOC(self.parameter, self.test_loader, trained_model)
        elif self.model == 'ts_tcc':
            trained_model = train_TS_TCC(self.parameter, self.train_loader, self.valid_loader)
            result = encode_TS_TCC(self.parameter, self.test_loader, trained_model)
        elif self.model == 'ts2vec':
            trained_model = train_TS2Vec(self.parameter, self.train_loader, self.valid_loader)
            result = encode_TS2Vec(self.parameter, self.test_loader, trained_model)
        return result
            
    def get_loaders(self, x_train, x_test):
        window_size = self.parameter['window_size']
        batch_size = self.parameter['batch_size']
        if self.model == 'stoc':
            forecast_step = self.parameter['forecast_step']

        # train data를 8:2의 비율로 train/validation set으로 분할
        x_train, x_valid = train_test_split(x_train, test_size=0.2, random_state=42)

        datasets = []
        for dataset in [x_train, x_valid, x_test]:
            # 전체 시간 길이 설정
            T = dataset.shape[-1]

            # 예측 모델을 위한 train/validation/test 데이터셋 생성: shape=(batch_size, input_dims, window_size)
            if self.model == 'stoc':
                # 전체 데이터를 window_size 크기의 time window로 분할하여 input을 생성함
                windows = np.split(dataset[:, :, -1 * forecast_step][:, :, :window_size * (T // window_size)], (T // window_size), -1)
                windows = np.concatenate(windows, 0)

                # input에 대하여 forecast_step 시점만큼의 미래 데이터를 target으로 사용함
                targets = np.roll(dataset, -1 * forecast_step)
                targets = np.split(targets[:, :, -1 * forecast_step][:, :, :window_size * (T // window_size)], (T // window_size), -1)
                
                # 분할된 time window 단위의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(windows), torch.FloatTensor(targets)))
            
            # 복원 모델을 위한 train/validation/test 데이터셋 생성: shape=(batch_size, input_dims, window_size)
            else:
                # 전체 데이터를 window_size 크기의 time window로 분할하여 input을 생성함
                windows = np.split(dataset[:, :, :window_size * (T // window_size)], (T // window_size), -1)
                windows = np.concatenate(windows, 0)

                # 분할된 time window 단위의 데이터를 tensor 형태로 축적
                datasets.append(torch.utils.data.TensorDataset(torch.FloatTensor(windows)))

        # train/validation/test DataLoader 구축
        trainset, validset, testset = datasets[0], datasets[1], datasets[2]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader
