import copy
import pandas as pd
import torch.nn.functional as F

import sys
sys.path.append('./models/rae_mepc')

from model import *

import warnings
warnings.filterwarnings("ignore")


class Trainer_RAE_MEPC:
    def __init__(self, args):
        """
        Initialize Trainer_RAE_MEPC class and prepare model and optimizer for training

        :param args: configuration
        :type args: dictionary
        """

        self.args = args
        self.device = args['device']

        # set decoding mask
        self.decoding_mask()

        # build model and optimizer
        self.model = RAE_MEPC(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=1e-6)

    def fit(self, train_loader, valid_loader):
        """
        Train the RAE-MEPC model

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: trained RAE-MEPC model
        :rtype: model
        """

        # start training
        best_model = copy.deepcopy(self.model)
        best_valid_loss = 1000000000
        
        for epoch in range(self.args['num_epoch']):
            # training
            train_loss = self.model_train(train_loader)
            print(f"Epoch #{epoch + 1}: train loss={train_loss}")
            
            # validate
            valid_loss = self.model_evaluate(valid_loader)
            print(f"Epoch #{epoch + 1}: validation loss={valid_loss}\n")

            # update best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)
        return best_model
            
    def model_train(self, train_loader):
        """
        Train the RAE-MEPC model for one epoch

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :return: training loss
        :rtype: float
        """

        total_loss = []
        self.model.train()

        for batch_idx, (data, pred_data) in enumerate(train_loader):
            data = data.transpose(1, 2).to(self.device)
            pred_data = pred_data.transpose(1, 2).to(self.device)

            # get loss
            loss, _, _, _ = self.model(data, pred_data)
            total_loss.append(loss.item())

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def model_evaluate(self, valid_loader):
        """
        Evaluate the RAE-MEPC model

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: validation loss
        :rtype: float
        """

        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, pred_data) in enumerate(valid_loader):
                data = data.transpose(1, 2).to(self.device)
                pred_data = pred_data.transpose(1, 2).to(self.device)

                # get loss
                loss, _, _, _ = self.model(data, pred_data, mode='test')
                total_loss.append(loss.item())

        total_loss = torch.tensor(total_loss).mean()
        return total_loss

    def encode(self, test_loader):
        """
        Encode raw data to representation using trained RAE-MEPC model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: representation vectors
        :rtype: dataFrame
        """

        window_size = self.args['window_length']

        self.model.eval()
        with torch.no_grad():
            output = []
            for data in test_loader:
                data = data[0]

                # split input into set of time windows without overlapping
                # windows: shape=(T // window_size, window_size, input_dim)
                windows = np.split(data[:, :, :window_size * (data.shape[-1] // window_size)],
                                   (data.shape[-1] // window_size), -1)
                windows = np.concatenate(windows, 0)
                windows = torch.from_numpy(windows).transpose(1, 2).to(self.device)

                # out: shape=(1, T // window_size, repr_dim)
                out = self.model.get_enc_outputs(windows)
                out = out[0][0].unsqueeze(0)
                output.append(out.cpu())

            output = torch.cat(output, dim=0)

        # 각 관측치의 전체 데이터를 time window로 분할하여 모델링하므로 time window마다 representation vector가 도출되므로 time window를 기준으로 1D max pooling을 적용하여 최종 representation 도출
        output = F.max_pool1d(
            output.transpose(1, 2),
            kernel_size=output.size(1),
        ).transpose(1, 2)
        output = output.squeeze(1)

        output = pd.DataFrame(output.numpy())
        output.columns = [f'V{i + 1}' for i in range(output.shape[-1])]
        return output
    
    def decoding_mask(self):
        """
        Make mask for multi-resolution decoding
        """

        self.args['all_decode_masks'] = []
        for i in range(self.args['dec_nlayers']):
            mask_i = np.random.randint(1, high=4, size=self.args['window_length'], dtype='l')
            temp_mask = []
            for j in mask_i:
                temp_mask.append([1, 0])
            self.args['all_decode_masks'].append(temp_mask)

        ratios = [1 / (self.args['tau'] ** (self.args['dec_nlayers'] - i - 1)) for i in range(self.args['dec_nlayers'])]
        self.args['dec_Ls'] = np.random.randint(1, high=10 + 1, size=self.args['dec_nlayers'], dtype='l')

        self.args['dec_lengths'] = []
        for i in range(self.args['dec_nlayers']):
            self.args['dec_lengths'].append(int(self.args['window_length'] * ratios[i]))
