import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .modules import TSEncoder
from .modules.losses import hierarchical_contrastive_loss
from .utils import take_per_row, torch_pad_nan
from einops import rearrange
import copy


class Trainer_TS2Vec:
    def __init__(self, input_dims, output_dims, hidden_dims, device, lr, batch_size, n_epochs, depth=10, temporal_unit=0):
        """
        Initialize a TS2Vec model

        :param input_dims: The input dimension. For a univariate time series, this should be set to 1.
        :type input_dims: int

        :param output_dims: The representation dimension.
        :type output_dims: int

        :param hidden_dims: The hidden dimension of the encoder.
        :type hidden_dims: int

        :param depth: The number of hidden residual blocks in the encoder.
        :type depth: int

        :param device: The device used for training and inference(cuda or cpu).
        :type device: str

        :param lr: The learning rate.
        :type lr: float

        :param batch_size: The batch size.
        :type batch_size: int

        :param n_epochs: The number of epochs.
        :type n_epochs: int

        :param temporal_unit: The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
        :type temporal_unit: int
        """

        super().__init__()

        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.best_val_loss = 1e+08

    def fit(self, train_loader, valid_loader):
        """
        Train the TS2Vec model

        :param train_loader: train dataloader
        :type train_loader: DataLoader

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: trained TS2Vec Encoder
        :rtype: model
        """

        # build optimizer
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            cum_loss = 0
            n_epoch_iters = 0

            for batch in train_loader:
                x = batch[0].permute(0, 2, 1)
                x = x.to(self.device)

                # make input
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                optimizer.zero_grad()

                # get output
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]

                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1

            cum_loss /= n_epoch_iters
            print(f"Epoch #{epoch + 1}: train loss={cum_loss}")

            val_loss = self.save_best_model(valid_loader)
            print(f"Epoch #{epoch + 1}: validation loss={val_loss}\n")
        return self.best_model

    def save_best_model(self, valid_loader):
        """
        Evaluate the TS2Vec model

        :param valid_loader: validation dataloader
        :type valid_loader: DataLoader

        :return: validation loss
        :rtype: float
        """

        val_loss = 0
        n_iters = 0
        org_training = self.net.training

        self.net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                x = batch[0].permute(0, 2, 1)
                x = x.to(self.device)

                # make input
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                # get output
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]

                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )

                val_loss += loss.item()
                n_iters += 1

        val_loss /= n_iters

        # update best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = copy.deepcopy(self.net)

        self.net.train(org_training)
        return val_loss

    def encode(self, test_loader):
        """
        Encode raw data to representation using trained TS2Vec model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: representation vectors
        :rtype: dataFrame
        """

        org_training = self.net.training

        self.net.eval()
        with torch.no_grad():
            output = []
            for batch in test_loader:
                x = batch[0].permute(0, 2, 1)

                # out: shape=(batch_size, T, repr_dim)
                out = self.net(x.to(self.device, non_blocking=True), mask=None).cpu()
                output.append(out)

            output = torch.cat(output, dim=0)

        # 각 관측치의 모든 시점에 대하여 representation vector가 도출되므로 시점을 기준으로 1D max pooling을 적용하여 최종 representation 도출
        output = F.max_pool1d(
            output.transpose(1, 2),
            kernel_size=output.size(1),
        ).transpose(1, 2)
        output = output.squeeze(1)

        self.net.train(org_training)
        output = pd.DataFrame(output.numpy())
        output.columns = [f'V{i + 1}' for i in range(output.shape[-1])]
        return output