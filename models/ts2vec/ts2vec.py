import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .modules import TSEncoder
from .modules.losses import hierarchical_contrastive_loss
from .utils import take_per_row, torch_pad_nan
from einops import rearrange
import copy


class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        temporal_unit=0
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.n_epochs = 0
        
        self.best_val_loss = 1e+08
    
    def fit(self, train_loader, valid_loader, n_epochs=None, verbose=True):
        ''' Training the TS2Vec model.
        
        Args:
            train_loader (DataLoader): train dataloader.
            valid_loader (DataLoader): valid dataloader.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log (list): a list containing the training losses on each epoch.
            best_model (model): trained TS2Vec Encoder. 
        '''
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
                        
            cum_loss = 0
            n_epoch_iters = 0
                        
            for batch in train_loader:
                
                x = batch[0].permute(0,2,1)
                x = x.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
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
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
                
            self.save_best_model(valid_loader)
            
        return loss_log, self.best_model
    
    def save_best_model(self, valid_loader):
        ''' Save the beset model.
        
        Args:
            valid_loader (DataLoader): valid dataloader.
        '''
        val_loss = 0
        n_iters = 0
        org_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            for batch in valid_loader:

                x = batch[0].permute(0,2,1)
                x = x.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

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
        
        if val_loss < self.best_val_loss: 
            print(f"Val Loss : {self.best_val_loss} >>> {val_loss}")
            self.best_val_loss = val_loss
            self.best_model = copy.deepcopy(self.net)
            
        self.net.train(org_training)
        
    def encode(self, test_loader):
        ''' Compute representations using the model.
        
        Args:
            test_loader (DataLoader): test dataset loader.

        Returns:
            output (dataframe): Output representation vector.
        '''

        org_training = self.net.training
        self.net.eval()
        
        with torch.no_grad():
            output = []
            for batch in test_loader:
                x = batch[0].permute(0,2,1)
                out = self.net(x.to(self.device, non_blocking=True), mask=None).cpu()
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        output = F.max_pool1d(
                output.transpose(1, 2),
                kernel_size = output.size(1),
            ).transpose(1, 2)
        output = output.squeeze(1)
        
        self.net.train(org_training)
        output = pd.DataFrame(output.numpy())
        output.columns = [f'V{i+1}' for i in range(output.shape[-1])]
        return output
    