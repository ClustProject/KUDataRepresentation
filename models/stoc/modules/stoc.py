import torch
import torch.nn as nn
import torch.nn.init as init
import math

from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class STOC(nn.Module):
    def __init__(self, input_dim, feature_size, output_dim):
        super(STOC, self).__init__()
        
        self.input_dim = input_dim
        self.feature_size = feature_size
        self.output_dim = output_dim

        self.nhead = 8

        
        self.in_proj = Linear(self.input_dim, self.feature_size)

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.feature_size)

        # encoder: stacked transformer
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=self.nhead, dropout=0.1)
        self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=1)    

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=self.nhead, dropout=0.1)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=1)    

        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=self.nhead, dropout=0.1)
        self.transformer_encoder3 = nn.TransformerEncoder(self.encoder_layer3, num_layers=1)    

        # decoder: 1d conv + fully-connected layer
        self.conv_layer = nn.Conv1d(in_channels= self.feature_size*3, out_channels=self.feature_size, kernel_size=3, padding=1, bias=False, padding_mode='zeros')
        self.relu = nn.ReLU()

        self.extractor  = Linear(self.feature_size, self.output_dim)
        self.out_proj = Linear(self.output_dim, input_dim)

    def forward(self, is_train, src):
    
        # src.shape = batch, seq_len, feature 
        src = self.in_proj(src)

        # to make dim (seq_len, batch, dim)
        src = src.transpose(1, 0)
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)

        # Masked attention in Encoder layer
        output1 = self.transformer_encoder1(src, self.src_mask)
        output2 = self.transformer_encoder2(output1, self.src_mask)
        output3 = self.transformer_encoder3(output2, self.src_mask)

        output = torch.cat([output1, output2, output3], dim=2) # (seq_len, batch, dim[feature_size*3])

        output = output.permute(1, 2, 0)
        output = self.conv_layer(output) #(batch, dim[feature_size], seq_len)

        output = self.relu(output)


        output = output.transpose(-1, 1)
        output = self.extractor(output)

        # training 시, output shape: batch x seq_len x imput_dim
        # representation extraction 시, output shape: batch x seq_len x output_dim
        if is_train:
            output = self.out_proj(output) #(batch, seq_len, dim)

        return output


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)