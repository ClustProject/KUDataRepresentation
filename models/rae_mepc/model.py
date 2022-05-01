import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from modules.dilate_loss import dilate_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMCell(nn.Module):
    def __init__(self, ninp, nhid, device):
        super(LSTMCell, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.device = device

        self.i2h = nn.Linear(ninp, 4 * nhid)
        self.h2h = nn.Linear(nhid, 4 * nhid)

    def forward(self, input, hidden=None):
        batch_size = input.size(0)

        if hidden is None:
            hx = torch.zeros(batch_size, self.nhid).to(self.device)
            cx = torch.zeros(batch_size, self.nhid).to(self.device)
        else:
            hx, cx = hidden

        gates = self.i2h(input) + self.h2h(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = nn.Sigmoid()(ingate)
        forgetgate = nn.Sigmoid()(forgetgate)
        cellgate = nn.Tanh()(cellgate)
        outgate = nn.Sigmoid()(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * nn.Tanh()(cy)
        return hy, cy


class StackedLSTMCell(nn.Module):
    def __init__(self, nhid, nlayer, device):
        super(StackedLSTMCell, self).__init__()
        self.nhid = nhid
        self.nlayer = nlayer
        self.device = device

        lstm = []
        for l in range(nlayer):
            lstm.append(LSTMCell(nhid, nhid, device))

        self.lstm = nn.ModuleList(lstm)

    def forward(self, input, hiddens=None):
        if hiddens is None:
            hiddens = [None for l in range(self.nlayer)]

        new_hiddens = []

        for l in range(self.nlayer):
            hy, cy = self.lstm[l](input, hiddens[l])
            new_hiddens.append((hy, cy))

            input = hy
        return input, new_hiddens


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, hiddens = self.lstm(x)
        return hiddens


class FusionLayer_Enc(nn.Module):
    def __init__(self, input_size):
        super(FusionLayer_Enc, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, input_size)

    def forward(self, inputs):
        input1, input2, input3 = inputs

        input_fusion = self.fc1(input1)
        input_fusion += input2
        input_fusion = self.fc2(input_fusion)
        input_fusion += input3
        input_fusion = self.fc3(input_fusion)
        return input_fusion


class MultiEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, decay_ratio):
        super(MultiEncoder, self).__init__()
        self.window_size = window_size
        self.window_size1 = int(window_size * (decay_ratio ** 2))
        self.window_size2 = int(window_size * decay_ratio)

        self.encoder1 = Encoder(
            input_size=input_size,
            hidden_size=hidden_size
        )
        self.encoder2 = Encoder(
            input_size=input_size,
            hidden_size=hidden_size
        )
        self.encoder3 = Encoder(
            input_size=input_size,
            hidden_size=hidden_size
        )

        # Fusion layer
        self.hidden_fusion_layer = FusionLayer_Enc(hidden_size)
        self.cell_fusion_layer = FusionLayer_Enc(hidden_size)

    def forward(self, x):
        input_idx1 = np.linspace(0, (self.window_size - 1), self.window_size1, dtype=int)
        enc_hidden1 = self.encoder1(x[:, input_idx1, :])

        input_idx2 = np.linspace(0, (self.window_size - 1), self.window_size2, dtype=int)
        enc_hidden2 = self.encoder2(x[:, input_idx2, :])

        enc_hidden3 = self.encoder3(x)

        enc_hiddens = (enc_hidden1[0], enc_hidden2[0], enc_hidden3[0])
        enc_cells = (enc_hidden1[1], enc_hidden2[1], enc_hidden3[1])

        fusion_h = self.hidden_fusion_layer(enc_hiddens)
        fusion_c = self.cell_fusion_layer(enc_cells)
        fusion_enc_hidden = (fusion_h, fusion_c)
        return fusion_enc_hidden


class Decoder(nn.Module):
    def __init__(self, args, all_decode_masks, ninp, nhid, nlayer, device, skip_len=2):
        super(Decoder, self).__init__()
        self.args = args
        self.ninp = ninp
        self.nhid = nhid
        self.nlayer = nlayer
        self.skip_len = skip_len
        self.all_decode_masks = all_decode_masks
        self.device = device
        self.lambda_combine = 0.1

        self.input_layer = nn.Linear(ninp, nhid)
        self.stack_lstm = StackedLSTMCell(nhid, nlayer, device)
        self.output_layer = nn.Linear(nhid, ninp)

        self.combine_multiscales = nn.Sequential(
            nn.Linear(self.nhid * 2, 32),
            nn.PReLU(),
            nn.Linear(32, self.nhid),
        )

    def forward(self, hiddens, timestep):
        outputs = [self.output_layer(hiddens[-1][0])]
        all_hiddens = []
        all_hiddens.append(hiddens[-1][0])

        for t in range(timestep - 1):
            output = self.input_layer(outputs[-1])

            if t >= self.skip_len:
                w1 = self.all_decode_masks[t][0]
                w2 = self.all_decode_masks[t][1]
                if w1 and w2:
                    w1, w2 = 0.5, 0.5
                hiddens_combined = w1 * hiddens[-1][0] + w2 * all_hiddens[t - self.skip_len]
                hiddens[-1] = (hiddens_combined, hiddens[-1][1])

            output, hiddens = self.stack_lstm(output, hiddens)
            output = self.output_layer(hiddens[-1][0])
            all_hiddens.append(hiddens[-1][0])
            outputs.append(output)

        outputs = torch.stack(outputs, 1)
        all_hiddens = torch.stack(all_hiddens, 0)
        return outputs, all_hiddens

    def forward_combine(self, hiddens, previous_hiddens, timestep):
        outputs = [self.output_layer(hiddens[-1][0])]
        all_hiddens = []
        all_hiddens.append(hiddens[-1][0])

        for t in range(timestep - 1):
            output = self.input_layer(outputs[-1])
            if t >= self.skip_len:
                w1 = self.all_decode_masks[t][0]
                w2 = self.all_decode_masks[t][1]
                if w1 and w2:
                    w1, w2 = 0.5, 0.5
                hiddens_combined = w1 * hiddens[-1][0] + w2 * all_hiddens[t - self.skip_len]
                hiddens[-1] = (hiddens_combined, hiddens[-1][1])

            index = int(np.floor(t / self.args['tau']))
            if index >= len(previous_hiddens):
                index = -1
            hiddens_combined = self.lambda_combine * hiddens[-1][0] + (1 - self.lambda_combine) * (
                self.combine_multiscales(torch.cat((hiddens[-1][0], previous_hiddens[index]), dim=-1)))
            hiddens[-1] = (hiddens_combined, hiddens[-1][1])

            output, hiddens = self.stack_lstm(output, hiddens)
            output = self.output_layer(hiddens[-1][0])
            all_hiddens.append(hiddens[-1][0])
            outputs.append(output)

        outputs = torch.stack(outputs, 1)
        all_hiddens = torch.stack(all_hiddens, 0)
        return outputs, all_hiddens


class PredDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PredDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        _, seq_len, _ = x.size()

        h = torch.unsqueeze(hidden[0], axis=0)
        c = torch.unsqueeze(hidden[1], axis=0)
        hidden = (h, c)

        outputs = []
        for t in range(seq_len):
            temp_input = torch.unsqueeze(x[:, t, :], axis=1)
            output, hidden = self.lstm(temp_input, hidden)
            output = self.output_layer(output)
            outputs.append(output)
        outputs = torch.cat(outputs, 1)
        return outputs


class RAE_MEPC(nn.Module):
    def __init__(self, args):
        super(RAE_MEPC, self).__init__()
        self.args = args
        self.ninp = args['ninp']
        self.nhid = args['hidden_size']
        self.dec_nlayers = args['dec_nlayers']
        self.dec_Ls = args['dec_Ls']
        self.device = args['device']
        self.criterion = nn.MSELoss()
        self.lambda_dtw = 0.0001
        self.lambda_pred = 1
        self.gamma = 0.1

        self.set_encoders = MultiEncoder(self.ninp, self.nhid, args['window_length'], 1 / args['tau']).to(self.device)

        self.set_decoders = nn.ModuleList()
        for i in range(self.dec_nlayers):
            self.set_decoders.append(
                Decoder(self.args, self.args['all_decode_masks'][i], self.ninp, self.nhid, 1, self.device,
                        self.dec_Ls[i]).to(self.device))

        self.pred_decoder = PredDecoder(self.ninp, self.nhid).to(self.device)

    def get_enc_outputs(self, inputs):
        hiddens = self.set_encoders(inputs)
        hiddens = [(hiddens[0][0], hiddens[1][0])]
        return hiddens

    def get_dec_outputs(self, hiddens):
        all_final_outputs = []
        all_final_hiddens = []

        shared_hiddens = hiddens

        for i in range(self.dec_nlayers):

            dec_length = self.args['dec_lengths'][i]
            if i == 0:
                outputs, hiddens = self.set_decoders[i](shared_hiddens, dec_length)
            else:
                outputs, hiddens = self.set_decoders[i].forward_combine(shared_hiddens, hiddens, dec_length)

            all_final_outputs.append(outputs)
            all_final_hiddens.append(hiddens)

        return all_final_outputs, all_final_hiddens

    def forward(self, pure_inputs, pred_target, mode="train"):
        if mode == "train":
            # add a small amount of noise to the LSTM's input
            noise = 1e-3 * torch.randn(pure_inputs.size()).to(self.device)
            inputs = pure_inputs + noise
        else:
            inputs = pure_inputs

        # get output
        enc_hid = self.get_enc_outputs(inputs)
        all_final_outputs, _ = self.get_dec_outputs(enc_hid)
        pred_outputs = self.pred_decoder(pure_inputs, enc_hid[0])

        target_x = self.get_target(pure_inputs).to(self.args['device'])
        teachers = []
        for i in range(self.dec_nlayers):
            dowm_seq_len = self.args['dec_lengths'][i]
            step_size = int(np.floor(np.shape(target_x)[1] / dowm_seq_len))
            teachers.append(target_x[:, 0:step_size:, :])

        collected_dtw_paths = []
        loss = 0
        dtw_losses = 0
        if mode == "train":
            for i in range(len(all_final_outputs)):
                if i < (len(all_final_outputs) - 1):
                    # smooted DTW loss
                    dtw_loss, dtw_path = dilate_loss(all_final_outputs[i], target_x, self.gamma, self.args['device'])
                    collected_dtw_paths.append(dtw_path)
                    dtw_losses += dtw_loss
                else:
                    dtw_losses = dtw_losses / (len(all_final_outputs) - 1)
                    loss += self.lambda_dtw * dtw_losses
                    loss += self.criterion(all_final_outputs[i], target_x)
                    loss += self.lambda_pred * self.criterion(pred_outputs, pred_target)
        else:
            loss += self.criterion(all_final_outputs[-1], target_x)

        dec_out = all_final_outputs[-1]
        error = dec_out - target_x
        return loss, dec_out, error, enc_hid

    def get_target(self, data_x):
        # reverse as original order
        target_idx = torch.arange(data_x.size(1) - 1, -1, -1, dtype=torch.int64).to(self.device)
        target = data_x.index_select(1, target_idx)
        return target 
