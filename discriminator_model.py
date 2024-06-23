import torch
import torch.nn as nn
import torch.nn.functional as F




class DiscriminatorCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.convs = nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(config.hidden_size, config.filter_sizes))

        self.droptout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * 1, 1)

        self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x  # [32, 256]

    def forward(self, x):
        # x [batch_size,sentence_len]
        x = x.unsqueeze(1)  # [batch_size,1,sentence_len] [32,1,100]
        x = x.unsqueeze(-1)  # [batch_size,1,1,sentence_len] [32,1,100,-1]
        out = x.float()
        out = self.conv_and_pool(out, self.convs)  # [32,256]
        out = self.droptout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class DiscriminatorBiLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.lstm = nn.LSTM(1, config.rnn_hidden, config.num_layers, batch_first=True,
                            dropout=config.dropout, bidirectional=True)

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1)  # shape[batch_size,sentence_len,1]
        float_x = x.float()
        out, _ = self.lstm(float_x)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class DiscriminatorFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.convs = nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(config.hidden_size, config.filter_sizes))
        self.lstm = nn.LSTM(1, config.rnn_hidden, config.num_layers, batch_first=True,
                            dropout=config.dropout, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.CNN_fc = nn.Linear(config.num_filters * 1, 1)
        self.LSTM_fc = nn.Linear(config.rnn_hidden * 2, 1)
        self.fc = nn.Linear(config.num_filters * 1 + config.rnn_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x  # [32, 256]

    def forward(self, x):
        x = x.float()  # (64,75)
        # BiLSTM
        x = x.unsqueeze(-1)
        Out_BILSTM, _ = self.lstm(x)
        Out_BILSTM = self.dropout(Out_BILSTM)
        Out_BILSTM = Out_BILSTM[:, -1, :]
        # CNN
        x = x.unsqueeze(1)
        Out_CNN = self.conv_and_pool(x, self.convs)
        Out_CNN = self.dropout(Out_CNN)

        out = torch.cat((Out_BILSTM, Out_CNN), 1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
