import torch.nn as nn
import torch.nn.functional as F

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.numel()
    return np


class CRNN(nn.Module):
    def __init__(self, rnn_input_dim, rnn_hidden_dim, n_rnn_layers, output_dim, drop_prob=0.,rnn_cell='LSTM'):
        super(CRNN, self).__init__()
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_rnn_layers = n_rnn_layers
        self.output_dim = output_dim
        self.rnn_cell = rnn_cell

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in',nonlinearity='relu')
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in',nonlinearity='relu')
        self.dropout2 = nn.Dropout2d(p=drop_prob)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.norm3 = nn.BatchNorm2d(128)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in',nonlinearity='relu')
        self.dropout3 = nn.Dropout2d(p=drop_prob)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2))

        if self.rnn_cell == 'GRU':
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim, n_rnn_layers, batch_first=True, bidirectional=True, dropout=drop_prob)
        else:
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim, n_rnn_layers, batch_first=True, bidirectional=True, dropout=drop_prob)
        
        self.fc1 = nn.Linear(1024, rnn_input_dim)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in',nonlinearity='relu')

        self.fc2 = nn.Linear(2*rnn_hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in',nonlinearity='relu')

                                      # Batch, Channel, Width, Height
    def forward(self, x, h, c=None):  # Size : b,  1, 256, 64
        x = self.conv1(x)             # Size : b, 32, 256, 64
        x = self.norm1(x)             # Size : b, 32, 256, 64
        x = F.relu(x)                 # Size : b, 32, 256, 64
        x = self.maxpool(x)           # Size : b, 32, 128, 32

        x = self.conv2(x)             # Size : b, 64, 128, 32
        x = self.norm2(x)             # Size : b, 64, 128, 32
        x = F.relu(x)                 # Size : b, 64, 128, 32
        x = self.maxpool(x)           # Size : b, 64,  64, 16
        x = self.dropout2(x)

        x = self.conv3(x)             # Size : b, 128, 64, 16
        x = self.norm3(x)             # Size : b, 128, 64, 16
        x = F.relu(x)                 # Size : b, 128, 64, 16
        x = self.maxpool2(x)          # Size : b, 128, 64,  8
        x = self.dropout3(x)
        
        x = x.permute(0,2,3,1)        # Size : b, 64, 8, 128
        x = x.reshape((x.shape[0], x.shape[1], 1024)) # Size : b, 64, 1024

        x = self.fc1(x)               # Size : b,  64, 64
        x = F.relu(x)                 # Size : b,  64, 64

        if self.rnn_cell == 'LSTM': 
            x, (h, c) = self.rnn(x, (h, c))   # Size : b, 64, 1024
        else:                       
            x, h = self.rnn(x, h)     # Size : b, 64, 1024
        
        x = self.fc2(x)               # Size : b, 64, 30
        x = F.log_softmax(x, dim=2)
        return x, h, c

    def init_hidden(self, batch_size):
        # Initialize the hidden state of the RNN to zeros
        weight = next(self.parameters()).data
        
        if self.rnn_cell == 'LSTM': # in LSTM we have a cell state and a hidden state
            return weight.new(2*self.n_rnn_layers, batch_size, self.rnn_hidden_dim).zero_().cuda(), weight.new(2*self.n_rnn_layers, batch_size, self.rnn_hidden_dim).zero_().cuda()
        else:                       # in GRU we only have a hidden state
            return weight.new(2*self.n_rnn_layers, batch_size, self.rnn_hidden_dim).zero_().cuda(), None

        