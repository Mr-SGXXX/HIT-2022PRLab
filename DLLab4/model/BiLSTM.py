# 双层双向LSTM
import torch.nn as nn
import torch
from .LSTM import BasicLSTM


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.forward_LSTM = BasicLSTM(input_size, hidden_size)
        self.backward_LSTM = BasicLSTM(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(hidden_size * 2, 64)
        self.linear2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (f, _) = self.forward_LSTM(x)
        _, (b, _) = self.backward_LSTM(x.flip(dims=(1,)))
        x = torch.cat((f, b), dim=1)
        x = self.sigmoid(x)
        x = self.sigmoid(self.linear1(x.contiguous().view(x.size(0), -1)))
        return self.softmax(self.linear2(x))
