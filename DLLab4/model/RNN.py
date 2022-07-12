# 单层单向RNN
import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
        self.rnn = BasicRNN(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_size, 64)
        self.linear2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, x = self.rnn(x)
        # x = x.permute(1, 2, 0)
        x = self.relu(x)
        x = self.relu(self.linear1(x.contiguous().view(x.size(0), -1)))
        return self.softmax(self.linear2(x))


class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_h = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.weight_x = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size, 1))
        nn.init.xavier_normal(self.weight_h)
        nn.init.xavier_normal(self.weight_x)

    def forward(self, x):
        assert x.size(2) == self.input_size, "输入向量大小错误"
        seqs = torch.split(x, 1, 1)
        h = torch.zeros(x.size(0), self.hidden_size, 1).to(x.device)
        output = []
        for seq in seqs:
            h = self.step(seq.view(x.size(0), -1, 1), h)
            output.append(h.view(x.size(0), 1, -1))
        return torch.cat(output, dim=1), h.view(x.size(0), -1)

    def step(self, x, h):
        return torch.tanh(self.weight_x @ x + self.weight_h @ h + self.bias)
