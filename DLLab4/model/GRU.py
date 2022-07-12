# 单层单向GRU
import torch.nn as nn
import torch


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.rnn = BasicGRU(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(hidden_size, 64)
        self.linear2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, x = self.rnn(x)
        x = self.sigmoid(x)
        x = self.sigmoid(self.linear1(x.contiguous().view(x.size(0), -1)))
        return self.softmax(self.linear2(x))

class RegGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegGRU, self).__init__()
        self.rnn = BasicGRU(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, 288)

    def forward(self, x):
        _, x = self.rnn(x)
        x = self.relu(x)
        return self.linear(x.contiguous().view(x.size(0), -1))


class BasicGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.weight_xr = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.zeros(hidden_size, 1))
        self.weight_xz = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_hz = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias_z = nn.Parameter(torch.zeros(hidden_size, 1))
        self.weight_x = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_h = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size, 1))
        nn.init.xavier_normal(self.weight_x)
        nn.init.xavier_normal(self.weight_h)
        nn.init.xavier_normal(self.weight_xz)
        nn.init.xavier_normal(self.weight_hz)
        nn.init.xavier_normal(self.weight_xr)
        nn.init.xavier_normal(self.weight_hr)

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
        r = self.sigmoid(self.weight_xr @ x + self.weight_hr @ h + self.bias_r)
        z = self.sigmoid(self.weight_xz @ x + self.weight_hz @ h + self.bias_z)
        ht = self.tanh(self.weight_x @ x + self.weight_h @ (r * h) + self.bias)
        return z * h + (1 - z) * ht
