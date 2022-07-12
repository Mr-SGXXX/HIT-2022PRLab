# 单层单向LSTM
import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = BasicLSTM(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_size, 64)
        self.linear2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (x, _) = self.rnn(x)
        x = self.relu(x)
        x = self.relu(self.linear1(x.contiguous().view(x.size(0), -1)))
        return self.softmax(self.linear2(x))


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_fx = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_fh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias_f = nn.Parameter(torch.zeros(hidden_size, 1))
        self.weight_ix = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_ih = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias_i = nn.Parameter(torch.zeros(hidden_size, 1))
        self.weight_ox = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_oh = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias_o = nn.Parameter(torch.zeros(hidden_size, 1))
        self.weight_cx = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.weight_ch = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.bias_c = nn.Parameter(torch.zeros(hidden_size, 1))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_normal(self.weight_fx)
        nn.init.xavier_normal(self.weight_fh)
        nn.init.xavier_normal(self.weight_ix)
        nn.init.xavier_normal(self.weight_ih)
        nn.init.xavier_normal(self.weight_cx)
        nn.init.xavier_normal(self.weight_ch)

    def forward(self, x):
        assert x.size(2) == self.input_size, "输入向量大小错误"
        seqs = torch.split(x, 1, 1)
        h = torch.zeros(x.size(0), self.hidden_size, 1).to(x.device)
        c = torch.zeros(x.size(0), self.hidden_size, 1).to(x.device)
        output = []
        for seq in seqs:
            h, c = self.step(seq.view(x.size(0), -1, 1), h, c)
            output.append(h.view(x.size(0), 1, -1))
        return torch.cat(output, dim=1), (h.view(x.size(0), -1), c.view(x.size(0), -1))

    def step(self, x, h, c):
        f = self.sigmoid(self.weight_fx @ x + self.weight_fh @ h + self.bias_f)
        i = self.sigmoid(self.weight_ix @ x + self.weight_ih @ h + self.bias_i)
        o = self.sigmoid(self.weight_ox @ x + self.weight_oh @ h + self.bias_o)
        ct = self.tanh(self.weight_cx @ x + self.weight_ch @ h + self.bias_c)
        c = f * c + i * ct
        h = o * self.tanh(c)
        return h, c
