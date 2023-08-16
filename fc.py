from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable

class FCNet(nn.Module):
    def __init__(self, dims):
        super(FCNet, self).__init__()
        self.main = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.main(x)

class FFN(nn.Module):
    def __init__(self, inp_size, mid_size, out_size, drop_rate=0.1, use_relu=True):
        super(FFN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(inp_size, mid_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate))
        self.main = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.main(self.fc(x))
