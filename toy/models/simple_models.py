import numpy as np
import torch
import torch.nn as nn


class SimpleNet1d(nn.Module):
    """A simple 3-layer MLP. Reference: https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py, line 198."""
    def __init__(self, data_dim, hidden_dim, sigmas, act=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)
        self.act = act
        self.sigmas = sigmas


    def forward(self, x, y):
        # x: (batch_size, data_dim)
        # y: (batch_size,), noise level index for each sample
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        x = x / used_sigmas
        return x
