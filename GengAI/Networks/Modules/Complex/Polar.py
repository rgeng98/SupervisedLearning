import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.layer = nn.Linear(i_dim, o_dim)
        self.phase = nn.Linear(i_dim, o_dim)
    def forward(self, x):
        mag = x.abs()
        phase = torch.angle(x)

        m = self.layer(mag)
        p = self.phase(phase)

        return m * torch.exp(1j*p)
