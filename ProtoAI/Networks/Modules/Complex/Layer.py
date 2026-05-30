import torch
import torch.nn as nn
import torch.nn.functional as F

class FcLayer(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.layer = nn.Linear(i_dim, o_dim, dtype = torch.cfloat)
    
    def forward(self, x):
        x = self.layer(x)
        x = x / x.abs().mean()
        return x
