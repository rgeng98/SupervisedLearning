import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self, i_dim, o_dim):
        super().__init__()
        self.layer_key = nn.Linear(i_dim, o_dim)
        self.layer_query = nn.Linear(i_dim, o_dim)
        self.phase_value = nn.Linear(i_dim, o_dim)
        
    def forward(self, x):
        mag = x.abs()
        phase = torch.angle(x)
        
        # Keys and queries are governed by the magnitude of the signal - amplitude can be used to separate signal sources and phase is important
        k = self.layer_key(mag)
        q = self.layer_query(mag)
        v = self.phase_value(phase)

        attn_output = self.mha(k, q, v)

        return attn_output[0].mean(1)
