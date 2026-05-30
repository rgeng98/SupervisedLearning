import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict
from pathlib import Path

class Mod(nn.Module):
    def __init__(self, i_dim: int, hidden_dim: int, num_heads: int, pooling: Callable = None):
        super().__init__()
        self.k = nn.Linear(i_dim, hidden_dim)
        self.q = nn.Linear(i_dim, hidden_dim)
        self.v = nn.Linear(i_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads)
        if pooling:
            self.pool = pooling
        else:
            self.pool = torch.max

    def forward(self, x):
        # x = [batch, seq, dim]
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        attn_output = self.mha(k, q, v)
        # Return the average across the sequence length
        return attn_output[0].mean(dim=1)


def factory(layer_def: Dict[str, int]) -> Callable:
    if "pooling" in layer_def.keys():
        pooling = layer_def["pooling"]
    else:
        pooling = None

    return Mod(layer_def["i_dim"], layer_def["hidden_dim"], layer_def["num_heads"], pooling=pooling)