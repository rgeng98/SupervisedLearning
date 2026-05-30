import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict
from pathlib import Path

class Mod(nn.Module):
    def __init__(self, io_dim, hidden_dim, n_layers):
        super().__init__()
        self.in_layer = nn.Linear(io_dim, hidden_dim)
        layers = []

        if n_layers > 2:

            for _ in range(n_layers-2):
                layers.append( nn.ReLU() )
                layers.append( nn.LayerNorm(hidden_dim) )
                layers.append( nn.Linear(hidden_dim, hidden_dim) )

        layers.append( nn.ReLU() )
        layers.append( nn.LayerNorm(hidden_dim) )

        self.hidden = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, io_dim)
    
    def forward(self, x):
        y = self.in_layer(x)
        y = self.hidden(y)
        y = self.out_layer(y)
        return y + x
    
def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["io_dim"], layer_def["hidden_dim"], layer_def["n_layers"])