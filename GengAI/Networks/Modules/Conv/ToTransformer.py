import torch
import torch.nn as nn
from typing import Dict, Callable

class Mod(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x: tuple) -> torch.Tensor:
        y = x[0].view(x[1], -1, self.n_channels)
        return y
    
def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["n_channels"])