import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict
from pathlib import Path

class Mod(nn.Module):
    def __init__(self, io_channels: int, num_layers: int, kernel: int = 3) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(io_channels, io_channels, kernel_size=kernel))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
def factory(layer_def: Dict[str, int]) -> Callable:
    if "kernel"  in layer_def.keys():
        return Mod(layer_def["io_channels"], layer_def["num_layers"], layer_def["kernel"])
    else:
        return Mod(layer_def["io_channels"], layer_def["num_layers"])