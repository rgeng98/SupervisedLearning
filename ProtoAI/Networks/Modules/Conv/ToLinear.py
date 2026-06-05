import torch
import torch.nn as nn
from typing import Dict, Callable
import torch.nn.functional as F

class LambdaModule(nn.Module):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class Mod(nn.Module):
    def __init__(self, mode: str = "AvgPool") -> None:
        super().__init__()
        
        if mode.upper() == "AVGPOOL":
            self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        elif mode.upper() == "CONV":
            self.gap = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        
        elif mode.upper() == "MAX":
            self.gap = LambdaModule( lambda x: x.max(dim=1, keepdim=True).values )

        elif mode.upper() == "MEAN":
            self.gap = LambdaModule( lambda x: x.mean(dim=1, keepdim=True).values )
        

    def forward(self, x: tuple) -> torch.Tensor:
        return torch.flatten(self.gap(x[0]), start_dim=1)

def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["mode"])