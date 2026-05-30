import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict
from pathlib import Path

class Mod(nn.Module):
    def __init__(self, i_dim: int, hidden_dim: int, num_layers: int, o_dim: int) -> None:
        super().__init__()
        if num_layers > 0:
            c = [
                nn.Linear(i_dim, hidden_dim), 
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ]
            
            for _ in range(num_layers - 1):
                c.append(nn.Linear(hidden_dim, hidden_dim))
                c.append(nn.LayerNorm(hidden_dim))
                c.append(nn.GELU())
            
            c.append(nn.Linear(hidden_dim, o_dim))
            self.classifier = nn.Sequential(*c)
        else:
            self.classifier = nn.Linear(i_dim, o_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["i_dim"], layer_def["hidden_dim"], layer_def["num_layers"], layer_def["o_dim"])