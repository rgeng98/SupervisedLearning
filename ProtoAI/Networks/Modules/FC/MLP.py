import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict
from pathlib import Path

class Mod(nn.Module):
    def __init__(self, i_dim: int, hidden_dim: int, num_layers: int, o_dim: int) -> None:
        super().__init__()
        self.in_model = nn.Sequential(*[
            nn.LayerNorm(i_dim),
            nn.Linear(i_dim, hidden_dim), 
            nn.GELU()
        ])

        
        self.body = nn.ModuleList(nn.Sequential(*[nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]) for _ in range(num_layers - 1) )
        
        self.out_model = nn.Sequential(*[nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, o_dim)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_model(x)
        y = x
        for module in self.body:
            x = module(x)
        x = x + y
        return self.out_model(x)

def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["i_dim"], layer_def["hidden_dim"], layer_def["num_layers"], layer_def["o_dim"])