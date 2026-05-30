import torch
import torch.nn as nn
from typing import Callable, Dict
from pathlib import Path
from ProtoTorch.Networks.Modules.Conv import Plain

class Mod(Plain.Mod):
    def __init__(self, in_channels=3, layer_channels=[32, 64, 128, 256, 512], latent_dim=1024):
        """
        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            layer_channels (list): List of channel sizes for each consecutive layer.
            latent_dim (int): Final dimension size of the latent space vector.
        """
        super().__init__(in_channels, layer_channels)
                
        # Compress spatial dimensions to 1x1 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Map the final channel size dynamically to the requested latent dimension
        self.fc = nn.Linear(layer_channels[-1], latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        latent_vector = self.fc(x)
        return latent_vector
    
def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["in_channels"], layer_def["layer_channels"], layer_def["latent_dim"])