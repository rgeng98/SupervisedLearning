import torch
import torch.nn as nn
from typing import Callable, Dict
from pathlib import Path

class Mod(nn.Module):
    def __init__(self, latent_dim=1024, layer_channels=[512, 256, 128, 64, 32], out_channels=3):
        """
        Args:
            latent_dim (int): Input dimension size of the latent space vector.
            layer_channels (list): Channel sizes for the upsampling layers (usually reversed encoder channels).
            out_channels (int): Number of output image channels (e.g., 3 for RGB).
        """
        super().__init__()
        
        # Match the starting channel configuration of the first decoder layer
        start_channels = layer_channels[0]
        self.fc = nn.Linear(latent_dim, start_channels * 1 * 1)
        self.start_channels = start_channels
        
        layers = []
        
        # Kickstart spatial structure: Convert 1x1 feature map to 4x4
        layers.extend([
            nn.ConvTranspose2d(start_channels, start_channels, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(start_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Dynamically build the upsampling block pipeline
        current_channels = start_channels
        for next_channels in layer_channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True)
            ])
            current_channels = next_channels
            
        # Final output layer to reconstruct original image channels
        layers.extend([
            nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.start_channels, 1, 1)
        reconstructed_image = self.decoder(x)
        return reconstructed_image
    
def factory(layer_def: Dict[str, int]) -> Callable:
    return Mod(layer_def["latent_dim"], layer_def["layer_channels"], layer_def["out_channels"])