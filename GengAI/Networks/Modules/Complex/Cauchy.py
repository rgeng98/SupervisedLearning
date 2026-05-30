import torch
import torch.nn as nn

class ModReLU(nn.Module):
    def __init__(self, num_features):
        super(ModReLU, self).__init__()
        # b is a learnable real-valued bias per channel/feature
        self.b = nn.Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize bias to small negative values
        nn.init.constant_(self.b, -0.5)

    def forward(self, z):
        # Calculate the magnitude (absolute value) of the complex tensor
        abs_z = torch.abs(z)
        
        # Avoid division by zero by adding a tiny epsilon
        eps = 1e-6
        scale = abs_z + self.b
        
        # Apply ReLU to the scaled magnitude and normalize by original magnitude
        # We use clamp to mimic ReLU behavior on the magnitude
        activated_scale = torch.clamp(scale, min=0.0) / (abs_z + eps)
        
        # Multiply the scaling factor back to the complex input
        # PyTorch handles real * complex multiplication element-wise natively
        return activated_scale * z