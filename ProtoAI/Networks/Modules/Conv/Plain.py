# import torch.nn as nn
# from typing import Callable, Dict, List

# class Mod(nn.Module):
#     def __init__(self, in_channels: int, layer_channels: List[int], kernel: List[int], stride: List[int], padding: List[int], pooling: Dict[str, int] = None):
#         """
#         Args:
#             in_channels (int): Number of input image channels (e.g., 3 for RGB).
#             layer_channels (list): List of channel sizes for each consecutive layer.
#             latent_dim (int): Final dimension size of the latent space vector.
#         """
#         super().__init__()
        
#         layers = []
#         current_channels = in_channels
        
#         # Dynamically build the convolutional blocks based on user configuration
#         for i, next_channels in enumerate(layer_channels):
#             layers.extend([
#                 nn.Conv2d(current_channels, next_channels, kernel_size=kernel[i], stride=stride[i], padding=padding[i]),
#                 nn.GroupNorm(num_groups=1, num_channels=next_channels),
#                 nn.GELU()
#             ])

#             if pooling:
#                 # Using standard 2x2 max pooling with stride 2
#                 layers.append(nn.MaxPool2d(kernel_size=pooling["kernel"], stride=pooling["stride"]))

#             current_channels = next_channels
            
#         self.features = nn.Sequential(*layers)
        
        
#     def forward(self, x):
#         x = self.features(x)
#         return x, x.shape[0]
    
# def factory(layer_def: Dict[str, int]) -> Callable:
#     return Mod(
#         layer_def["in_channels"], 
#         layer_def["layer_channels"], 
#         layer_def["kernel"],
#         layer_def["stride"],
#         layer_def["padding"],
#         layer_def["pooling"]
#         )

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Union

class Mod(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        layer_channels: List[int], 
        kernel: Union[int, List[int]], 
        stride: Union[int, List[int]] = 1, 
        padding: Union[int, List[int]] = 1, 
        pooling: Dict[str, any] = None
    ):
        super().__init__()
        
        num_layers = len(layer_channels)
        
        # Helper to normalize int inputs into full lists matching layer depth
        def _normalize_arg(arg, name):
            if isinstance(arg, list):
                if len(arg) != num_layers:
                    raise ValueError(f"Length of {name} list must match layer_channels length.")
                return arg
            return [arg] * num_layers

        # Standardize all hyperparameters to match the number of layers
        kernels = _normalize_arg(kernel, "kernel")
        strides = _normalize_arg(stride, "stride")
        paddings = _normalize_arg(padding, "padding")
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i, next_channels in enumerate(layer_channels):
            block = nn.Sequential(
                nn.Conv2d(
                    current_channels, 
                    next_channels, 
                    kernel_size=kernels[i], 
                    stride=strides[i], 
                    padding=paddings[i],
                    bias=False # GroupNorm renders Conv2d bias redundant
                ),
                nn.GroupNorm(num_groups=1, num_channels=next_channels),
                nn.GELU()
            )
            
            # Check if this layer index specifically requires pooling
            pool_layer = None
            if pooling and i in pooling.get("layers", range(num_layers)):
                pool_layer = nn.MaxPool2d(
                    kernel_size=pooling.get("kernel", 2), 
                    stride=pooling.get("stride", 2)
                )
                
            self.layers.append(nn.ModuleDict({
                "block": block,
                "pool": pool_layer
            }))
            
            current_channels = next_channels

    def forward(self, x):
        for layer in self.layers:
            identity = x
            x = layer["block"](x)
            
            # Apply residual connection if spatial and channel dimensions match
            if identity.shape == x.shape:
                x = x + identity
                
            if layer["pool"] is not None:
                layer["pool"](x)
                
        return x, x.shape[0]
    
def factory(layer_def: Dict[str, any]) -> Callable:
    # Use .get() to prevent KeyErrors if defaults are missing in JSON
    return Mod(
        in_channels=layer_def["in_channels"], 
        layer_channels=layer_def["layer_channels"], 
        kernel=layer_def["kernel"],
        stride=layer_def.get("stride", 1),
        padding=layer_def.get("padding", 1),
        pooling=layer_def.get("pooling", None)
    )