# fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from config.config import Options

_NUMBER_OF_COLOR_CHANNELS = 256

class Fusion(nn.Module):
    """
    The decoder ("fusion") stage of FilmNet, implemented in PyTorch.
    
    Takes a list of warped image+feature pyramids (finest-first) and
    progressively upsamples and merges them U‑Net style to produce
    a final RGB image.
    """
    def __init__(self, config: Options):
        super().__init__()
        self.levels = config.fusion_pyramid_levels
        
        # Build per‑level conv modules (fine to coarse, index 0=finest)
        self.convs = nn.ModuleList()
        k = config.filters
        m = config.specialized_levels
        
        for i in range(self.levels - 1):
            # Number of output channels at this level
            num_filters = (k << i) if i < m else (k << m)
            if i == 0:
                in_chans = 240 * 2
            
            level_convs = nn.ModuleList([
                nn.Conv2d(in_channels=in_chans, out_channels=num_filters, kernel_size=2, padding='same'),
                nn.Conv2d(in_channels=num_filters + 240, out_channels=num_filters, kernel_size=3, padding='1'),
                nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding='1'),
            ])
            self.convs.append(level_convs)
            in_chans = num_filters + 240
        
        # Final 1×1 conv to produce RGB output
        self.output_conv = nn.Conv2d(config.filters, _NUMBER_OF_COLOR_CHANNELS, kernel_size=1)
    
    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pyramid: list of length `fusion_pyramid_levels`, each
                     a float tensor [B, C_i, H_i, W_i], finest-first.
        Returns:
            Tensor [B, 3, H, W]: the fused RGB image.
        """
        if len(pyramid) != self.levels:
            raise ValueError(
                f"Expected {self.levels} pyramid levels, but got {len(pyramid)}"
            )
        
        # Start from the coarsest level (last in list)
        net = pyramid[-1]
        
        # Decode / fuse from coarse → fine
        for i in reversed(range(self.levels - 1)):
            # Upsample from previous coarser output to current resolution
            B, C, H_i, W_i = pyramid[i].shape
            net = F.interpolate(net, size=(H_i, W_i), mode='nearest')
            
            # 2×2 conv (no activation), then concat skip connection
            net = self.convs[i][0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            
            # Two 3×3 convs with leaky ReLU
            net = F.leaky_relu(self.convs[i][1](net), negative_slope=0.2)
            net = F.leaky_relu(self.convs[i][2](net), negative_slope=0.2)
        
        # Final 1×1 → RGB
        out = self.output_conv(net)
        return out
