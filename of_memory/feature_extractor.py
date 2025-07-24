# feature_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from config.config import Options

def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.2, inplace=True)

class SubTreeExtractor(nn.Module):
    """
    Conventional, hierarchical feature extractor that produces a small pyramid
    of features from a single input image (or feature map).
    """
    def __init__(self, config: Options):
        """
        Args:
            config.filters: base number of conv filters (k).
            config.sub_levels: depth of the subtree (n).
        """
        super().__init__()
        k = config.filters
        n = config.sub_levels

        # Build 2 convs per level: conv3x3 → activation → conv3x3 → activation
        self.convs = nn.ModuleList()
        for i in range(n):
            out_ch = k << i
            # LazyConv2d will infer in_channels on first forward pass
            self.convs.append(nn.LazyConv2d(out_ch, kernel_size=3, padding=1))
            self.convs.append(nn.LazyConv2d(out_ch, kernel_size=3, padding=1))

        # Average‐pool to downsample between levels
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor, n_levels: int) -> List[torch.Tensor]:
        """
        Args:
            x:        [B, C, H, W] input tensor.
            n_levels: number of pyramid levels to extract (≤ config.sub_levels).
        Returns:
            List of length n_levels, each element [B, out_ch_i, H_i, W_i].
        """
        head = x
        pyramid: List[torch.Tensor] = []

        for i in range(n_levels):
            # First conv + activation
            head = _leaky_relu(self.convs[2*i](head))
            # Second conv + activation
            head = _leaky_relu(self.convs[2*i + 1](head))
            pyramid.append(head)

            # Downsample for next level, if any
            if i < n_levels - 1:
                head = self.pool(head)

        return pyramid


class FeatureExtractor(nn.Module):
    """
    Cascaded feature pyramid extractor. Uses a shared SubTreeExtractor at each
    resolution level to build a concatenated, semantic-consistent feature pyramid.
    """
    def __init__(self, config: Options):
        """
        Args:
            config.filters:     base filter count for SubTreeExtractor.
            config.sub_levels:  subtree depth.
        """
        super().__init__()
        self.sub_levels = config.sub_levels
        self.extract_sublevels = SubTreeExtractor(config)

    def forward(self, image_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            image_pyramid: list of images [B, C, H_i, W_i], finest-first.
        Returns:
            cascaded feature pyramid: list of [B, C_feat, H_i, W_i].
        """
        L = len(image_pyramid)
        sub_pyramids: List[List[torch.Tensor]] = []

        # 1) For each image level, extract a small subtree of features
        for i in range(L):
            # don't exceed the coarsest level
            capped = min(L - i, self.sub_levels)
            sub_pyramids.append(
                self.extract_sublevels(image_pyramid[i], capped)
            )

        # 2) Build the cascaded pyramid by concatenating diagonals of sub-pyramids
        feature_pyramid: List[torch.Tensor] = []
        for i in range(L):
            # start with the finest feature at this level
            feats = sub_pyramids[i][0]
            # concatenate deeper subtree features from finer-to-coarser
            for j in range(1, self.sub_levels):
                if j <= i:
                    feats = torch.cat(
                        [feats, sub_pyramids[i - j][j]],
                        dim=1
                    )
            feature_pyramid.append(feats)

        return feature_pyramid
