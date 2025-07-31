# flow_estimator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from config.config import Options
from . import util

def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.2, inplace=True)

class FlowEstimator(nn.Module):
    """
    Small‐receptive‐field residual flow predictor.
    Takes two feature maps A and B, concatenates them, and
    applies a sequence of conv layers to predict a 2‐channel
    flow correction.
    """
    def __init__(self, num_convs: int, num_filters: int):
        """
        Args:
          num_convs: number of 3×3 conv layers before the 1×1s.
          num_filters: number of filters in each 3×3 conv.
        """
        super().__init__()
        layers = []

        # num_convs of 3×3 conv + leaky ReLU
        for _ in range(num_convs):
            # LazyConv2d defers in_channels inference until the first forward
            layers.append(nn.LazyConv2d(num_filters, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # one 1×1 conv (half the filters) + leaky ReLU
        half_filters = num_filters // 2
        layers.append(nn.LazyConv2d(half_filters, kernel_size=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # final 1×1 conv → 2 channels, *no* activation
        layers.append(nn.LazyConv2d(2, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
          feat_a, feat_b: [B, C, H, W] feature maps.
        Returns:
          flow_residual: [B, 2, H, W]
        """
        x = torch.cat([feat_a, feat_b], dim=1)
        return self.net(x)


class PyramidFlowEstimator(nn.Module):
    """
    Coarse‐to‐fine residual flow pyramid estimator.
    Uses a sequence of FlowEstimators: one per "specialized" level,
    then a shared estimator for all coarser levels.
    """
    def __init__(self, config: Options):
        super().__init__()
        self.levels = config.pyramid_levels
        self.specialized = config.specialized_levels

        # Build one predictor per level up to specialized_levels
        self.predictors = nn.ModuleList()
        for i in range(self.specialized):
            self.predictors.append(
                FlowEstimator(config.flow_convs[i], config.flow_filters[i])
            )
        # Shared predictor for all remaining coarser levels
        shared = FlowEstimator(config.flow_convs[-1], config.flow_filters[-1])
        for _ in range(self.specialized, self.levels):
            self.predictors.append(shared)

    def forward(self,
                feat_pyr_a: List[torch.Tensor],
                feat_pyr_b: List[torch.Tensor]
               ) -> List[torch.Tensor]:
        """
        Args:
          feat_pyr_a, feat_pyr_b: lists of [B, C, H_i, W_i] tensors,
                                  finest-first, length = pyramid_levels.
        Returns:
          residuals: list of [B, 2, H_i, W_i] tensors (finest-first),
                     where each is the correction to the upsampled flow.
        """
        L = len(feat_pyr_a)
        # Start at the coarsest level
        v = self.predictors[-1](feat_pyr_a[-1], feat_pyr_b[-1])
        residuals = [v]

        # Iterate from coarser-1 down to finest (reverse order)
        for i in reversed(range(L - 1)):
            # Upsample and scale flow to current resolution
            _, _, H_i, W_i = feat_pyr_a[i].shape
            v = F.interpolate(v * 2.0, size=(H_i, W_i),
                              mode='bilinear', align_corners=True)

            # Warp features of B at this level with the upsampled flow
            warped_b = util.warp(feat_pyr_b[i], v)

            # Predict residual at this level
            v_res = self.predictors[i](feat_pyr_a[i], warped_b)
            residuals.append(v_res)

            # Update the flow by adding the residual
            v = v + v_res
        with torch.no_grad():
            mse = F.mse_loss(feat_pyr_a[0], warped_b)
        # Return in finest-first order
        return list(reversed(residuals))
