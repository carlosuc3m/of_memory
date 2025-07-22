# util.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .options import Options

def build_image_pyramid(image: torch.Tensor,
                        options: Options) -> List[torch.Tensor]:
    """
    Builds an image pyramid by repeatedly downsampling by a factor of 2.
    
    Args:
        image: Tensor of shape [B, C, H, W].
        options: Options object with attribute `pyramid_levels`.
    
    Returns:
        List of Tensors, length `options.pyramid_levels`,
        from full resolution down to coarsest.
    """
    pyramid: List[torch.Tensor] = []
    curr = image
    for level in range(options.pyramid_levels):
        pyramid.append(curr)
        if level < options.pyramid_levels - 1:
            # Average‐pool down by 2×2
            curr = F.avg_pool2d(curr, kernel_size=2, stride=2, padding=0)
    return pyramid


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Backward warp an image (or feature) tensor according to a flow field.
    
    For each output pixel (y, x), samples input at (y + flow_y, x + flow_x).
    
    Args:
        image: [B, C, H, W] float tensor.
        flow:  [B, 2, H, W] float tensor, channels = (dx, dy) in pixels.
    
    Returns:
        Warped image: [B, C, H, W].
    """
    B, C, H, W = image.shape
    # Create mesh grid of pixel coordinates
    device = image.device
    # y coords: 0..H-1, x coords: 0..W-1
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    # Stack to shape [H, W, 2], then expand to [B, H, W, 2]
    base_grid = torch.stack((grid_x, grid_y), dim=2).float()  # [H, W, 2]
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)      # [B, H, W, 2]

    # Permute flow to [B, H, W, 2]
    flow_perm = flow.permute(0, 2, 3, 1)

    # Compute sampling coordinates
    coords = base_grid + flow_perm  # [B, H, W, 2]

    # Normalize to [-1, 1] for grid_sample
    coords_x = 2.0 * coords[..., 0] / (W - 1) - 1.0
    coords_y = 2.0 * coords[..., 1] / (H - 1) - 1.0
    sampling_grid = torch.stack((coords_x, coords_y), dim=3)  # [B, H, W, 2]

    # Perform bilinear sampling
    warped = F.grid_sample(
        image, sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return warped


def multiply_pyramid(pyramid: List[torch.Tensor],
                     scalar: torch.Tensor) -> List[torch.Tensor]:
    """
    Multiply each tensor in a pyramid by a per-example scalar.
    
    Args:
        pyramid: List of [B, C, H, W] tensors.
        scalar:  [B] or [B, 1] tensor.
    
    Returns:
        List of [B, C, H, W] tensors, each multiplied by scalar[b].
    """
    B = scalar.shape[0]
    s = scalar.view(B, 1, 1, 1).to(pyramid[0].dtype)
    return [level * s for level in pyramid]


def flow_pyramid_synthesis(
    residual_pyramid: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Convert a residual flow pyramid into a full flow pyramid.
    
    Args:
        residual_pyramid: List of [B, 2, H_i, W_i] tensors, finest-first.
    
    Returns:
        List of [B, 2, H_i, W_i] full flows, finest-first.
    """
    # Start from the coarsest scale
    flow = residual_pyramid[-1]
    flow_pyr = [flow]
    # Iterate from coarsest-1 up to finest
    for res in reversed(residual_pyramid[:-1]):
        _, _, h, w = res.shape
        # Upsample flow (multiply by 2 for pixel-space consistency)
        flow = F.interpolate(flow * 2.0, size=(h, w),
                             mode='bilinear', align_corners=True)
        flow = res + flow
        flow_pyr.append(flow)
    # Reverse to get finest-first ordering
    return list(reversed(flow_pyr))


def pyramid_warp(feature_pyramid: List[torch.Tensor],
                 flow_pyramid: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Warp each level of a feature pyramid using its corresponding flow.
    
    Args:
        feature_pyramid: List of [B, C_i, H_i, W_i] tensors.
        flow_pyramid:    List of [B, 2, H_i, W_i] tensors.
    
    Returns:
        List of warped [B, C_i, H_i, W_i] tensors.
    """
    return [
        warp(feat, flow)
        for feat, flow in zip(feature_pyramid, flow_pyramid)
    ]


def concatenate_pyramids(pyramid1: List[torch.Tensor],
                         pyramid2: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Concatenate two pyramids level-by-level along the channel dimension.
    
    Args:
        pyramid1, pyramid2: Lists of [B, C1_i, H_i, W_i] and [B, C2_i, H_i, W_i].
    
    Returns:
        List of [B, C1_i + C2_i, H_i, W_i] tensors.
    """
    return [
        torch.cat([l1, l2], dim=1)
        for l1, l2 in zip(pyramid1, pyramid2)
    ]
