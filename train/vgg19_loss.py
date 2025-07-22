# vgg19_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Sequence, Union

class VGG19FeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from a pretrained VGG‑19 network,
    corresponding to layers conv1_2, conv2_2, conv3_2, conv4_2, conv5_2.
    Input images are expected in [0,1] RGB; this module multiplies by 255
    and subtracts ImageNet means to match the original TF implementation.
    """
    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__()
        # Load pretrained VGG‑19 up to the last conv layer we need
        vgg_pretrained = models.vgg19(pretrained=True).features.to(device)
        # Freeze parameters
        for p in vgg_pretrained.parameters():
            p.requires_grad = False
        self.vgg = vgg_pretrained
        # Mapping from feature names to layer indices in torchvision VGG19.features
        self.layer_ids = {
            'conv1_2': 2,
            'conv2_2': 7,
            'conv3_2': 12,
            'conv4_2': 21,
            'conv5_2': 30,
        }
        # ImageNet mean in RGB order
        self.register_buffer(
            'mean',
            torch.tensor([123.6800, 116.7790, 103.9390])
                .view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] RGB in [0,1]
        Returns:
            Dict mapping each layer name to its [B,C,H_i,W_i] activations.
        """
        # Scale to [0,255] and subtract mean
        x = x * 255.0
        x = x - self.mean

        features: Dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            for name, layer_idx in self.layer_ids.items():
                if idx == layer_idx:
                    features[name] = x
            # Stop once we've collected conv5_2
            if idx >= max(self.layer_ids.values()):
                break
        return features


def _compute_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    L1 error, optionally masked.
    Args:
        pred, target: [B, C, H, W]
        mask:         [B, 1, H_m, W_m] or None – will be resized
    """
    diff = torch.abs(pred - target)
    if mask is None:
        return diff.mean()
    # resize mask to [B,1,H,W]
    mask = F.interpolate(mask, size=pred.shape[2:], mode='bilinear', align_corners=False)
    return (diff * mask).mean()


def _gram_matrix(
    feat: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes per‑batch Gram matrices.
    Args:
        feat: [B, C, H, W]
        mask: [B, 1, H_m, W_m] or None – will be resized
    Returns:
        [B, C, C] Gram matrices normalized by H*W
    """
    B, C, H, W = feat.shape
    x = feat.view(B, C, -1)  # [B, C, N]
    if mask is not None:
        mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        mask = mask.view(B, 1, -1)  # [B,1,N]
        x = x * mask
    # Gram = x @ xᵀ / (H*W)
    return torch.bmm(x, x.transpose(1, 2)) / float(H * W)


def vgg_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    extractor: VGG19FeatureExtractor,
    weights: Optional[Sequence[float]] = None,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perceptual VGG loss: weighted L1 between feature maps of prediction vs. target.
    Args:
        prediction, target: [B,3,H,W] in [0,1]
        extractor:         a VGG19FeatureExtractor instance
        weights:           five-layer weights; defaults to [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        mask:              optional [B,1,H,W] mask for per-pixel weighting
    """
    if weights is None:
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10.0/1.5]

    feat_pred = extractor(prediction)
    feat_targ = extractor(target)

    losses = []
    layer_names = ['conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']
    for w, name in zip(weights, layer_names):
        fp, ft = feat_pred[name], feat_targ[name]
        losses.append(_compute_l1(fp, ft, mask) * w)

    loss = sum(losses) / 255.0
    return loss


def style_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    extractor: VGG19FeatureExtractor,
    weights: Optional[Sequence[float]] = None,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Style loss: weighted L2 between Gram matrices of feature maps.
    Args:
        prediction, target: [B,3,H,W] in [0,1]
        extractor:         a VGG19FeatureExtractor instance
        weights:           five-layer weights; if None, same defaults as vgg_loss
        mask:              optional [B,1,H,W] mask
    """
    if weights is None:
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10.0/1.5]

    feat_pred = extractor(prediction)
    feat_targ = extractor(target)

    losses = []
    layer_names = ['conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']
    for w, name in zip(weights, layer_names):
        gp = _gram_matrix(feat_pred[name], mask)
        gt = _gram_matrix(feat_targ[name], mask)
        # squared Frobenius norm (mean squared difference)
        losses.append(F.mse_loss(gp, gt, reduction='mean') * w)

    return sum(losses)
