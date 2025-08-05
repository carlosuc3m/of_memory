import torch
import torch.nn as nn
from config.config import Options

from . import util
from .feature_extractor import FeatureExtractor
from .pyramid_flow_estimator import PyramidFlowEstimator
from .fusion_carlos import Fusion

import torch.nn.functional as F

from .unet_parts import *

def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.2, inplace=True)

class OFMNet(nn.Module):
    """
    End-to-end learned neural frame interpolator.
    
    Takes two images x0, x1 and a time tensor t, and predicts the
    intermediate frame at t (in [0,1]). Returns a dict with keys:
      'image': the interpolated RGB image,
      plus, if config.use_aux_outputs, various warps and flow pyramids.
    """
    def __init__(self, config: Options):
        super().__init__()
        if config.pyramid_levels < config.fusion_pyramid_levels:
            raise ValueError(
                "config.pyramid_levels must be >= config.fusion_pyramid_levels."
            )
        self.config = config

        # Siamese feature extractor
        self.feature_extractor = FeatureExtractor(config)
        # Shared flow predictor
        self.predict_flow = PyramidFlowEstimator(config)
        # Fusion (decoder) network
        self.fusion = Fusion(config)

        n_chans = 16
        self.inc = DoubleConv(1, n_chans)
        self.down1 = Down(n_chans, n_chans * 2)
        n_chans *= 2
        self.down2 = Down(n_chans, n_chans * 2)
        n_chans *= 2
        self.down3 = Down(n_chans, n_chans * 2)
        n_chans *= 2
        # factor = 2 if bilinear else 1
        self.down4 = Down(n_chans, n_chans * 2)
        n_chans *= 2

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, encoding0: torch.Tensor):
        """
        x0, x1: [B, C, H, W] floats
        time:  [B] or [B,1] float in [0,1]; we fix to 0.5 by default.
        """
        # Build image pyramids: list of tensors from full res downwards
        img_pyr0 = util.build_image_pyramid(x0, self.config)
        img_pyr1 = util.build_image_pyramid(x1, self.config)
        for l2 in img_pyr1:
            print(l2.shape)

        levels_diff = x0.shape[2] / encoding0.shape[2]
        levels = 1
        while (levels_diff % 2 == 0):
            levels += 1
            levels_diff /= 2

        L = self.config.fusion_pyramid_levels
        enc_pyr = [encoding0]
        for i in range(L - 1):
            encoding0 = F.avg_pool2d(encoding0, kernel_size=2, stride=2, padding=0)
            enc_pyr.append(encoding0)

        feat_pyr0 = self.feature_extractor(img_pyr0)
        feat_pyr1 = self.feature_extractor(img_pyr1)
        for l2 in feat_pyr1:
            print(l2.shape)

        bwd_res_flow = self.predict_flow(feat_pyr1, feat_pyr0)

        # Synthesize full flows and truncate to fusion levels
        #fwd_flow_pyr = util.flow_pyramid_synthesis(fwd_res_flow)
        bwd_flow_pyr = util.flow_pyramid_synthesis(bwd_res_flow)
        bwd_flow_pyr = bwd_flow_pyr[:L]
        feat_pyr1 = feat_pyr1[-L:]
        bwd_flow_pyr_tot = [0] * L
        for i in range(L):
            aux_fl = self.down1(self.inc(bwd_flow_pyr[i][:, :1, :, :] / (2^levels)))
            aux_fl = self.down2(aux_fl)
            aux_fl = self.down3(aux_fl)
            aux_fl = self.down4(aux_fl)
            bwd_flow_pyr_1 = aux_fl
            aux_fl = self.down1(self.inc(bwd_flow_pyr[i][:, 1:, :, :] / (2^levels)))
            aux_fl = self.down2(aux_fl)
            aux_fl = self.down3(aux_fl)
            aux_fl = self.down4(aux_fl)
            bwd_flow_pyr_2 = aux_fl
            bwd_flow_pyr_tot[i] = torch.cat((bwd_flow_pyr_1.unsqueeze(2), bwd_flow_pyr_2.unsqueeze(2)), axis=2)


        k_size = int(2 ^ (levels - (self.config.pyramid_levels - L )))
        if k_size > 1:
            for i in range(L):
                feat_pyr1[i] = F.avg_pool2d(feat_pyr1[i] / k_size, kernel_size=k_size, stride=k_size, padding=0)

        to_warp_0_a = enc_pyr[:L]
        # Warp using backward warping (reads from source via flow)
        bwd_warped = util.pyramid_channel_warp(to_warp_0_a, bwd_flow_pyr_tot)
        """
        fwd_flow_on_t1 = util.pyramid_warp(fwd_flow_pyr, bwd_flow_pyr)
        # (b) Invert it (negate) so it tells us where in encoding0 to sample:
        inv_fwd_flow = [ -flow for flow in fwd_flow_on_t1 ]
        # (c) Warp encoding0 by that inverted‐forward field:
        fwd_warped = util.pyramid_warp(to_warp_0_a, inv_fwd_flow)
        # Build the aligned pyramid: [warp0, warp1, bwd_flow, fwd_flow]
        aligned = util.concatenate_pyramids(fwd_warped, bwd_warped)
        aligned = util.concatenate_pyramids(aligned, bwd_flow_pyr)
        aligned = util.concatenate_pyramids(aligned, fwd_flow_pyr)
        """
        for i in range(L):
            B, C, D, H, W = bwd_flow_pyr_tot[i].shape
            bwd_flow_pyr_tot[i] = bwd_flow_pyr_tot[i].reshape(B, C * D, H, W)
        aligned = util.concatenate_pyramids(bwd_warped, bwd_flow_pyr_tot)
        aligned = util.concatenate_pyramids(aligned, feat_pyr1)
        # Fuse to get final prediction
        pred = self.fusion(aligned)
        out = {'image': pred}  # assume final channels include RGB

        # Optionally add aux outputs for debugging/supervision
        if self.config.use_aux_outputs:
            out.update({
                #'forward_residual_flow_pyramid': fwd_res_flow,
                'backward_residual_flow_pyramid': bwd_res_flow,
                #'forward_flow_pyramid': fwd_flow_pyr,
                'backward_flow_pyramid': bwd_flow_pyr,
            })

        return out

def create_model(config: Options) -> OFMNet:
    """
    Factory to match the TF signature. In PyTorch we don't
    pre‐define input tensors; we return a nn.Module ready
    to be called with (x0, x1, time).
    """
    return OFMNet(config)
