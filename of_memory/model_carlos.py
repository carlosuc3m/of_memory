import torch
import torch.nn as nn
from config.config import Options

from . import util
from .feature_extractor import FeatureExtractor
from .pyramid_flow_estimator import PyramidFlowEstimator
from .fusion_carlos import Fusion

import torch.nn.functional as F


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

        self.shared_adapter = nn.ModuleList([nn.LazyConv2d(256, kernel_size=3, padding=1),
                               nn.LazyConv2d(256, kernel_size=7, padding=3),
                               nn.LazyConv2d(256, kernel_size=1, padding=0)])
        self.non_shared_adapter = nn.ModuleList()
        for i in range(config.specialized_levels):
            self.non_shared_adapter.append(nn.ModuleList([nn.LazyConv2d(256, kernel_size=3, padding=1),
                                             nn.LazyConv2d(256, kernel_size=7, padding=3),
                                             nn.LazyConv2d(256, kernel_size=1, padding=0)]))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, encoding0: torch.Tensor):
        """
        x0, x1: [B, C, H, W] floats
        time:  [B] or [B,1] float in [0,1]; we fix to 0.5 by default.
        """
        # Build image pyramids: list of tensors from full res downwards
        img_pyr0 = util.build_image_pyramid(x0, self.config)
        img_pyr1 = util.build_image_pyramid(x1, self.config)

        levels_diff = x0.shape[2] / encoding0.shape[2]
        levels = 1
        while (levels_diff % 2 == 0):
            levels += 1
            levels_diff /= 2

        enc_pyr = [encoding0]
        for i in range(levels - 1):
            if i < levels - self.config.specialized_levels - 1:
                encoding0 = _leaky_relu(self.shared_adapter[0](encoding0))
                encoding0 = F.interpolate(encoding0, size=(encoding0.shape[2] * 2, encoding0.shape[2] * 2),
                                mode='bilinear', align_corners=True)
                encoding0 = _leaky_relu(self.shared_adapter[1](encoding0))
                encoding0 = self.shared_adapter[2](encoding0)
            else:
                encoding0 = _leaky_relu(self.non_shared_adapter[i - self.config.specialized_levels][0](encoding0))
                encoding0 = F.interpolate(encoding0, size=(encoding0.shape[2] * 2, encoding0.shape[2] * 2),
                                mode='bilinear', align_corners=True)
                encoding0 = _leaky_relu(self.non_shared_adapter[i - self.config.specialized_levels][1](encoding0))
                encoding0 = self.non_shared_adapter[i - self.config.specialized_levels][2](encoding0)
            enc_pyr.append(encoding0)

        enc_pyr = list(reversed(enc_pyr))
        #enc_pyr2 = util.build_image_pyramid(_leaky_relu(self.adapterconv(encoding0)), self.config)


        # Extract feature pyramids (Siamese)
        feat_pyr0 = self.feature_extractor(img_pyr0)
        feat_pyr1 = self.feature_extractor(img_pyr1)

        ## OG enc_feat_pyr = self.feature_extractor(enc_pyr)


        # Estimate residual flow pyramids (forward and backward)
        #fwd_res_flow = self.predict_flow(feat_pyr0, feat_pyr1)
        bwd_res_flow = self.predict_flow(feat_pyr1, feat_pyr0)

        # Synthesize full flows and truncate to fusion levels
        #fwd_flow_pyr = util.flow_pyramid_synthesis(fwd_res_flow)
        bwd_flow_pyr = util.flow_pyramid_synthesis(bwd_res_flow)
        L = self.config.fusion_pyramid_levels
        #fwd_flow_pyr = fwd_flow_pyr[:L]
        bwd_flow_pyr = bwd_flow_pyr[:L]

        # Prepare pyramids to warp: stack image + features per level
        to_warp_0_a = enc_pyr[:L]
        # Warp using backward warping (reads from source via flow)
        bwd_warped = util.pyramid_warp(to_warp_0_a, bwd_flow_pyr)
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

        aligned = util.concatenate_pyramids(bwd_warped, bwd_flow_pyr)
        print(bwd_warped[0].shape)
        print(bwd_flow_pyr[0].shape)
        print(aligned[0].shape)
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
