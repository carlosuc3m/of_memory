import torch
import torch.nn as nn

from . import util
from .feature_extractor import FeatureExtractor
from .pyramid_flow_estimator import PyramidFlowEstimator
from .fusion import Fusion
from .options import Options

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
        self.feature_extractor = FeatureExtractor('feat_net', config)
        # Shared flow predictor
        self.predict_flow = PyramidFlowEstimator('predict_flow', config)
        # Fusion (decoder) network
        self.fusion = Fusion('fusion', config)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor):
        """
        x0, x1: [B, C, H, W] floats
        time:  [B] or [B,1] float in [0,1]; we fix to 0.5 by default.
        """
        # Build image pyramids: list of tensors from full res downwards
        img_pyr0 = util.build_image_pyramid(x0, self.config)
        img_pyr1 = util.build_image_pyramid(x1, self.config)

        # Extract feature pyramids (Siamese)
        feat_pyr0 = self.feature_extractor(img_pyr0)
        feat_pyr1 = self.feature_extractor(img_pyr1)

        # Estimate residual flow pyramids (forward and backward)
        fwd_res_flow = self.predict_flow(feat_pyr0, feat_pyr1)
        bwd_res_flow = self.predict_flow(feat_pyr1, feat_pyr0)

        # Synthesize full flows and truncate to fusion levels
        fwd_flow_pyr = util.flow_pyramid_synthesis(fwd_res_flow)
        bwd_flow_pyr = util.flow_pyramid_synthesis(bwd_res_flow)
        L = self.config.fusion_pyramid_levels
        fwd_flow_pyr = fwd_flow_pyr[:L]
        bwd_flow_pyr = bwd_flow_pyr[:L]

        # Use fixed mid‐time = 0.5 (override if you want arbitrary t)
        t_mid = torch.ones_like(time).view(-1) * 0.5
        # Scale flows
        bwd_flow_scaled = util.multiply_pyramid(bwd_flow_pyr,      t_mid)
        fwd_flow_scaled = util.multiply_pyramid(fwd_flow_pyr, 1.0 - t_mid)

        # Prepare pyramids to warp: stack image + features per level
        to_warp_0 = util.concatenate_pyramids(img_pyr0[:L], feat_pyr0[:L])
        to_warp_1 = util.concatenate_pyramids(img_pyr1[:L], feat_pyr1[:L])

        # Warp using backward warping (reads from source via flow)
        fwd_warped = util.pyramid_warp(to_warp_0, bwd_flow_scaled)
        bwd_warped = util.pyramid_warp(to_warp_1, fwd_flow_scaled)

        # Build the aligned pyramid: [warp0, warp1, bwd_flow, fwd_flow]
        aligned = util.concatenate_pyramids(fwd_warped, bwd_warped)
        aligned = util.concatenate_pyramids(aligned, bwd_flow_scaled)
        aligned = util.concatenate_pyramids(aligned, fwd_flow_scaled)

        # Fuse to get final prediction
        pred = self.fusion(aligned)
        out = {'image': pred[..., :3]}  # assume final channels include RGB

        # Optionally add aux outputs for debugging/supervision
        if self.config.use_aux_outputs:
            out.update({
                'x0_warped': fwd_warped[0][..., :3],
                'x1_warped': bwd_warped[0][..., :3],
                'forward_residual_flow_pyramid': fwd_res_flow,
                'backward_residual_flow_pyramid': bwd_res_flow,
                'forward_flow_pyramid': fwd_flow_pyr,
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
