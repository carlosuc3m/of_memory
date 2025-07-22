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

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, encoding0: torch.Tensor):
        """
        x0, x1: [B, C, H, W] floats
        time:  [B] or [B,1] float in [0,1]; we fix to 0.5 by default.
        """
        # Build image pyramids: list of tensors from full res downwards
        img_pyr0 = util.build_image_pyramid(x0, self.config)
        img_pyr1 = util.build_image_pyramid(x1, self.config)

        enc_pyr = util.build_image_pyramid(encoding0, self.config)


        # Extract feature pyramids (Siamese)
        feat_pyr0 = self.feature_extractor(img_pyr0)
        feat_pyr1 = self.feature_extractor(img_pyr1)

        enc_feat_pyr = self.feature_extractor(enc_pyr)


        # Estimate residual flow pyramids (forward and backward)
        bwd_res_flow = self.predict_flow(feat_pyr1, feat_pyr0)

        # Synthesize full flows and truncate to fusion levels
        bwd_flow_pyr = util.flow_pyramid_synthesis(bwd_res_flow)
        L = self.config.fusion_pyramid_levels
        bwd_flow_pyr = bwd_flow_pyr[:L]

        # Prepare pyramids to warp: stack image + features per level
        to_warp_0_a = util.concatenate_pyramids(enc_pyr[:L], enc_feat_pyr[:L])
        # Warp using backward warping (reads from source via flow)
        bwd_warped = util.pyramid_warp(to_warp_0_a, bwd_flow_pyr)

        # Build the aligned pyramid: [warp0, warp1, bwd_flow, fwd_flow]
        aligned = util.concatenate_pyramids(bwd_warped, bwd_flow_pyr)

        # Fuse to get final prediction
        pred = self.fusion(aligned)
        out = {'image': pred[..., :3]}  # assume final channels include RGB

        # Optionally add aux outputs for debugging/supervision
        if self.config.use_aux_outputs:
            out.update({
                'x1_warped': bwd_warped[0][..., :3],
                'backward_residual_flow_pyramid': bwd_res_flow,
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
