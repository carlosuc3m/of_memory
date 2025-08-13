import torch
import torch.nn as nn
from config.config import Options

import time

from of_memory_interp import util
from of_memory_interp.feature_extractor import FeatureExtractor
from of_memory_interp.pyramid_flow_estimator import PyramidFlowEstimator
from of_memory_interp.fusion import Fusion

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


    def forward(self, x0: torch.Tensor, x1: torch.Tensor,
                encoding0: torch.Tensor, encoding1: torch.Tensor):
        """
        x0, x1: [B, C, H, W] floats
        time:  [B] or [B,1] float in [0,1]; we fix to 0.5 by default.
        """
        # Build image pyramids: list of tensors from full res downwards
        tt = time.time()
        img_pyr0 = util.build_image_pyramid(x0, self.config)
        img_pyr1 = util.build_image_pyramid(x0, self.config)
        print("pyr", time.time() - tt)
        tt = time.time()

        enc_pyr0 = util.build_image_pyramid(encoding0, self.config)[:3]
        enc_pyr1 = util.build_image_pyramid(encoding1, self.config)[:3]
        print("enc pyr", time.time() - tt)
        tt = time.time()


        # Extract feature pyramids (Siamese)
        feat_pyr0 = self.feature_extractor(img_pyr0)
        feat_pyr1 = self.feature_extractor(img_pyr1)
        print("feats", time.time() - tt)
        tt = time.time()


        # Estimate residual flow pyramids (forward and backward)
        fwd_res_flow = self.predict_flow(feat_pyr0, feat_pyr1)[-3:]
        bwd_res_flow = self.predict_flow(feat_pyr1, feat_pyr0)[-3:]
        print("flows", time.time() - tt)
        tt = time.time()

        # Synthesize full flows and truncate to fusion levels
        fwd_flow_pyr = util.flow_pyramid_synthesis(fwd_res_flow)
        bwd_flow_pyr = util.flow_pyramid_synthesis(bwd_res_flow)
        print("synth", time.time() - tt)
        tt = time.time()
        #L = self.config.fusion_pyramid_levels
        #fwd_flow_pyr = fwd_flow_pyr[:L]
        #bwd_flow_pyr = bwd_flow_pyr[:L]
        scale = torch.tensor([0.5], device=fwd_flow_pyr[0].device, dtype=fwd_flow_pyr[0].dtype)
        bwd_flow_pyr = util.multiply_pyramid(fwd_flow_pyr, scale)
        fwd_flow_pyr = util.multiply_pyramid(bwd_flow_pyr, scale)

        # Prepare pyramids to warp: stack image + features per level
        # Warp using backward warping (reads from source via flow)
        bwd_warped = util.pyramid_warp(enc_pyr0, bwd_flow_pyr)
        fwd_warped = util.pyramid_warp(enc_pyr1, fwd_flow_pyr)
        print("warp", time.time() - tt)
        tt = time.time()

        aligned = util.concatenate_pyramids(bwd_warped, fwd_warped)
        aligned = util.concatenate_pyramids(aligned, bwd_flow_pyr)
        aligned = util.concatenate_pyramids(aligned, fwd_flow_pyr)
        print("align", time.time() - tt)
        tt = time.time()
        # Fuse to get final prediction
        pred = self.fusion(aligned)
        print("pred", time.time() - tt)
        tt = time.time()
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

@torch.no_grad
def main():

    ## OG PARAMS
    pyramid_levels = 7
    fusion_pyramid_levels = 5
    specialized_levels = 3
    sub_levels = 4
    flow_convs = [3, 3, 3, 3]
    flow_filters = [32, 64, 128, 256]
    filters = 64
    config = Options(pyramid_levels=pyramid_levels,
                            fusion_pyramid_levels=fusion_pyramid_levels,
                            specialized_levels=specialized_levels,
                            flow_convs=flow_convs,
                            flow_filters=flow_filters,
                            sub_levels=sub_levels,
                            filters=filters,
                            use_aux_outputs=True)
    model = OFMNet(config).cpu().eval()

    im1 = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32, device="cpu")
    im2 = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32, device="cpu")
    enc1 = torch.zeros((1, 256, 64, 64), dtype=torch.float32, device="cpu")
    enc2 = torch.zeros((1, 256, 64, 64), dtype=torch.float32, device="cpu")
    tt = time.time()
    model(im1, im2, enc1, enc2)
    print(time.time() - tt)

if __name__ == '__main__':
    main()