from of_memory_interp.hiera_large import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine
import torch
import numpy as np
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.backends.cudnn.benchmark = True

"""LARGE
model = Hiera(embed_dim=144, num_heads=2, stages=[2, 6, 36, 4], global_att_blocks=[23, 33, 43],
              window_pos_embed_bkg_spatial_size=[7, 7], window_spec=[8, 4, 16, 8]).cuda().eval()
pos_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=1000).cuda().eval()
neck = FpnNeck(position_encoding=pos_encoding, d_model=256, backbone_channel_list=[1152, 576, 288, 144],
      fpn_top_down_levels=[2, 3], fpn_interp_model="nearest").cuda().eval()
"""
""" TINY
model = Hiera(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7]).cuda()
pos_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=1000).cuda()
neck = FpnNeck(position_encoding=pos_encoding, d_model=256, backbone_channel_list=[768, 384, 192, 96],
      fpn_top_down_levels=[2, 3], fpn_interp_model="nearest").cuda()
"""
""" MINI1
model = Hiera(embed_dim=24, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9], window_pos_embed_bkg_spatial_size=[7, 7]).cuda()
pos_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=1000).cuda()
neck = FpnNeck(position_encoding=pos_encoding, d_model=256, backbone_channel_list=[256, 128, 64, 32],
      fpn_top_down_levels=[2, 3], fpn_interp_model="nearest").cuda()
"""
"""MINI2"""
model = Hiera(embed_dim=32, num_heads=1, stages=[1, 2, 3, 2], global_att_blocks=[4, 5], window_pos_embed_bkg_spatial_size=[7, 7]).cuda()
pos_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=1000).cuda()
neck = FpnNeck(position_encoding=pos_encoding, d_model=256, backbone_channel_list=[256, 128, 64, 32],
      fpn_top_down_levels=[2, 3], fpn_interp_model="nearest").cuda()


def bench_step(model, neck, im, enc0, enc1, enc2, iters=50, warmup=10):
    # warmup
    for _ in range(warmup):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = neck(model(im, enc0, enc1, enc2))
            #_ = neck(model(im))
    torch.cuda.synchronize()

    # timed loop with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            #_ = neck(model(im))
            #_ = neck(model(im))
            _ = model(im, enc0, enc1, enc2)
            #_ = model(im)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"Avg iter: {ms/1000:.6f} s")

im = torch.zeros(8,3,1024,1024, device="cuda", dtype=torch.float32)
enc0 = torch.zeros(8, 96, 256,256, device="cuda", dtype=torch.float32)
enc1 = torch.zeros(8, 192, 128,128, device="cuda", dtype=torch.float32)
enc2 = torch.zeros(8,384,64,64, device="cuda", dtype=torch.float32)
bench_step(model, neck, im, enc0, enc1, enc2)