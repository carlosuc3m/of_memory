import os
import bisect
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader

from PIL import Image
import cv2

def load_encoder(device=torch.device("cuda")):
    # Example stub — replace with your actual model
    from sam2.build_sam import build_sam2

    sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path="/home/carlos/Downloads/sam2.1_hiera_large.pt", device=device)
    return SAM2ImagePredictor(sam_model.eval())

class LiveDataset(Dataset):
    """
    Accepts a single HDF5 path or a list of paths and exposes them as one
    concatenated dataset. Files are opened lazily per worker to avoid
    multiprocessing/pickling issues with open file handles.
    """
    def __init__(self, video_files, separation_options=[3, 5, 7, 9, 11, 13], size=100000, n_cached=180, num_threads=8):
        if isinstance(video_files, (str, os.PathLike)):
            h5_paths = [video_files]
        self.video_files = list(video_files)
        self.size = size
        self.n_cached = n_cached
        self.num_threads = num_threads
        self.sam = load_encoder()
        self.max_separation = np.max(separation_options)
        self.separation_options = separation_options
        self.rng = np.random.default_rng()
        self.shuffled = self.rng.permutation(np.arange(size))
        self.space = self.rng.integers(0, self.max_separation + 1, size=self.size)

        samples_per_vide0 = self.size // len(self.video_files)

        self.file_map = [None] * self.size
        ii = 0
        for vv in self.video_files:
            ll = min(self.size - ii, samples_per_vide0)
            self.file_map[ii : ii + ll] = [vv] * ll
            ii += samples_per_vide0
        self.file_map = np.array(self.file_map)

        self.im1_cache = torch.zeros(0, 3, 1024, 1024)
        self.im2_cache = torch.zeros(0, 3, 1024, 1024)
        self.im3_cache = torch.zeros(0, 3, 1024, 1024)
        self.enc1_cache = torch.zeros(0, 256, 64, 64)
        self.enc2_cache = torch.zeros(0, 256, 64, 64)
        self.enc3_cache = torch.zeros(0, 256, 64, 64)

    def __len__(self):
        return self.size
    
    def _get_metas(self, p, readers, metas):
        if p in readers.keys():
            return
        vr = VideoReader(p, "video", num_threads=self.num_threads)
        md = vr.get_metadata()["video"]  # keys differ by torchvision version; we only need duration+fps-ish
        # Try to get fps & duration robustly
        fps = md.get("fps", None) or md.get("framerate", None) or md.get("average_fps", None)
        if hasattr(fps, "numerator"):  # sometimes Rational
            fps = float(fps.numerator) / float(fps.denominator)
        fps = float(fps) if fps is not None else 30.0
        dur = md.get("duration", None)
        if dur is None:
            # Fallback if duration missing: assume 1h? No. Better to skip; but most builds provide duration.
            raise RuntimeError(f"Video metadata missing duration for {p}")
        dur = float(dur)
        nframes_est = max(1, int(dur * fps + 0.5))
        readers[p] = vr
        metas[p] = {"fps": fps, "duration": dur, "nframes": nframes_est}

    
    def renew_cache(self, idx):
        readers = {}
        metas = {}
        im1_list = []
        im2_list = []
        im3_list = []
        for p in self.file_map[self.shuffled[idx : idx + self.n_cached]]:
            self._get_metas(p, readers, metas)
            vr = readers[p]
            pos = random.randint(0, metas[p]["nframes"] - self.max_separation)

            separation = self.separation_options(random.randin(len(self.separation_options)))

            vr.seek(pos, keyframe=True)
            frame_count = 0
            for fr in vr:
                if frame_count == 0:
                    im1_list.append(fr["data"]) # uint8 HxWxC torch tensor (CPU)
                elif frame_count == separation - 1:
                    im3_list.append(fr["data"]) # uint8 HxWxC torch tensor (CPU)
                elif (separation - 1) / 2 == frame_count:
                    im2_list.append(fr["data"]) # uint8 HxWxC torch tensor (CPU)
                    
                frame_count += 1
                if frame_count >= separation:
                    break
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.im1_cache = self.sam._transforms.forward_batch(im1_list)
            self.im2_cache = self.sam._transforms.forward_batch(im2_list)
            self.im3_cache = self.sam._transforms.forward_batch(im3_list)

            self.enc1_cache = self.sam.model.encoder.neck(self.sam.model.encoder.trunk(self.im1_cache))
            self.enc2_cache = self.sam.model.encoder.neck(self.sam.model.encoder.trunk(self.im2_cache))
            self.enc3_cache = self.sam.model.encoder.neck(self.sam.model.encoder.trunk(self.im3_cache))


    def __getitem__(self, idx):
        if self.im1_cache.shape[0] == 0:
            self.renew_cache(idx)
        im1 = self.im1[:1]
        im2 = self.im2[:1]
        im3 = self.im3[:1]
        enc1 = self.enc1[:1]
        enc2 = self.enc2[:1]
        enc3 = self.enc3[:1]
        self.im1 = self.im1[1:]
        self.im2 = self.im2[1:]
        self.im3 = self.im3[1:]
        self.enc1 = self.enc1[1:]
        self.enc2 = self.enc2[1:]
        self.enc3 = self.enc3[1:]
        return im1, im2, im3, enc1, enc2, enc3
