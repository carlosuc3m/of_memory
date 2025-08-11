import os
import bisect
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset

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
    def __init__(self, video_files, max_separation=10, size=100000):
        if isinstance(video_files, (str, os.PathLike)):
            h5_paths = [video_files]
        self.video_files = list(video_files)
        self.size = size
        self.sam = load_encoder()
        self.max_separation = max_separation
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

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        vv = self.file_map[self.shuffled[idx]]
        cap = cv2.VideoCapture(vv)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.rng.integers(0, total - self.space[idx]))
        ok, frame = cap.read()
        cap.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor1 = self.sam._transforms(img)[None, ...].cpu().numpy()
        features, pos = self.sam.model.encoder.neck(self.sam.model.encoder.trunk(tensor1))

        return tensor1, features, pos
