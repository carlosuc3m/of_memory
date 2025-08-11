import os
import bisect
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

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
        frames.append(Image.fromarray(rgb))

        

        # load all four arrays at this index
        arrs = {
            'img1': h5f['img1'][local_idx],   # uint8, (3,H,W)
            'img2': h5f['img2'][local_idx],   # uint8, (3,H,W)
            'enc1': h5f['enc1'][local_idx],   # float32, (256,64,64)
            'enc2': h5f['enc2'][local_idx],   # float32, (256,64,64)
        }

        if random.uniform(0, 1) > 0.5:
            input_1 = torch.from_numpy(arrs['img1']).float()
            enc_1   = torch.from_numpy(arrs['enc1']).float()
            input_2 = torch.from_numpy(arrs['img2']).float()
            enc_2   = torch.from_numpy(arrs['enc2']).float()
        else:
            input_2 = torch.from_numpy(arrs['img1']).float()
            enc_2   = torch.from_numpy(arrs['enc1']).float()
            input_1 = torch.from_numpy(arrs['img2']).float()
            enc_1   = torch.from_numpy(arrs['enc2']).float()

        return input_1, input_2, enc_1, enc_2

    # Make multiprocessing-friendly: avoid pickling open handles
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5s'] = [None] * len(self._h5s)
        return state

    def close(self):
        for f in self._h5s:
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._h5s = [None] * len(self._h5s)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
