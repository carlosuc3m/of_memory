import os
import bisect
import h5py
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class EncodingDataset(Dataset):
    """
    Accepts a single HDF5 path or a list of paths and exposes them as one
    concatenated dataset. Files are opened lazily per worker to avoid
    multiprocessing/pickling issues with open file handles.
    """
    def __init__(self, h5_paths):
        # Normalize to a list
        if isinstance(h5_paths, (str, os.PathLike)):
            h5_paths = [h5_paths]
        self.h5_paths = list(h5_paths)

        # One handle per file (opened lazily in __getitem__)
        self._h5s = [None] * len(self.h5_paths)

        # Gather per-file lengths (assumes consistent keys across files)
        self.file_lengths = []
        for p in self.h5_paths:
            with h5py.File(p, 'r') as f:
                # Using 'img1' to define length, as in your original dataset
                self.file_lengths.append(f['img1'].shape[0])

        # Build cumulative boundaries for fast index -> (file_idx, local_idx)
        # Example: file_lengths=[5,7,3] -> cum=[0,5,12,15]
        self.cum_lengths = np.zeros(len(self.file_lengths) + 1, dtype=np.int64)
        self.cum_lengths[1:] = np.cumsum(self.file_lengths)
        self.length = int(self.cum_lengths[-1])

    def __len__(self):
        return self.length

    def _map_index(self, idx):
        # Support negative indices
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f'Index {idx} out of range for dataset of len {self.length}')

        # Find the file that contains this idx
        # bisect_right so that idx==cum_lengths[k] maps to file k
        file_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        local_idx = idx - int(self.cum_lengths[file_idx])
        return file_idx, local_idx

    def _ensure_open(self, file_idx):
        if self._h5s[file_idx] is None:
            self._h5s[file_idx] = h5py.File(self.h5_paths[file_idx], 'r')
        return self._h5s[file_idx]

    def __getitem__(self, idx):
        file_idx, local_idx = self._map_index(idx)
        h5f = self._ensure_open(file_idx)

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
