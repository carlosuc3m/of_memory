import h5py
import random
import torch
from torch.utils.data import Dataset, DataLoader

class EncodingDataset(Dataset):
    def __init__(self, h5_path):
        """
        h5_path: path to your .h5 file
        transform: optional callable on both input & target tensors
        """
        self.h5_path = h5_path
        self._h5 = None
        # figure out the length up‐front
        with h5py.File(h5_path, 'r') as f:
            self.length = f['img1'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # open HDF5 for this worker if not already
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

        # load all four arrays at this idx
        arrs = {
            'img1': self._h5['img1'][idx],   # uint8, (3,H,W)
            'img2': self._h5['img2'][idx],   # uint8, (3,H,W)
            'enc1': self._h5['enc1'][idx],   # float32, (256,64,64)
            'enc2': self._h5['enc2'][idx],   # float32, (256,64,64)
        }

        if random.uniform(0, 1) > 0.5:
            input_1 = torch.from_numpy(arrs['img1']).float()
            enc_1 = torch.from_numpy(arrs['enc1']).float()
            input_2 = torch.from_numpy(arrs['img2']).float()
            enc_2 = torch.from_numpy(arrs['enc2']).float()
        else:
            input_2 = torch.from_numpy(arrs['img1']).float()
            enc_2 = torch.from_numpy(arrs['enc1']).float()
            input_1 = torch.from_numpy(arrs['img2']).float()
            enc_1 = torch.from_numpy(arrs['enc2']).float()


        return input_1, input_2, enc_1, enc_2
