import h5py
import random
import torch
from torch.utils.data import Dataset, DataLoader

class EncodingDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        """
        h5_path: path to your .h5 file
        transform: optional callable on both input & target tensors
        """
        self.h5_path = h5_path
        self.transform = transform
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
            input_1 = torch.from_numpy(arrs['img1'].copy())
            enc_1 = torch.from_numpy(arrs['enc1'].copy())
            input_2 = torch.from_numpy(arrs['img2'].copy())
            enc_2 = torch.from_numpy(arrs['enc2'].copy())
        else:
            input_2 = torch.from_numpy(arrs['img1'].copy())
            enc_2 = torch.from_numpy(arrs['enc1'].copy())
            input_1 = torch.from_numpy(arrs['img2'].copy())
            enc_1 = torch.from_numpy(arrs['enc2'].copy())

        hflip = random.choice([True, False])
        vflip = random.choice([True, False])
        if not (hflip or vflip):
            # force at least one
            if random.random() < 0.5:
                hflip = True
            else:
                vflip = True

        # channels‐first: inp/tgt shape is (C, H, W)
        if hflip:
            input_1 = input_1.flip(dims=[2])
            enc_1 = enc_1.flip(dims=[2])
            input_2 = input_2.flip(dims=[2])
            enc_2 = enc_2.flip(dims=[2])
        if vflip:
            input_1 = input_1.flip(dims=[1])
            enc_1 = enc_1.flip(dims=[1])
            input_2 = input_2.flip(dims=[1])
            enc_2 = enc_2.flip(dims=[1])

        # 4) optional transforms (e.g. normalization, to‐float, etc.)
        if self.transform is not None:
            input_1 = self.transform(input_1)
            input_2 = self.transform(input_2)
            enc_1 = self.transform(enc_1)
            enc_2 = self.transform(enc_2)

        return input_1, input_2, enc_1, enc_2
