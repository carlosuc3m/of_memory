# prepare_h5.py
import os
import glob
import random
import cv2
import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

# -----------------------------------------------------------------------------
# 1) Replace this with however you load your encoder.
#    It must be a nn.Module that maps a batch of images to a batch of vectors:
#        input:  [B x 3 x H x W]  (float tensor, range [0,1])
#        output: [B x D]           (float tensor)
# -----------------------------------------------------------------------------
def load_encoder():
    # Example stub — replace with your actual model
    model = torch.hub.load('pytorch/vision:v0.15.2', 'resnet18', pretrained=True)
    # chop off final classifier
    model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
    model.output_dim = 512
    return model

# -----------------------------------------------------------------------------
# 2) Settings: adjust paths, augmentation count, and target size here
# -----------------------------------------------------------------------------
VIDEO_DIR             = 'videos/'        # where your .mp4 files live
OUT_H5                = 'data_pairs.h5'  # output HDF5
AUGS_PER_PAIR         = 3                # how many random augs per consecutive pair
TARGET_SIZE           = (224, 224)       # spatial size for crop/resize
DEVICE                = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 3) Paired augmentation: same transform for both images in a pair
# -----------------------------------------------------------------------------
def paired_augment(img1: Image.Image, img2: Image.Image):
    # random crop
    i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=TARGET_SIZE)
    img1 = F.crop(img1, i, j, h, w)
    img2 = F.crop(img2, i, j, h, w)
    # random horizontal flip
    if random.random() < 0.5:
        img1 = F.hflip(img1)
        img2 = F.hflip(img2)
    # random color jitter
    cj = transforms.ColorJitter(0.2,0.2,0.2,0.1)
    fn = cj.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)
    img1 = fn(img1)
    img2 = fn(img2)
    return img1, img2

# -----------------------------------------------------------------------------
# 4) Main extraction + encoding + storage
# -----------------------------------------------------------------------------
def main():
    # load encoder
    encoder = load_encoder().to(DEVICE).eval()
    to_tensor = transforms.ToTensor()

    # gather all mp4s
    video_paths = glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))
    if not video_paths:
        raise RuntimeError(f"No .mp4 files found in {VIDEO_DIR!r}")

    # open HDF5 with resizable datasets
    with h5py.File(OUT_H5, 'w') as h5:
        img1_ds = h5.create_dataset('img1',
                                    shape=(0, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                                    maxshape=(None, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                                    dtype='uint8')
        img2_ds = h5.create_dataset('img2',
                                    shape=(0, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                                    maxshape=(None, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                                    dtype='uint8')
        enc1_ds = h5.create_dataset('enc1',
                                    shape=(0, encoder.output_dim),
                                    maxshape=(None, encoder.output_dim),
                                    dtype='float32')
        enc2_ds = h5.create_dataset('enc2',
                                    shape=(0, encoder.output_dim),
                                    maxshape=(None, encoder.output_dim),
                                    dtype='float32')

        idx = 0
        for vid_path in video_paths:
            cap = cv2.VideoCapture(vid_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR→RGB, to PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            cap.release()

            # process each consecutive pair
            for i in range(len(frames) - 1):
                imgA, imgB = frames[i], frames[i+1]
                for _ in range(AUGS_PER_PAIR):
                    a1, a2 = paired_augment(imgA, imgB)
                    t1 = to_tensor(a1).unsqueeze(0).to(DEVICE)
                    t2 = to_tensor(a2).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        f1 = encoder(t1).squeeze(0).cpu().numpy()
                        f2 = encoder(t2).squeeze(0).cpu().numpy()

                    # append
                    for ds, arr in ((img1_ds, np.array(a1, dtype='uint8')),
                                    (img2_ds, np.array(a2, dtype='uint8')),
                                    (enc1_ds, f1),
                                    (enc2_ds, f2)):
                        ds.resize((idx+1, *ds.shape[1:]))
                        ds[idx] = arr
                    idx += 1

            print(f"[{os.path.basename(vid_path)}] → total pairs so far: {idx}")

if __name__ == '__main__':
    main()
