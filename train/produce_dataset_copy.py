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

import sys

from sam2.sam2_image_predictor import SAM2ImagePredictor


MAX_SIZE = 5 * 1024 * 1024 * 1024

# -----------------------------------------------------------------------------
# 1) Replace this with however you load your encoder.
#    It must be a nn.Module that maps a batch of images to a batch of vectors:
#        input:  [B x 3 x H x W]  (float tensor, range [0,1])
#        output: [B x D]           (float tensor)
# -----------------------------------------------------------------------------
def load_encoder():
    # Example stub — replace with your actual model
    from sam2.build_sam import build_sam2

    sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path="/home/carlos/Downloads/sam2.1_hiera_large.pt")
    return SAM2ImagePredictor(sam_model.to(DEVICE).eval())

# -----------------------------------------------------------------------------
# 2) Settings: adjust paths, augmentation count, and target size here
# -----------------------------------------------------------------------------
VIDEO_DIR             = '/home/carlos/git_amazon/of_memory/videos/'        # where your .mp4 files live
OUT_H5                = '/home/carlos/git_amazon/of_memory/dataset/data_pairs_1_toy.h5'  # output HDF5
AUGS_PER_PAIR         = 3                # how many random augs per consecutive pair
TARGET_SIZE           = (1024, 1024)       # spatial size for crop/resize
DEVICE                = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# 3) Paired augmentation: same transform for both images in a pair
# -----------------------------------------------------------------------------
def paired_augment(img1: Image.Image, img2: Image.Image):
    # random crop
    w_scale = random.uniform(0.75, 1.0)
    h_scale = random.uniform(0.75, 1.0)
    crop_w = int(w_scale * img1.width)
    crop_h = int(h_scale * img1.height)
    i, j, h, w = transforms.RandomCrop.get_params(
        img1, output_size=(crop_h, crop_w)
    )
    img1 = F.crop(img1, i, j, h, w)
    img2 = F.crop(img2, i, j, h, w)
    # random horizontal flip
    if random.random() < 0.5:
        img1 = F.hflip(img1)
        img2 = F.hflip(img2)
    # random color jitter
    cj = transforms.ColorJitter(0.2,0.2,0.2,0.1)

    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = cj.get_params(
        cj.brightness, cj.contrast, cj.saturation, cj.hue
    )

    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            img1 = F.adjust_brightness(img1, brightness_factor)
            img2 = F.adjust_brightness(img2, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            img1 = F.adjust_contrast(img1, contrast_factor)
            img2 = F.adjust_contrast(img2, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            img1 = F.adjust_saturation(img1, saturation_factor)
            img2 = F.adjust_saturation(img2, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            img1 = F.adjust_hue(img1, hue_factor)
            img2 = F.adjust_hue(img2, hue_factor)

    return img1, img2

# -----------------------------------------------------------------------------
# 4) Main extraction + encoding + storage
# -----------------------------------------------------------------------------
def main():
    # load encoder
    encoder = load_encoder()
    transforms_rgb = torch.jit.script(
            transforms.Resize((1024, 1024))
        )

    # gather all mp4s
    video_paths = glob.glob(os.path.join(VIDEO_DIR, '**', '*.mp4'), recursive=True)
    if not video_paths:
        raise RuntimeError(f"No .mp4 files found in {VIDEO_DIR!r}")

    # open HDF5 with resizable datasets
    unit_size = 3 * TARGET_SIZE[0] * TARGET_SIZE[1] * 2 * 4 + 256 * 64 * 64 * 2 * 4
    max_images = MAX_SIZE // unit_size
    with h5py.File(OUT_H5, 'w') as h5:
        img1_ds = h5.create_dataset('img1',
                                    shape=(0, 3, TARGET_SIZE[0], TARGET_SIZE[1]),
                                    maxshape=(None, 3, TARGET_SIZE[0], TARGET_SIZE[1]),
                                    dtype='float32')
        img2_ds = h5.create_dataset('img2',
                                    shape=(0, 3, TARGET_SIZE[0], TARGET_SIZE[1]),
                                    maxshape=(None, 3, TARGET_SIZE[0], TARGET_SIZE[1]),
                                    dtype='float32')
        enc1_ds = h5.create_dataset('enc1',
                                    shape=(0, 256, 64, 64),
                                    maxshape=(None, 256, 64, 64),
                                    dtype='float32')
        enc2_ds = h5.create_dataset('enc2',
                                    shape=(0, 256, 64, 64),
                                    maxshape=(None, 256, 64, 64),
                                    dtype='float32')

        idx = 0
        iter_per = 200
        for vid_path in video_paths[len(video_paths) // 2 :]:
            print(vid_path)
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
            MAX_FRAME_DIS = 10
            L = len(frames) - 1
            selected_inds = random.sample(range(1, L - 1), L // 50)
            for i in selected_inds:
                if idx > iter_per:
                    h5.flush()
                    print(idx)
                    iter_per += 200
                imgA = frames[i]
                ind_b = random.randint(max(i- MAX_FRAME_DIS, 0), min(i + MAX_FRAME_DIS, L))
                imgB = frames[ind_b]
                for _ in range(AUGS_PER_PAIR):
                    a1, a2 = paired_augment(imgA, imgB)
                    #t1 = to_tensor(a1).unsqueeze(0).to(DEVICE)
                    #t2 = to_tensor(a2).unsqueeze(0).to(DEVICE)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tensor1 = encoder._transforms(a1)[None, ...].cpu().numpy()
                        tensor2 = encoder._transforms(a2)[None, ...].cpu().numpy()
                        encoder.set_image(a1)
                        image_embed = encoder._features["image_embed"]
                        f1 = image_embed.cpu().numpy()
                        encoder.set_image(a2)
                        image_embed = encoder._features["image_embed"]
                        f2 = image_embed.cpu().numpy()

                    # append
                    for ds, arr in ((img1_ds, tensor1),
                                    (img2_ds, tensor2),
                                    (enc1_ds, f1),
                                    (enc2_ds, f2),):
                        ds.resize((idx+1, *ds.shape[1:]))
                        ds[idx] = arr
                    idx += 1
                    if idx > max_images:
                        break
                if idx > max_images:
                    break
            if idx > max_images:
                break

            print(f"[{os.path.basename(vid_path)}] → total pairs so far: {idx}")

if __name__ == '__main__':
    main()
