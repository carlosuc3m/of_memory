import glob
import os
import time
from collections import deque, OrderedDict, defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torchvision.io import VideoReader
import torch.nn.functional as F

from torchvision.transforms.functional import normalize

from sam2.sam2_image_predictor import SAM2ImagePredictor


RES = 1024
device = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, -1, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, -1, 1, 1)
torch.backends.cudnn.benchmark = True  # post-resize shapes are fixed

VIDEO_DIR = '/home/carlos/git_amazon/of_memory/videos/'

def seed_worker(_):
    """Make NumPy/Python RNG different per worker but reproducible."""
    import random, numpy as np
    wi = get_worker_info()
    base = wi.seed if wi else 0
    random.seed(base)
    np.random.seed(base % (2**32))

def load_encoder_large(device=torch.device("cuda")):
    # Example stub — replace with your actual model
    from sam2.build_sam import build_sam2

    sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpt_path="/home/carlos/Downloads/sam2.1_hiera_large.pt", device=device)
    return SAM2ImagePredictor(sam_model.eval())

def load_encoder(device=torch.device("cuda")):
    # Example stub — replace with your actual model
    from sam2.build_sam import build_sam2

    sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path="/home/carlos/Downloads/sam2.1_hiera_tiny.pt", device=device)
    return SAM2ImagePredictor(sam_model.eval())


class CollateCPU:
    """Return CPU uint8 batches bucketed by (H, W) so we can resize per-bucket on GPU later."""
    def __call__(self, batch):
        # flatten triplets from dataset
        imgs, metas = [], []
        for b in batch:
            for i, fr in enumerate(b["frames"]):
                imgs.append(fr)  # (3,H,W) uint8 CPU
                metas.append((b["meta"][0], b["meta"][1][i]))  # (path, frame_idx)

        # bucket indices by (H,W)
        buckets = defaultdict(list)              # (H,W) -> [idx,...]
        for i, t in enumerate(imgs):
            h, w = int(t.shape[-2]), int(t.shape[-1])
            buckets[(h, w)].append(i)

        # stack each bucket on CPU
        cpu_batches = []                         # list[Tensor (k,3,H,W) uint8 CPU]
        cpu_metas   = []                         # list[list[meta]] aligned with cpu_batches
        for _, idxs in buckets.items():
            cpu_batches.append(torch.stack([imgs[i] for i in idxs], 0))
            cpu_metas.append([metas[i] for i in idxs])

        return cpu_batches, cpu_metas



class _ReaderPool:
    def __init__(self, capacity=64, num_threads=4):
        self.capacity = capacity
        self.num_threads = num_threads
        self.cache = OrderedDict()  # path -> (VideoReader, meta)

    def get(self, path):
        if path in self.cache:
            vr, meta = self.cache[path]
            self.cache.move_to_end(path)
            return vr, meta
        vr = VideoReader(path, "video", num_threads=self.num_threads)
        md = vr.get_metadata()["video"]
        fps = md.get("fps")[0] or md.get("framerate")[0] or md.get("average_fps")[0] or 30.0
        dur = float(md["duration"][0])
        nframes = max(1, int(dur * float(fps) + 0.5))
        meta = {"fps": float(fps), "duration": dur, "nframes": nframes}
        self.cache[path] = (vr, meta)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return vr, meta

class LiveDataset(Dataset):
    def __init__(self, video_files, separation_options=(3,5,7,9,11,13),
                 size=100_000, n_cached=180, pool_capacity=64, num_threads=4, seed=0):
        self.video_files = [str(video_files)] if isinstance(video_files, (str, bytes, os.PathLike)) \
                           else list(map(str, video_files))
        self.size = int(size)
        self.n_cached = int(n_cached)
        self.rng = np.random.default_rng(seed)
        self.seps = np.array(separation_options, dtype=np.int32)
        self.max_sep = int(self.seps.max())
        # map each index to a video path (roughly uniform)
        per = int(np.ceil(self.size / len(self.video_files)))
        self.file_map = np.repeat(self.video_files, per)[:self.size]
        # epoch shuffle
        self.perm = self.rng.permutation(self.size)

        # caches (CPU)
        self._im_cache = deque()   # list of triplets: [(3,H,W) uint8, ...]
        self._meta_cache = deque() # (path, (f0,f1,f2))
        self._pool = _ReaderPool(capacity=pool_capacity, num_threads=num_threads)

    def set_epoch(self, epoch:int):
        self.rng = np.random.default_rng(int(epoch))
        self.perm = self.rng.permutation(self.size)

    def __len__(self): return self.size

    def _decode_triplet(self, path):
        vr, meta = self._pool.get(path)
        fps, T = meta["fps"], meta["nframes"]
        if T <= self.max_sep:
            start = 0
            sep = min(int(self.seps[self.rng.integers(0, len(self.seps))]), max(T-1,1))
        else:
            sep = int(self.seps[self.rng.integers(0, len(self.seps))])
            start = int(self.rng.integers(0, T - sep))
        f0, f2 = start, start + sep - 1

        # seek to time in seconds
        vr.seek(f0 / fps, keyframes_only=True)

        decoded = []
        needed = f2 - f0 + 1
        for i, fr in enumerate(vr):
            if i == 0:
                decoded.append(fr["data"])  # uint8 CHW CPU
            elif i == sep - 1:
                decoded.append(fr["data"])  # uint8 CHW CPU
                break

        # HWC->CHW uint8
        #triplet = [t.permute(2,0,1).contiguous() for t in decoded]
        return decoded, (f0, f2)

    def _renew_cache(self, start_idx):
        # Fill cache with next n_cached samples
        paths = self.file_map[self.perm[start_idx : start_idx + self.n_cached]]
        for p in paths:
            tri, idxs = self._decode_triplet(p)
            self._im_cache.append(tri)       # list of three (3,H,W) uint8 tensors
            self._meta_cache.append((p, idxs))

    def __getitem__(self, idx):
        if not self._im_cache:
            self._renew_cache(idx)
        im1, im2 = self._im_cache.popleft()
        meta = self._meta_cache.popleft()
        return {"frames": [im1, im2], "meta": meta}


def main():
    device = "cuda"
    sam = load_encoder(torch.device(device))  # loads encoder on GPU

    video_paths = glob.glob(os.path.join(VIDEO_DIR, '**', '*.mp4'), recursive=True)
    ds = LiveDataset(
        video_paths,
        separation_options=(3,5,7,9,11,13),
        size=100_000,
        n_cached=32,
        pool_capacity=32,      # if you used the ReaderPool variant
        num_threads=4,
        seed=42,
    )

    collate = CollateCPU()

    loader = DataLoader(
        ds,
        batch_size=6,                 # 8 triplets -> the collate sees 24 frames
        shuffle=False,                # dataset already shuffles internally; or keep True if you prefer
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,              # good for H2D overlap
        worker_init_fn=seed_worker,
        collate_fn=collate,           # <--- preprocess + encode happen here (main process)
    )

    # ---- example training loop ----
    tt = time.time()
    for epoch in range(10):
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)       # reshuffle inside dataset if implemented
        for counter, (cpu_batches, cpu_metas) in enumerate(loader):
            # move/normalize/resize per bucket (no padding), then concat
            xs = []
            metas = []
            for cpu_batch, metas_bucket in zip(cpu_batches, cpu_metas):
                x = cpu_batch.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                x = x.float().div_(255.0)
                x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
                x = normalize(x, MEAN, STD)
                #x = (x - MEAN) / STD
                xs.append(x)
                metas.extend(metas_bucket)

            x_all = torch.cat(xs, dim=0)  # (B*3, 3, RES, RES), channels_last
            # single encoder pass
            del xs, x, cpu_batches, cpu_metas, metas
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats, _ = sam.model.image_encoder.neck(sam.model.image_encoder.trunk(x_all))
            print(counter, time.time() - tt)
            del _
            B = int(x_all.shape[0] // 2)
            x_in, x_out = x_all.reshape(B, 2, *x_all.shape[1:]).unbind(1)
            del x_all
            (enc1_in, enc1_out), (enc2_in, enc2_out), (enc3_in, enc3_out) = [
                f.reshape(B, 2, *f.shape[1:]).unbind(1) for f in feats[:3]
                ]
            del feats
            tt = time.time()
            torch.cuda.empty_cache() 

if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()