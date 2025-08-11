import random
from collections import deque, OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from torchvision.io import VideoReader

RES = 1024
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])


class CollateSAM:
    def __init__(self, sam_predictor, device="cuda"):
        self.sam = sam_predictor
        self.device = device

    def __call__(self, batch):
        # Flatten frames
        imgs, metas = [], []
        for b in batch:
            for i, fr in enumerate(b["frames"]):
                imgs.append(fr)  # (3,H,W) uint8 CPU tensor
                # meta is (path, (f0,f1,f2)); we store the specific frame idx
                metas.append((b["meta"][0], b["meta"][1][i]))

        # Stack -> (N,3,H,W), move to GPU, channels_last
        x = torch.stack(imgs, 0).to(self.device, non_blocking=True) \
                                .contiguous(memory_format=torch.channels_last)
        x = x.float().div_(255.0)
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        mean = MEAN.to(self.device).view(1, -1, 1, 1)
        std  = STD.to(self.device).view(1, -1, 1, 1)
        x = (x - mean) / std

        # Fast convs on fixed shapes
        torch.backends.cudnn.benchmark = True

        with torch.inference_mode(), torch.cuda.amp.autocast(True):
            trunk = self.sam.model.encoder.trunk(x)
            feats, pos = self.sam.model.encoder.neck(trunk)

        return x, feats, pos, metas



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
        fps = md.get("fps") or md.get("framerate") or md.get("average_fps") or 30.0
        if hasattr(fps, "numerator"): fps = float(fps.numerator) / float(fps.denominator)
        dur = float(md["duration"])
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
        mid = sep // 2
        f0, f1, f2 = start, start + mid, start + sep - 1

        # seek to time in seconds
        vr.seek(f0 / fps, keyframe=True)

        decoded = []
        needed = f2 - f0 + 1
        for i, fr in enumerate(vr):
            decoded.append(fr["data"])  # uint8 HxWxC CPU
            if len(decoded) >= needed:
                break
        if len(decoded) < needed:
            # fallback: restart
            vr.seek(0.0, keyframe=True)
            decoded = []
            for fr in vr:
                decoded.append(fr["data"])
                if len(decoded) >= needed:
                    break
            if len(decoded) < needed:
                raise RuntimeError(f"Failed to decode {path}")

        take = [decoded[0], decoded[mid], decoded[sep-1]]
        # HWC->CHW uint8
        triplet = [t.permute(2,0,1).contiguous() for t in take]
        return triplet, (f0, f1, f2)

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
        im1, im2, im3 = self._im_cache.popleft()
        meta = self._meta_cache.popleft()
        return {"frames": [im1, im2, im3], "meta": meta}
