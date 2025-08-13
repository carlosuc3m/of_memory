import time
import copy
from typing import Optional, Dict, Any
import random
import math
import glob


from torch.optim.lr_scheduler import LambdaLR

import torch
import os
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from .hiera import Hiera
from .vit_model import ViTModel
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.backbones.image_encoder import FpnNeck
from .live_dataset_2 import LiveDataset, load_encoder, CollateCPU, seed_worker


from torch.amp import autocast, GradScaler


B_SIZE = 4
DEVICE = torch.device("cuda")
VIDEO_DIR = '/home/carlos/git_amazon/of_memory/videos/'
SEED = 69
TRAIN_PERC = 0.8

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'scheduler_state_dict':  scheduler.state_dict(),
        'loss': loss,
    }, path)

def main():

    sam = load_encoder(torch.device(DEVICE))  # loads encoder on GPU

    video_paths = glob.glob(os.path.join(VIDEO_DIR, '**', '*.mp4'), recursive=True)
    perm = torch.randperm(len(video_paths), generator=torch.Generator().manual_seed(SEED)).tolist()
    cut = int(len(video_paths) * TRAIN_PERC)
    train_videos, val_videos = [video_paths[i] for i in perm[:cut]], [video_paths[i] for i in perm[cut:]]
    train_ds = LiveDataset(
        train_videos,
        separation_options=(3,5,7,9,11,13),
        size=100_000,
        n_cached=32,
        pool_capacity=32,      # if you used the ReaderPool variant
        num_threads=4,
        seed=42,
    )
    val_ds = LiveDataset(
        val_videos,
        separation_options=(3,5,7,9,11,13),
        size=10_000,
        n_cached=32,
        pool_capacity=32,      # if you used the ReaderPool variant
        num_threads=4,
        seed=42,
    )

    collate = CollateCPU()

    train_loader = DataLoader(
        train_ds,
        batch_size=6,                 # 8 triplets -> the collate sees 24 frames
        shuffle=False,                # dataset already shuffles internally; or keep True if you prefer
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,              # good for H2D overlap
        worker_init_fn=seed_worker,
        collate_fn=collate,           # <--- preprocess + encode happen here (main process)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=6,                 # 8 triplets -> the collate sees 24 frames
        shuffle=False,                # dataset already shuffles internally; or keep True if you prefer
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,              # good for H2D overlap
        worker_init_fn=seed_worker,
        collate_fn=collate,           # <--- preprocess + encode happen here (main process)
    )

    hiera = Hiera(embed_dim=32, num_heads=1, stages=[1, 2, 3, 2], global_att_blocks=[4, 5], window_pos_embed_bkg_spatial_size=[7, 7]).cpu()
    pos_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=1000)
    neck = FpnNeck(position_encoding=pos_encoding, d_model=256, backbone_channel_list=[256, 128, 64, 32],
        fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
    model = ViTModel(neck=neck, trunk=hiera)

    base_lr    = 5e-4       # scratch.base_lr
    vision_lr  = 3e-4       # scratch.vision_lr
    wd         = 0.1        # default weight decay
    wd_exceptions = ("bias", "LayerNorm.weight")
    num_epochs = 40
    steps_per_epoch = len(train_loader)  # your DataLoader
    total_steps = num_epochs * steps_per_epoch

    # 3) Split params into 2 LR groups + handle weight‐decay exceptions
    param_groups = [
        {"params": [], "lr": vision_lr, "weight_decay": wd},  # vision trunk
        {"params": [], "lr": vision_lr, "weight_decay": 0.0}, # vision biases & LN
        {"params": [], "lr": base_lr,  "weight_decay": wd},  # scratch heads
        {"params": [], "lr": base_lr,  "weight_decay": 0.0},  # scratch biases & LN
    ]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_vision = name.startswith("image_encoder.trunk")
        is_exception = any(key in name for key in wd_exceptions)
        if is_vision and not is_exception:
            param_groups[0]["params"].append(param)
        elif is_vision and is_exception:
            param_groups[1]["params"].append(param)
        elif not is_vision and not is_exception:
            param_groups[2]["params"].append(param)
        else:
            param_groups[3]["params"].append(param)

    # 4) Optimizer + AMP scaler
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))

    # 5) Cosine‐decay schedule for each LR group to (lr/10)
    def make_cosine_lambda(start, end):
        def lr_lambda(step):
            # cosine ramp from 0→π over total_steps
            cos_out = 0.5 * (1 + math.cos(math.pi * step / total_steps))
            return end/start + (1 - end/start) * cos_out
        return lr_lambda

    # note: order of lambdas must match param_groups
    lambdas = [
        make_cosine_lambda(vision_lr, vision_lr/100),
        make_cosine_lambda(vision_lr, vision_lr/100),
        make_cosine_lambda(base_lr,   base_lr/100),
        make_cosine_lambda(base_lr,   base_lr/100),
    ]
    scheduler = LambdaLR(optimizer, lr_lambda=lambdas)







    train_model(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                criterion=nn.MSELoss(), device=torch.device("cuda"), num_epochs=num_epochs, scheduler=scheduler)



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Train and validate a PyTorch model.

    Args:
        model:        the neural network to train.
        train_loader: DataLoader for the training set.
        val_loader:   DataLoader for validation (or None to skip).
        optimizer:    torch.optim optimizer.
        criterion:    loss function (nn.Module).
        device:       torch.device('cuda') or ('cpu').
        num_epochs:   total epochs to run.
        scheduler:    optional LR scheduler (step each epoch).
        grad_clip:    optional max-norm for gradient clipping.

    Returns:
        A dict containing:
          - 'best_model': state_dict of best‐validation model,
          - 'history': { 'train_loss': [...], 'val_loss': [...] },
          - 'best_epoch': int index of best validation.
    """
    best_val_loss = float('inf')

    model.to(device)
    scaler = GradScaler("cuda")
    
    log_file = "/home/carlos/git_amazon/log.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    f = open(log_file, "w")

    # The LR schedule initialization resets the initial LR of the optimizer.
    beta_loss = 0
    counter_iter = -1
    for epoch in range(1, num_epochs + 1):
        # ——— Training phase ———
        model.train()
        running_loss = 0.0
        total_samples = 0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="batch") as tepoch:
            for batch in tepoch:
                counter_iter += 1
                # Unpack your batch: adjust names to your dataset
                x0, x1, encoding0, target = batch
                x0 = x0.to(device)
                x1 = x1.to(device)
                encoding0 = encoding0.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(x1, encoding0)
                    # If model returns dict:
                    pred = outputs["vision_features"]
                    total_loss = criterion(pred, target)
                    #l3, l4 = sam_loss(target, pred)


                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
                scaler.step(optimizer)
                scaler.update()   
                scheduler.step()
                
                if counter_iter % 20 == 0 and False:
                  print(total_loss.item())
                  print(beta_loss / 20)
                  beta_loss = 0
                beta_loss += total_loss.item()
                running_loss += (total_loss.item()) * x0.size(0)
                total_samples += x0.size(0)
                #running_seg_loss += l3.item() * x0.size(0)
                tepoch.set_postfix(train_loss=f"{(running_loss / (total_samples)):.6f}",
                                   #train_seg_loss=f"{(running_seg_loss / ((tepoch.n + 1)*x0.size(0))):.6f}"
                                   )
            epoch_train_loss = running_loss / len(train_loader.dataset)
            #epoch_train_seg_loss = running_seg_loss / len(train_loader.dataset)
            tepoch.set_postfix(train_loss=epoch_train_loss)

        #if prev_loss < epoch_train_loss:
            #optimizer.param_groups[0]['lr'] /= 2
        #prev_loss = epoch_train_loss
        print(f"epoch {epoch} -- train_loss {(running_loss / (total_samples)):.6f}", file=f)
        # ——— Validation phase ———
        if val_loader is not None and epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, total_loss, path="/home/carlos/git_amazon/of_memory/checkpoints/checkpoint.pt")
            model.eval()
            val_running = 0.0
            numbers = 0
            with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ", unit="batch") as vepoch:
                for batch in vepoch:
                    x0, x1, encoding0, target = batch
                    x0 = x0.to(device)
                    x1 = x1.to(device)
                    encoding0 = encoding0.to(device)
                    target = target.to(device)

                    outputs = model(x0, x1, encoding0)
                    #pred = model(x1, encoding0)
                    pred = outputs.get('image', outputs)
                    loss = criterion(pred, target)
                    #l3, l4 = sam_loss(target, pred)

                    val_running += (loss.item()) * x0.size(0)
                    numbers += x0.size(0)
                    vepoch.set_postfix(val_loss=val_running / numbers)

                epoch_val_loss = val_running / len(val_loader.dataset)
                vepoch.set_postfix(val_loss=epoch_val_loss)

            # Checkpoint best model
            if epoch_val_loss < best_val_loss:
                save_checkpoint(model, optimizer, scheduler, epoch, total_loss, path="/home/carlos/git_amazon/of_memory/checkpoints/best_checkpoint.pt")
                best_val_loss = epoch_val_loss
            print(f"epoch {epoch} -- val_loss {(val_running / len(val_loader.dataset)):.6f}", file=f)
        # ——— Scheduler step ———

    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_val_loss})")

    # Load best weights

    return {
        'best_model': model.state_dict(),
        'best_epoch': best_val_loss
    }


if __name__ == '__main__':
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  main()