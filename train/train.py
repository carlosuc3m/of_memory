import time
import copy
from typing import Optional, Dict, Any
import random



import pytorch_warmup as warmup

import torch
import os
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from of_memory.model_carlos import OFMNet
from of_memory.unet import UNet
from of_memory.ofm_transforms import OFMTransforms
from of_memory.encoding_dataset import EncodingDataset

from sam2_loss import SAM2Loss

from config.config import Options

from torch.amp import autocast, GradScaler


B_SIZE = 4

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'scheduler_state_dict':  scheduler.state_dict(),
        'loss': loss,
    }, path)

def main():

    ## OG PARAMS
    pyramid_levels = 7
    fusion_pyramid_levels = 5
    specialized_levels = 3
    sub_levels = 4
    flow_convs = [3, 3, 3, 3]
    flow_filters = [32, 64, 128, 256]
    filters = 64

    ## LIGHT PARAMS
    pyramid_levels = 8
    fusion_pyramid_levels = 5
    specialized_levels = 3
    sub_levels = 4
    flow_convs = [3, 3, 3, 3]
    flow_filters = [32, 64, 128, 256]
    filters = 64

    ## OG learning_rate = 0.0001
    learning_rate = 0.0003
    learning_rate_decay_steps = 750000
    learning_rate_decay_rate = 0.464158
    learning_rate_staircase = True
    num_steps = 3000000
    config = Options(pyramid_levels=pyramid_levels,
                            fusion_pyramid_levels=fusion_pyramid_levels,
                            specialized_levels=specialized_levels,
                            flow_convs=flow_convs,
                            flow_filters=flow_filters,
                            sub_levels=sub_levels,
                            filters=filters,
                            use_aux_outputs=True)

    model = OFMNet(config)
    model = UNet(3,256)
    #model = ResNet(256)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    if learning_rate_staircase:
        # “Staircase” decay: multiply by decay_rate every decay_steps steps
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=learning_rate_decay_steps,
            gamma=learning_rate_decay_rate
        )
    else:
        # Smooth exponential decay: continuous, exactly
        # lr = lr0 * decay_rate ** (step / decay_steps)
        gamma = learning_rate_decay_rate ** (1.0 / learning_rate_decay_steps)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    h5_path = '/content/data_pairs_1_toy.h5'
    transforms = OFMTransforms(1024, max_hole_area=0.0, max_sprinkle_area=0.0)
    dataset = EncodingDataset(h5_path)
    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len

    # reproducible split
    torch.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=B_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)

    train_model(model=model, transforms=transforms, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                criterion=nn.MSELoss(), device=torch.device("cuda"), num_epochs=400, scheduler=scheduler)



def train_model(
    model: nn.Module,
    transforms: nn.Module,
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
    #sam_loss = SAM2Loss(torch.device("cuda"))
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_seg_loss': [], 'val_seg_loss': []}

    model.to(device)
    scaler = GradScaler("cuda")
    accum_steps = 32
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(128 // B_SIZE) * 400)
    # The LR schedule initialization resets the initial LR of the optimizer.
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    beta_loss = 0
    counter_iter = -1
    """
    for layer in model.modules():
      if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
          nn.utils.spectral_norm(layer)
    """
    for epoch in range(1, num_epochs + 1):
        # ——— Training phase ———
        model.train()
        running_loss = 0.0
        total_samples = 0
        running_seg_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="batch") as tepoch:
            for batch in tepoch:
                counter_iter += 1
                # Unpack your batch: adjust names to your dataset
                x0, x1, encoding0, target = batch
                x0 = x0.to(device)
                x1 = x1.to(device)
                encoding0 = encoding0.to(device)
                target = target.to(device)

                hflip = random.choice([True, False])
                vflip = random.choice([True, False])

                # channels‐first: inp/tgt shape is (B, C, H, W)
                if hflip and False:
                    x0 = x0.flip(dims=[3])
                    encoding0 = encoding0.flip(dims=[3])
                    x1 = x1.flip(dims=[3])
                    target = target.flip(dims=[3])
                if vflip and False:
                    x0 = x0.flip(dims=[2])
                    encoding0 = encoding0.flip(dims=[2])
                    x1 = x1.flip(dims=[2])
                    target = target.flip(dims=[2])

                optimizer.zero_grad()
                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(x0, x1, encoding0)
                    # If model returns dict:
                    pred = outputs.get('image', outputs)
                    total_loss = criterion(pred, target)
                    #l3, l4 = sam_loss(target, pred)
                scaler.scale(total_loss).backward()


                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                with warmup_scheduler.dampening():
                  lr_scheduler.step()
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

        history['train_loss'].append(epoch_train_loss)
        #history['train_seg_loss'].append(epoch_train_seg_loss)

        # ——— Validation phase ———
        if val_loader is not None and epoch % 10 == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, total_loss, path="/content/drive/MyDrive/ofm/checkpoint.pt")
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
                    pred = outputs.get('image', outputs)
                    loss = criterion(pred, target)
                    #l3, l4 = sam_loss(target, pred)

                    val_running += (loss.item()) * x0.size(0)
                    numbers += x0.size(0)
                    vepoch.set_postfix(val_loss=val_running / numbers)

                epoch_val_loss = val_running / len(val_loader.dataset)
                vepoch.set_postfix(val_loss=epoch_val_loss)
            history['val_loss'].append(epoch_val_loss)

            # Checkpoint best model
            if epoch_val_loss < best_val_loss:
              save_checkpoint(model, optimizer, lr_scheduler, epoch, total_loss, path="/content/drive/MyDrive/ofm/best_checkpoint.pt")

        # ——— Scheduler step ———

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_val_loss})")

    # Load best weights
    model.load_state_dict(best_model_wts)

    return {
        'best_model': model.state_dict(),
        'history': history,
        'best_epoch': best_val_loss
    }


if __name__ == '__main__':
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
  main()