import time
import copy
from typing import Optional, Dict, Any
import random


import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from of_memory.model import OFMNet
from of_memory.ofm_transforms import OFMTransforms
from of_memory.encoding_dataset import EncodingDataset

from sam2_loss import SAM2Loss

from config.config import Options



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
    pyramid_levels = 5
    fusion_pyramid_levels = 4
    specialized_levels = 3
    sub_levels = 4
    flow_convs = [3, 3, 3, 3]
    flow_filters = [16, 32, 64, 128]
    filters = 16

    learning_rate = 0.0001
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


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
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
    h5_path = '/home/carlos/git_amazon/of_memory/dataset/data_pairs_1.h5'
    transforms = OFMTransforms(1024, max_hole_area=0.0, max_sprinkle_area=0.0)
    dataset = EncodingDataset(h5_path)
    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len

    # reproducible split
    torch.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2)

    train_model(model=model, transforms=transforms, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, 
                criterion=nn.L1Loss(), device=torch.device("cuda"), num_epochs=400, scheduler=scheduler)



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
    sam_loss = SAM2Loss(torch.device("cuda"))
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        # ——— Training phase ———
        model.train()
        running_loss = 0.0
        running_seg_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="batch") as tepoch:
            for batch in tepoch:
                # Unpack your batch: adjust names to your dataset
                x0, x1, encoding0, target = batch  
                x0 = x0.to(device)
                x1 = x1.to(device)
                encoding0 = encoding0.to(device)
                target = target.to(device)

                hflip = random.choice([True, False])
                vflip = random.choice([True, False])

                # channels‐first: inp/tgt shape is (B, C, H, W)
                if hflip:
                    x0 = x0.flip(dims=[3])
                    encoding0 = encoding0.flip(dims=[3])
                    x1 = x1.flip(dims=[3])
                    target = target.flip(dims=[3])
                if vflip:
                    x0 = x0.flip(dims=[2])
                    encoding0 = encoding0.flip(dims=[2])
                    x1 = x1.flip(dims=[2])
                    target = target.flip(dims=[2])

                # 4) optional transforms (e.g. normalization, to‐float, etc.)
                if transforms is not None:
                    x0 = transforms(x0)
                    x1 = transforms(x1)

                optimizer.zero_grad()
                outputs = model(x0, x1, encoding0)
                # If model returns dict:
                pred = outputs.get('image', outputs)
                loss = criterion(pred, target)
                l3, l4, l5 = sam_loss(target, pred)
                total_loss = loss + l3 + l4 + l5
                total_loss.backward()


                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                running_loss += (total_loss.item()) * x0.size(0)
                running_seg_loss += l5.item() * x0.size(0)
                tepoch.set_postfix(train_loss=running_loss / ((tepoch.n + 1)*x0.size(0)),
                                   train_seg_loss=running_seg_loss / ((tepoch.n + 1)*x0.size(0))
                                   )

            epoch_train_loss = running_loss / len(train_loader.dataset)
            vepoch.set_postfix(train_loss=epoch_train_loss)
            epoch_train_seg_loss = running_seg_loss / len(train_loader.dataset)
            vepoch.set_postfix(train_seg_loss=epoch_train_seg_loss)
        history['train_loss'].append(epoch_train_loss)
        history['train_seg_loss'].append(epoch_train_seg_loss)

        # ——— Validation phase ———
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_seg_running = 0.0
            with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ", unit="batch") as vepoch:
                for batch in vepoch:
                    x0, x1, encoding0, target = batch
                    x0 = x0.to(device)
                    x1 = x1.to(device)
                    encoding0 = encoding0.to(device)
                    target = target.to(device)

                    x0 = transforms(x0)
                    x1 = transforms(x1)
                    outputs = model(x0, x1, encoding0)
                    pred = outputs.get('image', outputs)
                    loss = criterion(pred, target)
                    l3, l4, l5 = sam_loss(target, pred)

                    val_running += (loss.item() + l3.item() + l4.item() + l5.item()) * x0.size(0)
                    val_seg_running += l5.item() * x0.size(0)
                    vepoch.set_postfix(val_loss=val_running / ((vepoch.n + 1)*x0.size(0)),
                                       val_seg_loss=val_seg_running / ((vepoch.n + 1)*x0.size(0)))

                epoch_val_loss = val_running / len(val_loader.dataset)
                epoch_val_seg_loss = val_seg_running / len(val_loader.dataset)
                vepoch.set_postfix(val_loss=epoch_val_loss)
                vepoch.set_postfix(val_seg_loss=epoch_val_seg_loss)
            history['val_loss'].append(epoch_val_loss)
            history['val_seg_loss'].append(epoch_val_seg_loss)

            # Checkpoint best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # ——— Scheduler step ———
        if scheduler is not None:
            scheduler.step()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

    # Load best weights
    model.load_state_dict(best_model_wts)

    return {
        'best_model': model.state_dict(),
        'history': history,
        'best_epoch': best_epoch
    }


if __name__ == '__main__':
  main()