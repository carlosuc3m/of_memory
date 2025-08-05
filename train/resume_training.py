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

def load_checkpoint(path, model, optimizer):
    """
    Loads model & optimizer & scheduler states, and returns:
      start_epoch, last_loss
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # next epoch to run
    last_loss = checkpoint['loss']
    return start_epoch, last_loss

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
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
    flow_filters = [32 // 2, 64 // 2, 128 // 2, 256 // 2]
    filters = 16

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

    #model = OFMNet(config)
    model = UNet(3,256).cpu()
    #model = ResNet(256)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00003,
    )
    _, best_val_loss = load_checkpoint("/home/carlos/git_amazon/of_memory/checkpoints/best_checkpoint.pt", model, optimizer)
    optimizer.param_groups[0]['lr'] = 0.00003
    h5_path = '/home/carlos/git_amazon/of_memory/dataset/data_pairs_1_toy.h5'
    dataset = EncodingDataset(h5_path)
    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len

    # reproducible split
    torch.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=B_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)

    best_model_wts = copy.deepcopy(model.state_dict())
     
    history = {'train_loss': [], 'val_loss': [], 'train_seg_loss': [], 'val_seg_loss': []}

    device=torch.device("cuda")
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    scaler = GradScaler("cuda")
    beta_loss = 0
    counter_iter = -1
    criterion = nn.MSELoss()
    num_epochs = 400
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
                    #outputs = model(x0, x1, encoding0)
                    pred = model(x1, encoding0)
                    # If model returns dict:
                    #pred = outputs.get('image', outputs)
                    total_loss = criterion(pred, target)
                    #l3, l4 = sam_loss(target, pred)
                scaler.scale(total_loss).backward()


                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
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
            save_checkpoint(model, optimizer, epoch, total_loss, path="/home/carlos/git_amazon/of_memory/checkpoints/checkpoint.pt")
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

                    #outputs = model(x0, x1, encoding0)
                    pred = model(x1, encoding0)
                    #pred = outputs.get('image', outputs)
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
              save_checkpoint(model, optimizer, epoch, total_loss, path="/home/carlos/git_amazon/of_memory/checkpoints/best_checkpoint.pt")

        # ——— Scheduler step ———

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