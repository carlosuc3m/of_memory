# eval.py

import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

# You’ll need to implement these in your PyTorch codebase:
#   - create_eval_dataloaders(configs) → Dict[str, DataLoader]
#   - create_metrics(configs)         → Dict[str, Metric]
# where each Metric has methods reset(), update(batch, outputs, step), compute().
from .data_lib import create_eval_dataloaders
from .metrics_lib import create_metrics

# Hard‐coded evaluation configurations (copied from eval/config/*.gin)
EVAL_CONFIGS = {
    'middlebury': {
        'max_examples': -1,
        'metrics': ['l1', 'l2', 'ssim', 'psnr'],
        'tfrecord': 'gs://xcloud-shared/fitsumreda/frame_interpolation/datasets/middlebury_other.tfrecord@3',
    },
    'ucf101': {
        'max_examples': -1,
        'metrics': ['l1', 'l2', 'ssim', 'psnr'],
        'tfrecord': 'gs://xcloud-shared/fitsumreda/frame_interpolation/datasets/UCF101_interp_test.tfrecord@2',
    },
    'vimeo_90K': {
        'max_examples': -1,
        'metrics': ['l1', 'l2', 'ssim', 'psnr'],
        'tfrecord': 'gs://xcloud-shared/fitsumreda/frame_interpolation/datasets/vimeo_interp_test.tfrecord@3',
    },
    'xiph_2K': {
        'max_examples': -1,
        'metrics': ['l1', 'l2', 'ssim', 'psnr'],
        'tfrecord': 'gs://xcloud-shared/fitsumreda/frame_interpolation/datasets/xiph_2K.tfrecord@2',
    },
    'xiph_4K': {
        'max_examples': -1,
        'metrics': ['l1', 'l2', 'ssim', 'psnr'],
        'tfrecord': 'gs://xcloud-shared/fitsumreda/frame_interpolation/datasets/xiph_4K.tfrecord@2',
    },
}

def eval_loop(
    model: torch.nn.Module,
    log_dir: str,
    device: torch.device,
    checkpoint_step: int
) -> None:
    """
    Evaluate the model on all configured datasets, writing images and scalars
    to TensorBoard.

    Args:
        model:           your PyTorch frame‐interpolation nn.Module.
        log_dir:         directory to write TensorBoard logs.
        device:          torch.device('cpu') or ('cuda').
        checkpoint_step: current training iteration / global step.
    """
    # Prepare
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    model.eval()

    # Instantiate PyTorch DataLoaders and Metrics
    dataloaders = create_eval_dataloaders(EVAL_CONFIGS)
    metrics_map = create_metrics(EVAL_CONFIGS)

    # For each dataset…
    for ds_name, loader in dataloaders.items():
        logging.info(f"Evaluating on {ds_name} (step {checkpoint_step})")

        # Reset all metrics before this dataset
        for m in metrics_map.values():
            m.reset()

        max_examples = EVAL_CONFIGS[ds_name]['max_examples']
        num_seen = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Stop if we've hit the max_examples limit
                if 0 <= max_examples <= num_seen:
                    break

                # Move batch to device
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # Forward pass
                outputs = model(batch['x0'], batch['x1'], batch['encoding0'])
                image = outputs.get('image', outputs).clamp(0.0, 1.0)

                # Update metrics
                for m in metrics_map.values():
                    # Metric.update(batch, outputs, step)
                    m.update(batch, outputs, checkpoint_step)

                # Possibly write image summaries for the first few batches
                if batch_idx < 10:
                    prefix = f"{ds_name}/eval_{batch_idx+1}"
                    combined = {**batch, 'image': image}
                    for name, img in combined.items():
                        if torch.is_tensor(img) and img.dim() == 4 and img.size(1) in (1, 3):
                            # writer.add_images expects shape [B,C,H,W]
                            writer.add_images(f"{prefix}/{name}", img, global_step=checkpoint_step)
                elif batch_idx == 10:
                    writer.flush()

                num_seen += batch['x0'].size(0)

        # After finishing the dataset, log each metric
        for name, m in metrics_map.items():
            val = m.compute()
            writer.add_scalar(f"{ds_name}/{name}", val, checkpoint_step)
            logging.info(f"{ds_name}/{name}: {val:.6f}")

    writer.flush()
