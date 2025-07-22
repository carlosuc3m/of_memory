# losses.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Callable, Optional, Any, Tuple

# If you have a PyTorch port of your VGG‑based perceptual/style losses,
# import them here. Otherwise you can implement them using torchvision.
from .vgg19_loss import vgg_loss as _vgg_loss, style_loss as _style_loss

try:
    # Kornia provides a convenient SSIM implementation
    from kornia.losses import SSIM
    _ssim_fn = SSIM(window_size=11, reduction='mean')
except ImportError:
    _ssim_fn = None



# A loss fn takes (example_dict, prediction_dict) → Tensor
LossFn = Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]
# A weight schedule takes (step:int) → float
WeightFn = Callable[[int], float]

def training_losses() -> Dict[str, Tuple[LossFn, WeightFn]]:
    """
    Hard‑coded training losses + schedules (as in film_net-Style.gin):
      - L1 loss: weight = 1.0 forever
      - VGG perceptual: weight = 1.0 until step 1.5M, then 0.25
      - Style       : weight = 1.0 until step 1.5M, then 40.0
    Returns:
      {
        'l1':    (l1_fn,    l1_weight_fn),
        'k*vgg': (vgg_fn,   vgg_weight_fn),
        'k*style':(style_fn,style_weight_fn),
      }
    """
    # 1) Get the actual loss functions by name
    l1_fn    = get_loss('l1')
    vgg_fn   = get_loss('vgg')
    style_fn = get_loss('style')

    # 2) Define their step‑dependent weight schedules
    def l1_wt(step: int) -> float:
        return 1.0

    def vgg_wt(step: int) -> float:
        # piecewise constant: 1.0 for steps <1.5M, else 0.25
        return 1.0 if step < 1_500_000 else 0.25

    def style_wt(step: int) -> float:
        # piecewise constant: 1.0 for steps <1.5M, else 40.0
        return 1.0 if step < 1_500_000 else 40.0

    # 3) Key names: omit 'k*' for constant‐1 weights, prefix 'k*' otherwise
    losses = {
        'l1':     (l1_fn,    l1_wt),
        'k*vgg':  (vgg_fn,   vgg_wt),
        'k*style':(style_fn, style_wt),
    }
    return losses



def vgg_loss(example: Dict[str, torch.Tensor],
             prediction: Dict[str, torch.Tensor],
             vgg_model_file: str,
             weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Perceptual VGG loss between prediction['image'] and example['y'].
    Delegates to your PyTorch vgg19_loss module.
    """
    return _vgg_loss(prediction['image'], example['y'], vgg_model_file, weights)


def style_loss(example: Dict[str, torch.Tensor],
               prediction: Dict[str, torch.Tensor],
               vgg_model_file: str,
               weights: Optional[List[float]] = None) -> torch.Tensor:
    """
    Style loss between prediction['image'] and example['y'].
    Delegates to your PyTorch vgg19_loss module.
    """
    return _style_loss(prediction['image'], example['y'], vgg_model_file, weights)


def l1_loss(example: Dict[str, torch.Tensor],
            prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean absolute error on the predicted image."""
    return F.l1_loss(prediction['image'], example['y'], reduction='mean')


def l1_warped_loss(example: Dict[str, torch.Tensor],
                   prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    L1 loss on the warped images (x0_warped, x1_warped) if present.
    """
    loss = 0.0
    if 'x0_warped' in prediction:
        loss = loss + F.l1_loss(prediction['x0_warped'], example['y'], reduction='mean')
    if 'x1_warped' in prediction:
        loss = loss + F.l1_loss(prediction['x1_warped'], example['y'], reduction='mean')
    return loss


def l2_loss(example: Dict[str, torch.Tensor],
            prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean squared error on the predicted image."""
    return F.mse_loss(prediction['image'], example['y'], reduction='mean')


def ssim_loss(example: Dict[str, torch.Tensor],
              prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Structural Similarity (SSIM) loss. Returns 1 − SSIM so that lower is better.
    Requires Kornia (pip install kornia).
    """
    if _ssim_fn is None:
        raise ImportError("Kornia is required for SSIM loss. Install with `pip install kornia`.")
    # SSIM returns a similarity in [0,1]; convert to a loss
    sim = _ssim_fn(prediction['image'], example['y'])
    return 1.0 - sim


def psnr_loss(example: Dict[str, torch.Tensor],
              prediction: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Peak Signal‑to‑Noise Ratio (PSNR) as a loss. We return −PSNR (in dB),
    so that minimizing this pushes PSNR higher.
    """
    mse = F.mse_loss(prediction['image'], example['y'], reduction='mean')
    # add small epsilon to avoid log(0)
    return -10.0 * torch.log10(mse + 1e-8)


# Map from string name → loss function
_LOSS_FUNCS: Dict[str, Callable[..., torch.Tensor]] = {
    'l1': l1_loss,
    'l2': l2_loss,
    'ssim': ssim_loss,
    'vgg': vgg_loss,
    'style': style_loss,
    'psnr': psnr_loss,
    'l1_warped': l1_warped_loss,
}


def get_loss(name: str) -> Callable[..., torch.Tensor]:
    """
    Retrieve a loss function by name.
    Example:
        loss_fn = get_loss('l1')
        loss = loss_fn(example_dict, prediction_dict)
    """
    try:
        return _LOSS_FUNCS[name]
    except KeyError:
        raise ValueError(f"Unknown loss: {name}")


def aggregate_batch_losses(
    batch_losses: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Given a list of per‐batch loss dicts, return the average loss per key.
    Example:
        batch_losses = [{'l1':0.2,'ssim':0.9}, {'l1':0.25,'ssim':0.88}]
        aggregate_batch_losses(batch_losses)
        → {'l1': 0.225, 'ssim': 0.89}
    """
    agg: Dict[str, List[float]] = {}
    for bl in batch_losses:
        for k, v in bl.items():
            agg.setdefault(k, []).append(v)
    return {k: sum(vals) / len(vals) for k, vals in agg.items()}
