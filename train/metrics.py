# metrics.py

import torch
from typing import Dict, Any

# Import your zero‑arg loss factories from your PyTorch losses module
from .losses_config import training_losses, test_losses  # both are @gin.configurable

class TrainLossMetric:
    """Compute the total training loss (with schedules) as an eval metric."""
    def __init__(self, name: str = 'training_loss'):
        self.name = name
        self.reset()

    def reset(self):
        self._acc = 0.0
        self._count = 0

    def update(self, batch: Dict[str, torch.Tensor],
                     preds: Dict[str, torch.Tensor],
                     step: int = 0):
        losses = training_losses()  # gets (loss_fn, weight_fn) via Gin
        total = 0.0
        for loss_fn, weight_fn in losses.values():
            val = loss_fn(batch, preds)
            w   = weight_fn(step)
            total = total + (val * w).item()
        self._acc   += total
        self._count += 1

    def compute(self) -> float:
        return self._acc / self._count if self._count > 0 else 0.0


class L1Metric:
    """Plain L1 loss as an eval metric."""
    def __init__(self, name: str = 'l1'):
        self.name = name
        self.reset()

    def reset(self):
        self._acc = 0.0
        self._count = 0

    def update(self, batch: Dict[str, torch.Tensor],
                     preds: Dict[str, torch.Tensor],
                     step: int = 0):
        from .losses import l1_loss  # raw L1 loss fn
        val = l1_loss(batch, preds)
        self._acc   += val.item()
        self._count += 1

    def compute(self) -> float:
        return self._acc / self._count if self._count > 0 else 0.0


class GenericLossMetric:
    """Wrap any (loss_fn, weight_fn) as a metric."""
    def __init__(self,
                 name: str,
                 loss_fn: Any,
                 weight_fn: Any):
        self.name = name
        self.loss_fn = loss_fn
        self.weight_fn = weight_fn
        self.reset()

    def reset(self):
        self._acc = 0.0
        self._count = 0

    def update(self, batch: Dict[str, torch.Tensor],
                     preds: Dict[str, torch.Tensor],
                     step: int = 0):
        val = self.loss_fn(batch, preds)
        w   = self.weight_fn(step)
        self._acc   += (val * w).item()
        self._count += 1

    def compute(self) -> float:
        return self._acc / self._count if self._count > 0 else 0.0


def create_metrics_fn() -> Dict[str, Any]:
    """
    Instantiate all evaluation metrics:
      - 'l1'           → L1Metric
      - 'training_loss'→ TrainLossMetric
      - one GenericLossMetric per entry in test_losses()
    """
    metrics: Dict[str, Any] = {}
    metrics['l1']            = L1Metric(name='l1')
    metrics['training_loss'] = TrainLossMetric(name='training_loss')

    # test_losses() is @gin.configurable and returns {name:(loss_fn,weight_fn)}
    for name, (loss_fn, weight_fn) in test_losses().items():
        metrics[name] = GenericLossMetric(name, loss_fn, weight_fn)

    return metrics
