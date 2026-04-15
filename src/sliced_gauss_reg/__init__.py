"""Sliced Gaussian regularization.

Model-agnostic loss modules for pushing embedding distributions toward
N(0, I) using sliced 1D statistics:

- ``SlicedW1Loss`` / ``SlicedW2Loss``: sort-based 1D Wasserstein distances
  to standard-normal quantiles, averaged over random projections.
- ``SIGReg`` / ``SIGRegLoss``: Epps-Pulley characteristic-function test
  (with biased / U-stat / sample-split variants).
- ``PooledSlicedLoss``: pooled 2-step procedure that breaks the small-batch
  bias floor by collecting many exact i.i.d. embeddings under no-grad and
  computing the loss on the full pooled set with one gradient batch.
"""

from .sigreg import SIGReg
from .losses import (
    PooledSlicedLoss,
    SIGRegLoss,
    SlicedW1Loss,
    SlicedW2Loss,
    pooled_loss,
)
from .data import generate_data, make_fixed_projection, epoch_iter, GENERATORS
from .models import DeepMLP
from .evaluate import eval_w1, eval_w2, evaluate_full, evaluate_full_gpu
