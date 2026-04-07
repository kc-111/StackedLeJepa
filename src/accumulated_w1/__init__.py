"""Accumulated sliced Wasserstein regularization.

Provides model-agnostic loss modules for pushing embedding distributions
toward N(0, I) using sliced Wasserstein distances with gradient accumulation.

The key innovation: instead of computing W1 on small mini-batches (noisy),
freeze the encoder for N steps, collect N×batch_size exact i.i.d. embeddings,
and compute W1 on all of them in one pooled sort. Gradient flows only through
the last batch; the rest provide CDF context. This gives dramatically better
quantile resolution without any staleness or approximation.
"""

from .losses import (
    AccumulatedSlicedLoss,
    SIGRegLoss,
    SlicedW1Loss,
    SlicedW2Loss,
    pooled_loss,
)
from .data import generate_data, make_fixed_projection, GENERATORS
from .models import DeepMLP
from .evaluate import eval_w1, eval_w2, evaluate_full
