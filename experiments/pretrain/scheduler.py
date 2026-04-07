"""Learning rate scheduler."""

import math
import torch


def make_scheduler(optimizer, warmup_steps, total_steps, base_lr, eta_min):
    """Linear warmup + cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 0.01)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return eta_min / base_lr + (1 - eta_min / base_lr) * 0.5 * (
            1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
