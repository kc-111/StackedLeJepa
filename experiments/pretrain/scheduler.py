"""Learning rate scheduler."""

import torch


def make_scheduler(optimizer, warmup_steps, base_lr):
    """Linear warmup then constant LR."""
    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 0.01)
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
