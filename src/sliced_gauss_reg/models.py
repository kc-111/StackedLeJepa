"""Neural network models for regularization experiments."""

import torch.nn as nn


class DeepMLP(nn.Module):
    """Multi-layer perceptron with GELU activations.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Width of hidden layers.
        output_dim: Output dimension.
        depth: Number of hidden layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 depth: int = 3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(in_d, hidden_dim), nn.GELU()]
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        return self.net(x)
