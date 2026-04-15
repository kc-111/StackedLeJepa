"""Synthetic data generators for regularization experiments."""

import math

import torch


def generate_blobs(K: int, N: int, num_blobs: int = 4, spread: float = 4.0,
                   blob_std: float = 0.3) -> torch.Tensor:
    """Mixture of well-separated Gaussian clusters.

    Cluster centers are arranged in a circle in the first two dimensions.

    Args:
        K: Number of points.
        N: Dimensionality.
        num_blobs: Number of clusters.
        spread: Radius of the circle on which centers are placed.
        blob_std: Standard deviation within each cluster.

    Returns:
        Tensor of shape (K, N).
    """
    centers = torch.zeros(num_blobs, N)
    angles = torch.linspace(0, 2 * math.pi, num_blobs + 1)[:num_blobs]
    centers[:, 0] = spread * angles.cos()
    if N >= 2:
        centers[:, 1] = spread * angles.sin()
    assignments = torch.randint(0, num_blobs, (K,))
    return centers[assignments] + torch.randn(K, N) * blob_std


def generate_diagonal_cross(K: int, N: int, arm_length: float = 3.0,
                            noise_std: float = 0.1) -> torch.Tensor:
    """Points along diagonal lines through hypercube vertices (X pattern).

    Creates 2^N arms radiating from the origin along the (±1, ±1, ..., ±1)
    diagonal directions. Points are uniformly distributed along each arm
    with Gaussian noise perpendicular to the arm. In 2D this produces an
    X shape; in higher dimensions a star of diagonal lines.

    Args:
        K: Number of points.
        N: Dimensionality.
        arm_length: Maximum distance from origin along each arm.
        noise_std: Standard deviation of Gaussian noise added perpendicular
            to the arm direction.

    Returns:
        Tensor of shape (K, N).
    """
    num_arms = 2 ** N
    # Diagonal directions: all {-1, +1}^N, normalized to unit vectors
    vertices = torch.tensor(
        [[((v >> d) & 1) * 2 - 1 for d in range(N)]
         for v in range(num_arms)],
        dtype=torch.float32,
    )
    dirs = vertices / vertices.norm(dim=1, keepdim=True)  # (num_arms, N)

    # Assign each point to a random arm
    arm_ids = torch.randint(0, num_arms, (K,))
    # Random distance along the arm (uniform from 0 to arm_length)
    t = torch.rand(K, 1) * arm_length

    points = t * dirs[arm_ids] + torch.randn(K, N) * noise_std
    return points


def generate_ring(K: int, N: int, radius: float = 3.0) -> torch.Tensor:
    """Points uniformly on a hypersphere shell.

    Args:
        K: Number of points.
        N: Dimensionality.
        radius: Radius of the shell.

    Returns:
        Tensor of shape (K, N).
    """
    z = torch.randn(K, N)
    z = z / z.norm(dim=1, keepdim=True)
    return z * radius


GENERATORS = {
    "blobs": generate_blobs,
    "diagonal_cross": generate_diagonal_cross,
    "ring": generate_ring,
}


def generate_data(name: str, K: int, N: int) -> torch.Tensor:
    """Generate synthetic data by name.

    Args:
        name: One of the keys in GENERATORS.
        K: Number of points.
        N: Dimensionality.

    Returns:
        Tensor of shape (K, N).
    """
    return GENERATORS[name](K, N)


def epoch_iter(K: int, batch_size: int, device, generator=None):
    """Yield batch_size-sized index chunks from a fresh permutation of [0, K).

    Defines an "epoch": one pass through the K data points without replacement.
    Drops the last partial chunk so each epoch yields exactly K // batch_size
    batches.

    Args:
        K: Total number of data points.
        batch_size: Indices per yielded chunk.
        device: Device for the index tensors.
        generator: Optional torch.Generator for reproducibility.

    Yields:
        1-D index tensors of shape (batch_size,) on ``device``.
    """
    perm = torch.randperm(K, generator=generator, device=device)
    n_chunks = K // batch_size
    for i in range(n_chunks):
        yield perm[i * batch_size:(i + 1) * batch_size]


def make_fixed_projection(N: int, M: int, seed: int = 0) -> torch.Tensor:
    """Create a fixed random projection matrix.

    Args:
        N: Input dimensionality.
        M: Output dimensionality (should be >= N).
        seed: Random seed for reproducibility.

    Returns:
        Tensor of shape (N, M).
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(N, M, generator=gen)
