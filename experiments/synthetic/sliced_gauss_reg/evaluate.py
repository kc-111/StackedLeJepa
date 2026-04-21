"""Evaluation metrics for measuring distance to N(0, I)."""

import math

import numpy as np
import torch


def eval_w1(x: np.ndarray, num_proj: int = 1024) -> float:
    """Sliced Wasserstein-1 distance between samples and N(0, 1).

    Uses a fixed random seed for consistent evaluation across calls.

    Args:
        x: Array of shape (N, D).
        num_proj: Number of random projection directions.

    Returns:
        Scalar W1 distance.
    """
    t = torch.tensor(x, dtype=torch.float32)
    n, d = t.shape
    torch.manual_seed(999)
    A = torch.randn(d, num_proj)
    A = A / A.norm(dim=0, keepdim=True)
    proj_sorted = torch.sort(t @ A, dim=0).values
    p = (torch.arange(1, n + 1, dtype=torch.float32) - 0.5) / n
    ref = torch.erfinv(2 * p - 1) * math.sqrt(2)
    return (proj_sorted - ref.unsqueeze(1)).abs().mean(0).mean().item()


def eval_w2(x: np.ndarray, num_proj: int = 1024) -> float:
    """Sliced Wasserstein-2 distance between samples and N(0, 1).

    Args:
        x: Array of shape (N, D).
        num_proj: Number of random projection directions.

    Returns:
        Scalar W2 distance.
    """
    t = torch.tensor(x, dtype=torch.float32)
    n, d = t.shape
    torch.manual_seed(999)
    A = torch.randn(d, num_proj)
    A = A / A.norm(dim=0, keepdim=True)
    proj_sorted = torch.sort(t @ A, dim=0).values
    p = (torch.arange(1, n + 1, dtype=torch.float32) - 0.5) / n
    ref = torch.erfinv(2 * p - 1) * math.sqrt(2)
    return ((proj_sorted - ref.unsqueeze(1)) ** 2
            ).mean(0).sqrt().mean().item()


def evaluate_full(output: np.ndarray, num_proj: int = 1024) -> dict:
    """Full evaluation of how close a distribution is to N(0, I).

    Args:
        output: Array of shape (N, D).
        num_proj: Number of projections for W1/W2.

    Returns:
        Dict with keys: w1, w2, mean_mse, cov_frob, cov_diag,
        cov_offdiag_max.
    """
    D = output.shape[1]
    mu = output.mean(axis=0)
    cov = np.cov(output, rowvar=False)
    return {
        "w1": eval_w1(output, num_proj),
        "w2": eval_w2(output, num_proj),
        "mean_mse": float(np.sum(mu ** 2)),
        "cov_frob": float(
            np.linalg.norm(cov - np.eye(D), "fro")
            / np.linalg.norm(np.eye(D), "fro")),
        "cov_diag": np.diag(cov).tolist(),
        "cov_offdiag_max": float(
            np.abs(cov - np.diag(np.diag(cov))).max()),
    }


def evaluate_full_gpu(output: torch.Tensor, num_proj: int = 1024) -> dict:
    """GPU/torch version of ``evaluate_full``. Same dict, no CPU temporaries.

    Sweeps that call ``evaluate_full`` once per epoch leak resident memory
    via glibc malloc fragmentation: each call materializes ~135 MB of CPU
    tensors (two ``(N, num_proj)`` matrices in ``eval_w1`` plus another two
    in ``eval_w2``), and large free blocks are not returned to the OS.
    Running the same math on the device the embeddings already live on uses
    the CUDA caching allocator (or just process-local CPU buffers reused
    in place) and keeps RSS flat across the sweep.
    """
    out = output.detach().float()
    n, d = out.shape
    device = out.device

    g = torch.Generator(device=device).manual_seed(999)
    A = torch.randn(d, num_proj, device=device, generator=g)
    A = A / A.norm(dim=0, keepdim=True)
    proj_sorted = torch.sort(out @ A, dim=0).values
    p = (torch.arange(1, n + 1, device=device, dtype=torch.float32) - 0.5) / n
    ref = (torch.erfinv(2 * p - 1) * math.sqrt(2)).unsqueeze(1)
    diff = proj_sorted - ref
    w1 = diff.abs().mean().item()
    w2 = diff.square().mean(dim=0).sqrt().mean().item()

    mu = out.mean(dim=0)
    centered = out - mu
    cov = centered.T @ centered / max(n - 1, 1)
    eye = torch.eye(d, device=device, dtype=cov.dtype)
    cov_frob = ((cov - eye).norm() / eye.norm()).item()
    diag = torch.diagonal(cov)
    off = cov - torch.diag(diag)
    return {
        "w1": w1,
        "w2": w2,
        "mean_mse": float((mu * mu).sum().item()),
        "cov_frob": cov_frob,
        "cov_diag": diag.detach().cpu().tolist(),
        "cov_offdiag_max": off.abs().max().item(),
    }
