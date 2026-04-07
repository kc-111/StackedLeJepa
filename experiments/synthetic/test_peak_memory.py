"""
Peak GPU memory test: T-1 separate no-grad passes vs 1 large no-grad pass.

Shows that storing detached embeddings is negligible — the entire CIFAR
training set (50K × 192-dim) costs only ~36 MB. So we can forward all
training samples in one no-grad pass for maximum CDF resolution.

Run:
    python experiments/synthetic/test_peak_memory.py
"""

import os
import sys
import argparse

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.accumulated_w1 import DeepMLP, generate_data, make_fixed_projection


def test_synthetic(device):
    """Compare separate vs single pass on synthetic data."""
    D, K, M, BS, T = 4, 1024, 32, 64, 16

    torch.manual_seed(42)
    data = generate_data("blobs", K, D)
    W = make_fixed_projection(D, M, seed=100).to(device)
    projected = data.to(device) @ W

    torch.manual_seed(200)
    mlp = DeepMLP(M, 128, D, depth=2).to(device)

    # Warmup
    with torch.no_grad():
        _ = mlp(projected[:BS])
    torch.cuda.synchronize()

    # --- T-1 separate mini-batch no-grad passes ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    detached = []
    with torch.no_grad():
        for _ in range(T - 1):
            idx = torch.randint(0, K, (BS,), device=device)
            detached.append(mlp(projected[idx]).detach())

    idx = torch.randint(0, K, (BS,), device=device)
    live = mlp(projected[idx])
    all_emb = torch.cat([live] + detached, dim=0)

    peak_separate = torch.cuda.max_memory_allocated()
    print(f"=== T-1 separate passes (T={T}, BS={BS}) ===")
    print(f"  CDF samples: {all_emb.shape[0]}")
    print(f"  Peak memory: {peak_separate / 1024**2:.1f} MB")

    del detached, live, all_emb
    torch.cuda.empty_cache()

    # --- 1 large no-grad pass ---
    torch.cuda.reset_peak_memory_stats()

    large_bs = (T - 1) * BS
    with torch.no_grad():
        idx_large = torch.randint(0, K, (large_bs,), device=device)
        detached_large = mlp(projected[idx_large]).detach()

    idx = torch.randint(0, K, (BS,), device=device)
    live = mlp(projected[idx])
    all_emb = torch.cat([live, detached_large], dim=0)

    peak_single = torch.cuda.max_memory_allocated()
    print(f"\n=== 1 large pass (BS_nograd={large_bs}) ===")
    print(f"  CDF samples: {all_emb.shape[0]}")
    print(f"  Peak memory: {peak_single / 1024**2:.1f} MB")

    del detached_large, live, all_emb
    torch.cuda.empty_cache()

    print(f"\n  Ratio: {peak_single / peak_separate:.2f}x")


def test_embedding_storage(device):
    """How much memory to store N embeddings of dim D?"""
    print("\n\n=== Embedding storage cost ===")
    print(f"  {'N':>8}  {'dim':>5}  {'MB':>8}")
    print(f"  {'-'*25}")
    for N in [1024, 5000, 10000, 50000, 100000]:
        for dim in [16, 192, 384, 768]:
            mb = N * dim * 4 / 1024**2
            print(f"  {N:>8}  {dim:>5}  {mb:>8.2f}")
        print()


def test_scaling(device):
    """Peak memory vs no-grad batch size with a realistic MLP."""
    print("\n=== Peak memory vs no-grad batch size ===")
    D, K, M = 4, 4096, 32

    torch.manual_seed(42)
    data = generate_data("blobs", K, D)
    W = make_fixed_projection(D, M, seed=100).to(device)
    projected = data.to(device) @ W

    torch.manual_seed(200)
    mlp = DeepMLP(M, 512, D, depth=4).to(device)

    # Warmup
    with torch.no_grad():
        _ = mlp(projected[:8])
    torch.cuda.synchronize()

    grad_bs = 8
    print(f"  grad_bs={grad_bs}")
    print(f"  {'nograd_bs':>10}  {'CDF':>6}  {'peak_MB':>8}  {'emb_KB':>8}")
    print(f"  {'-'*38}")

    for nograd_bs in [64, 128, 256, 512, 1024, 2048, 4096]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        with torch.no_grad():
            idx = torch.randint(0, K, (nograd_bs,), device=device)
            det = mlp(projected[idx]).detach()

        idx = torch.randint(0, K, (grad_bs,), device=device)
        live = mlp(projected[idx])
        all_e = torch.cat([live, det], dim=0)

        peak = torch.cuda.max_memory_allocated()
        emb_kb = det.numel() * 4 / 1024
        print(f"  {nograd_bs:>10}  {nograd_bs + grad_bs:>6}  "
              f"{peak / 1024**2:>8.1f}  {emb_kb:>8.1f}")

        del det, live, all_e


def main():
    p = argparse.ArgumentParser(description="Peak memory test")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type != "cuda":
        print("CUDA not available, skipping memory test")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")

    test_synthetic(device)
    test_embedding_storage(device)
    test_scaling(device)


if __name__ == "__main__":
    main()
