# Pooled overhead — NVIDIA GeForce RTX 4090

- **Backbone**: `tiny` (tiny)
- **Resolution**: 128² (global), 64² (local)
- **Views**: V_g=2 + V_l=6 (multi-crop)
- **T (pool depth)**: 8
- **Precision**: bf16
- **Iters per cell**: 10 (mean ± std)
- **Each cell INCLUDES** sample_batch + gpu_aug (the full per-step cost as it appears in training)

## Variants
- `std` — standard step (1F + 1B on BS samples)
- `pool_nograd` — pooled step, ONE big no-grad forward of (T-1)·BS samples + 1F + 1B on BS
- `pool_nograd_chunked` — pooled step, (T-1) sequential no-grad forwards of BS samples + 1F + 1B on BS. Same total compute, lower peak memory.

## Results — timing (mean ± std)

| BS | std | pool_nograd | pool_chunked | pool/std | chunked/std |
|---:|---:|---:|---:|---:|---:|
| 8 | 18.33 ± 1.05 ms | 25.56 ± 1.38 ms | 64.24 ± 1.24 ms | 1.39× | 3.50× |
| 16 | 17.83 ± 2.11 ms | 24.47 ± 1.07 ms | 66.75 ± 3.05 ms | 1.37× | 3.74× |
| 32 | 17.71 ± 0.82 ms | 34.04 ± 0.67 ms | 64.70 ± 1.88 ms | 1.92× | 3.65× |
| 64 | 21.04 ± 1.47 ms | 63.89 ± 4.50 ms | 68.21 ± 1.91 ms | 3.04× | 3.24× |

## Results — peak GPU memory (per method)

| BS | std | pool_nograd | pool_chunked | pool / std | chunked / std |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.91 GB | 0.95 GB | 0.92 GB | 1.04× | 1.01× |
| 16 | 1.08 GB | 1.16 GB | 1.09 GB | 1.07× | 1.01× |
| 32 | 1.42 GB | 1.58 GB | 1.45 GB | 1.11× | 1.02× |
| 64 | 2.10 GB | 2.41 GB | 2.15 GB | 1.15× | 1.02× |
