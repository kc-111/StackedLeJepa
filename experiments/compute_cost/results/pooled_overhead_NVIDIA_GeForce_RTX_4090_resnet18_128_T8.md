# Pooled overhead — NVIDIA GeForce RTX 4090

- **Backbone**: `resnet18` (resnet18)
- **Resolution**: 128²
- **Views**: V_g=2 (no locals)
- **T (pool depth)**: 8
- **Precision**: bf16
- **Iters per cell**: 12 (mean ± std)
- **Each cell INCLUDES** sample_batch + gpu_aug (the full per-step cost as it appears in training)

## Variants
- `std` — standard step (1F + 1B on BS samples)
- `pool_nograd` — pooled step, ONE big no-grad forward of (T-1)·BS samples + 1F + 1B on BS
- `pool_nograd_chunked` — pooled step, (T-1) sequential no-grad forwards of BS samples + 1F + 1B on BS. Same total compute, lower peak memory.

## Results — timing (mean ± std)

| BS | std | pool_nograd | pool_chunked | pool/std | chunked/std |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.41 ± 0.44 ms | 9.90 ± 0.54 ms | 24.84 ± 0.55 ms | 1.54× | 3.87× |
| 16 | 6.67 ± 0.46 ms | 15.48 ± 0.36 ms | 26.92 ± 2.47 ms | 2.32× | 4.03× |
| 32 | 8.90 ± 0.79 ms | 26.60 ± 0.81 ms | 30.02 ± 0.79 ms | 2.99× | 3.37× |
| 64 | 14.56 ± 0.57 ms | 54.67 ± 0.83 ms | 50.01 ± 0.84 ms | 3.76× | 3.43× |

## Results — peak GPU memory (per method)

| BS | std | pool_nograd | pool_chunked | pool / std | chunked / std |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.91 GB | 0.96 GB | 0.91 GB | 1.05× | 1.00× |
| 16 | 0.97 GB | 1.17 GB | 0.98 GB | 1.20× | 1.01× |
| 32 | 1.10 GB | 1.43 GB | 1.12 GB | 1.30× | 1.01× |
| 64 | 1.36 GB | 2.06 GB | 1.39 GB | 1.51× | 1.02× |
