# Saturation curve — NVIDIA GeForce RTX 4090
Backbone: `resnet18`, resolution: 128², bf16, eval mode

| n_imgs | time | ms/img | throughput | regime |
|---:|---:|---:|---:|---|
| 1 | 1.51 ms | 1.5116 ms | 662/s | flat (under-utilized) |
| 2 | 1.46 ms | 0.7324 ms | 1365/s | flat (under-utilized) |
| 4 | 1.48 ms | 0.3705 ms | 2699/s | flat (under-utilized) |
| 8 | 1.45 ms | 0.1811 ms | 5522/s | flat (under-utilized) |
| 16 | 1.46 ms | 0.0912 ms | 10961/s | flat (under-utilized) |
| 32 | 1.40 ms | 0.0438 ms | 22806/s | flat (under-utilized) |
| 48 | 1.53 ms | 0.0319 ms | 31305/s | flat (under-utilized) |
| 64 | 1.67 ms | 0.0261 ms | 38379/s | near-peak |
| 96 | 2.51 ms | 0.0261 ms | 38315/s | near-peak |
| 128 | 3.08 ms | 0.0241 ms | 41520/s | near-peak |
| 192 | 5.50 ms | 0.0286 ms | 34940/s | linear (compute-bound) |
| 256 | 7.00 ms | 0.0273 ms | 36583/s | linear (compute-bound) |
| 384 | 9.62 ms | 0.0251 ms | 39903/s | near-peak |
| 512 | 13.19 ms | 0.0258 ms | 38826/s | near-peak |
| 768 | 20.68 ms | 0.0269 ms | 37146/s | linear (compute-bound) |
| 1024 | 27.96 ms | 0.0273 ms | 36620/s | linear (compute-bound) |
| 1536 | 41.37 ms | 0.0269 ms | 37127/s | linear (compute-bound) |
| 2048 | 55.23 ms | 0.0270 ms | 37083/s | linear (compute-bound) |
