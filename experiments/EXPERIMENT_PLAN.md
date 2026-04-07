# Experiment Plan — Pooled Distributional Regularization (LeJEPA)

Concrete experiment list, figure/table mapping, and time budget for the
paper. LeWorldModel section is a future extension and is *not* covered here.

---

## Story

Standard sliced-distributional regularizers (W1, W2, sliced-Sobolev/SIGReg)
have a finite-sample bias floor of `O(1/√n)` that **no amount of training time
can fix**. We propose **pooled distributional regularization** as a drop-in
**late-stage refinement**: train normally with LeJEPA, then continue for
K=50 epochs with a pooled regularizer to extract a relative improvement that
the standard method cannot reach.

The story is told as:
1. **Theory + synthetic**: bias floor exists; pooling breaks it (`base_plan.md` Phase 1)
2. **Vision pretraining**: train standard LeJEPA → continuation with 4 methods → headline relative improvement, bs sweep
3. **Architecture coverage**: same effect across 4 backbones
4. **Multi-dataset breadth**: 8 vision datasets

---

## Default training config

| Setting | Value | Rationale |
|---|---|---|
| Backbone (default) | `resnet18` | Paper Table 5; 11M params; fastest measured |
| Base batch size | **8** | Smallest practical bs — bias floor pinned hardest here |
| Base epochs | 200 | Enough for convergence at bs=8 |
| Continuation epochs | **50** | Enough to see clear divergence between methods |
| Continuation bs values | **{8, 16, 32, 64, 128}** | 5-point sweep for Fig 3 |
| Optimizer | AdamW lr=1e-3, wd=5e-2 | Paper default |
| λ (reg weight) | 0.05 | Paper default |
| `accum_steps` (T) | 8 | Pool size = 8 × bs |
| `num_proj` | 1024 | Paper default |
| Multi-crop | V_g=2 @ **128** (CNN: V_l=0; ViT: V_l=6 @ 56) | **3× faster than paper's 224**, run sanity check at 224 |
| Memory format | `channels_last` | ~1.3× speedup |
| `torch.compile` | mode=`default` | ~1.4× speedup |

### NOTE: GPU saturation curve — why pooled is "almost free" at small batch sizes

This is the **key cost insight** for the paper. See `experiments/compute_cost/`
for the benchmark scripts, raw data tables, and plots.

**Empirical saturation curve** (RTX 4090, ResNet18 forward at 128² resolution,
bf16, eval mode):

| n_imgs | time | ms/img | regime |
|---:|---:|---:|---|
| 1 | 1.51 ms | 1.51 | flat (under-utilized) |
| 8 | 1.49 ms | 0.19 | flat |
| 32 | 1.40 ms | 0.044 | flat |
| **64** | **1.67 ms** | **0.026** | **knee** |
| **128** | **3.08 ms** | **0.024** | **saturation point** |
| 256 | 7.00 ms | 0.027 | linear (compute-bound) |
| 512 | 13.19 ms | 0.026 | linear |
| 1024 | 27.96 ms | 0.027 | linear |
| 2048 | 55.23 ms | 0.027 | linear |

**Two regimes**:
- **n ≤ 64**: time is essentially constant. The GPU is under-utilized; adding more
  samples is free. ResNet18 at 128² doesn't have enough work per image to keep
  16,384 CUDA cores busy.
- **n ≥ 128**: time scales linearly with n. The GPU is compute-bound; each extra
  sample adds proportional cost.

### Why pooled training is almost free at small BS

Per-step cost (timing INCLUDES sample_batch + gpu_aug). RTX 4090, bf16,
T=8, resolution 128². See `experiments/compute_cost/results/` for the raw
markdown tables and `experiments/compute_cost/plots/` for the figures.

**ResNet18 (V_g=2, no locals)**:

| BS | std | pool_nograd | pool / std | std GB | pool GB |
|---:|---:|---:|---:|---:|---:|
| **8** | 5.5 ms | **8.2 ms** | **1.50×** | 0.91 | 0.96 |
| 16 | 7.3 ms | 14.1 ms | 1.93× | 0.97 | 1.17 |
| 32 | 8.2 ms | 27.8 ms | 3.41× | 1.10 | 1.43 |
| 64 | 12.9 ms | 47.4 ms | 3.67× | 1.36 | 2.06 |

**ViT-tiny (V_g=2 + V_l=6 multi-crop)**:

| BS | std | pool_nograd | pool / std | std GB | pool GB |
|---:|---:|---:|---:|---:|---:|
| **8** | 16.8 ms | **23.2 ms** | **1.38×** | 0.91 | 0.95 |
| **16** | 17.2 ms | **23.5 ms** | **1.37×** | 1.08 | 1.16 |
| 32 | 17.4 ms | 30.5 ms | 1.75× | 1.42 | 1.58 |

**The headline finding**: at small BS, pooled training adds only **1.3-1.5×
wall clock** vs standard. At BS=8 ResNet18 it's exactly 1.50×. ViT-tiny multi-crop
stays at ~1.4× through BS=16 because each BS sample expands to 8 view-images,
keeping the GPU in its under-utilized regime longer.

This is **exactly the regime where the bias floor matters most**: small batches
have high finite-sample bias for distributional regularizers, and that's where
pooling helps most. Pooling is cheapest precisely where it's most useful.

ViT also demands more memory per BS step but the compute time stays roughly
flat through BS=16 (all in the under-utilized regime).

### Implications for the paper

**Fig 3 (the headline)** plots accuracy vs BS for sigreg / sigreg_pooled / w1 /
w1_pooled. The **compute cost story is implicitly already in the figure** — at
small BS, the pooled methods cost almost nothing extra, but they're the only ones
that escape the bias floor. At large BS, standard methods are already adequate
and pooling becomes optional.

**Compute-cost subsection text** (draft):
> "Pooled distributional regularization is essentially free in the regime where it
> matters most. At BS=8 on a single 4090, pooled training adds only 1.3-1.5× wall
> clock vs standard training (see Fig X / Tab Y), because the GPU is under-utilized
> at small batch sizes — adding the no-grad pass through (T-1)·BS samples fills
> previously idle compute. As BS grows past the GPU saturation point (~128 samples
> for ResNet18 at 128², ~256 for ViT-tiny multi-crop), the relative cost of pooling
> grows linearly. Crucially, this is exactly where standard methods are already
> sufficient: the small-batch regime where standard fails is the same regime where
> pooled is cheapest."

### Variants we benchmarked

| Variant | What it does | Used in production? |
|---|---|---|
| `std` | Standard step: 1F + 1B on BS samples | yes (baseline) |
| `pool_nograd` | Pooled step: 1F no-grad on (T-1)·BS + 1F + 1B on BS. Pass 1 stores no activations. | **yes** (our method) |
| `pool_nograd_chunked` | Same as `pool_nograd` but Pass 1 is split into (T-1) sub-batches of BS each, processed sequentially | **no** — supplementary discussion only |

### Supplementary discussion: chunked Pass 1

We also benchmarked a "chunked" variant of `pool_nograd` that splits the Pass 1
no-grad forward into (T-1) sequential sub-batches of BS each, instead of one big
forward of (T-1)·BS samples. The motivation: cuDNN intermediate buffers are
sized per-call, so smaller chunks → smaller peak memory.

**Results** (line plots in `experiments/compute_cost/plots/pooled_supplementary.png`):

- **ResNet18, BS=64**: chunked is **slightly faster** (-9%) and uses **32% less
  peak memory**. The (T-1)=7 sub-batches of 128 image-forwards each sit at the
  GPU saturation knee, which is more efficient than one big 896-image forward.
- **ResNet18, BS=8-32**: chunked is *slower* (+10-50%) because each sub-batch
  is too small and kernel-launch overhead dominates.
- **ViT-tiny multicrop, all BS**: chunked is **2-3× slower** because each chunk
  triggers two encoder calls (global + local views), so 7 chunks = 14 small
  forward passes — kernel launch overhead dominates entirely.
- **Memory savings**: 22-32% on ResNet18, ~2-9% on ViT.

**Why we don't use it in production:**

1. **We don't need the memory savings.** On the 4090 with our current config,
   `pool_nograd` peaks at ~2 GB out of 24 GB available. We're at 8% memory
   utilization. There's nothing to save by being clever.
2. **Chunking is only competitive at one BS value** (ResNet18, BS=64). Everywhere
   else it's slower or much slower.
3. **BN consistency adds complexity.** Our production code uses a "capture and
   inject" mechanism so both Pass 1 and Pass 2 use identical BN normalization
   stats. Splitting Pass 1 into chunks would require either parallel-variance
   aggregation across chunks (extra implementation) or eval-mode chunking
   (which forfeits the BN running-stat update from the larger pool).

**What this tells us:** the production approach (one big no-grad forward) is the
right default. Chunking is a knob you'd want only if memory becomes the
binding constraint — e.g., when training a much larger model or at much larger
BS than our experiment range. We mention it in the supplementary as evidence
that we considered the alternative and have a measured reason to not use it.

---

## Methods compared (4)

| Tag | regularizer | accumulate | Role |
|---|---|---|---|
| `sigreg` | sigreg | False | LeJEPA baseline (control) |
| `sigreg_pooled` | sigreg | True | Pool the ECF (our contribution applied to SIGReg) |
| `w1` | w1 | False | Sliced W1 single-batch |
| `w1_pooled` | w1 | True | **Headline**: pooled sort breaks the bias floor |

**Note on seeds:** `sigreg` is the **baseline starting point** (the converged
base checkpoint, single seed). All four continuation methods (including
`sigreg`-as-continuation, which is the "do nothing different" control) get
**3 seeds** with different RNG / data shuffling to give error bars on the
comparison.

---

## Experiment structure

### Phase 0 — Base trainings (single seed each)

For each of **`cifar100`**, **`food101`**, **`flowers102`**, train standard
`sigreg` at **bs=8** for **200 epochs** to convergence. This produces the
checkpoint that all continuations start from.

These three datasets span **three orders of magnitude in dataset size**:
- `flowers102`: 1,020 train (extreme small data)
- `cifar100`: 50,000 train (medium)
- `food101`: 75,750 train (large)

- **3 base runs**, single seed
- Output: `runs/base/{dataset}_resnet18_sigreg_bs8_seed42/final.pt`

### Phase 1 — Headline batch-size continuation sweep (Fig 3)

From each base checkpoint, continue for **50 epochs** with each of the 4
methods at each of **5 batch sizes** with **3 seeds each**.

- 3 datasets × 5 cont bs × 4 methods × 3 seeds = **180 continuation runs**
- Each cont run ≈ 15-100 min depending on bs and method

### Phase 2 — Headline all-datasets table (Tab 1)

For each of the 5 *additional* datasets (`cifar10`, `stl10`, `flowers102`,
`dtd`, `aircraft`), do a single base training + 4-method × 3-seed
continuation at the **headline cont bs** (default 32).

- 5 base runs + 5 ds × 4 methods × 3 seeds = **65 runs**
- (`cifar100`, `flowers102`, `pets` are reused from Phase 1)
- (`food101` is *omitted* — DataLoader path is too slow for the budget)

### Phase 3 — Architecture sweep (Tab 2)

For each of `resnet18`, `resnet34`, `convnextv2_nano`, `vit_tiny`, train base
on `cifar100` / `flowers102` / `pets`, then continue with 2 methods
(`sigreg` baseline + `w1_pooled` headline) at the headline cont bs.

- 4 archs × 3 ds × (1 base + 2 methods × 3 seeds) = 12 base + 72 cont
- − 18 reused (resnet18 rows already done in Phase 1) = **66 runs**

---

## Total run count

| Phase | Runs |
|---|---:|
| Phase 0 — Base trainings | 3 |
| Phase 1 — Fig 3 (cont bs sweep) | 180 |
| Phase 2 — Tab 1 (8-dataset headline) | 65 |
| Phase 3 — Tab 2 (arch sweep) | 66 |
| **Total** | **314** |

### Time budget (RTX 4090, single GPU, compile + channels_last + 128² resolution)

| Phase | Avg per run | Total |
|---|---:|---:|
| Phase 0 (3 bases at 200 ep, bs=8) | ~2 h | ~6 h |
| Phase 1 (180 cont at 50 ep, mixed bs) | ~10 min | ~30 h |
| Phase 2 (65 runs at 50 ep) | ~10 min | ~11 h |
| Phase 3 (66 runs, mixed archs) | ~17 min (some ViT) | ~19 h |
| Sanity check at 224² (Tab S1, 4 runs) | ~30 min | ~2 h |
| **Total** |  | **~68 h ≈ 3 days** |

This is the *Option A* plan (3 seeds for every non-baseline method) at the
new 128² default. With ~3 days of GPU time it's comfortable for a NeurIPS
deadline. The 224² sanity check covers the "but does it hold at full
resolution?" reviewer question.

---

## Figures and tables

| # | What | Source | Status |
|---|---|---|---|
| **Fig 1** | Method illustration (diagram) | none | manual |
| **Fig 2** | Synthetic — bias floor + pooled fix | `base_plan.md` Phase 1 | small compute |
| **Fig 3** | **HEADLINE**: relative improvement vs continuation bs, 3 panels (one per dataset), 4 method curves, 3-seed error bars | Phase 1 | new |
| **Fig 4** | Training curves (base plateau → continuation divergence) | derived from Phase 1 | reused |
| **Fig 5** *(supp)* | Per-dataset bar chart of relative improvement at the headline cont bs | derived from Phase 2 | reused |
| **Tab 1** | All-datasets headline: 8 datasets × 4 methods, mean ± std over 3 seeds | Phase 2 | new |
| **Tab 2** | Architecture sweep: 4 archs × 3 datasets × 2 methods, mean ± std | Phase 3 | new |
| **Tab 3** *(supp)* | Lambda sensitivity on cifar100 | optional | optional |

**5 figures (3 main + 2 supp/derived), 3 tables (2 main + 1 supp).** Comfortable NeurIPS layout.

---

## How to launch

```bash
# Phase 0: Base trainings (3 runs)
python experiments/pretrain/run_sweep.py --plan phase0_base \
    --save-dir runs/lejepa_v1

# Phase 1: Fig 3 — bs sweep continuation (180 runs)
python experiments/pretrain/run_sweep.py --plan phase1_fig3 \
    --save-dir runs/lejepa_v1

# Phase 2: Tab 1 — all-datasets headline (65 runs)
python experiments/pretrain/run_sweep.py --plan phase2_tab1 \
    --save-dir runs/lejepa_v1

# Phase 3: Tab 2 — architecture sweep (66 runs)
python experiments/pretrain/run_sweep.py --plan phase3_tab2 \
    --save-dir runs/lejepa_v1
```

Each plan resumes automatically — `trainer.py` skips runs whose `final.pt` already exists.

---

## Continuation training mechanics

`trainer.py` supports a `--continuation` mode:

```bash
python experiments/pretrain/trainer.py \
    --continuation runs/base/cifar100_resnet18_sigreg_bs8_seed42/final.pt \
    --regularizer w1 --accumulate \
    --batch-size 32 --epochs 50 --seed 0 \
    --save-dir runs/lejepa_v1
```

Behavior:
1. Load the encoder + probe state from the checkpoint
2. **Reset** optimizer + scheduler (we're starting a new training phase with possibly different hyperparams)
3. **Reset** the LR schedule (linear warmup + cosine over the new K=50 epochs)
4. Run continuation with the chosen `--regularizer` / `--accumulate` / `--batch-size`
5. Save final to `runs/lejepa_v1/{ds}_{arch}_{method}_bs{bs}_seed{seed}_cont/final.pt`

The continuation run dir is namespaced by both base config and continuation config, so different continuation configs from the same base don't collide.

---

## Open decisions

1. **Headline continuation bs** for Tab 1 / Tab 2 (the "single bs" picked from the Fig 3 sweep): default **32**, but could be 16 if we expect smallest batches to give the biggest relative gap.
2. **Tab 3 (lambda sweep)**: include or skip? Adds ~10 runs, ~10 h.
3. **Architectures in Tab 2**: keep `vit_tiny` (slower)? Or restrict to CNNs?
4. **food101**: confirmed *out* of Fig 3 (too slow), but should it appear in Tab 1?
   - In: gives "big data" point (1 base + 12 cont = 13 extra runs, ~30 h)
   - Out: cleanest budget

---

## Future work (not in this plan)

- **LeWorldModel section** — apply same pooled continuation idea to LeWM published checkpoints, measure planning success rate. To be added in a follow-up plan revision.
- **ImageNet-100** — if compute allows, add as a 9th dataset.
