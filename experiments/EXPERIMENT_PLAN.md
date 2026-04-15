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
| Backbone (default) | `convnextv2_pico` | LayerNorm-only CNN; 8.6M params, 512-d features (same as resnet18), ~2x faster than nano. See "Note on BatchNorm" below for why we default to LN. |
| Base batch size | **8** | Smallest practical bs — bias floor pinned hardest here |
| Base epochs | 200 | Enough for convergence at bs=8 |
| Continuation epochs | **50** | Enough to see clear divergence between methods |
| Continuation bs values | **{8, 16, 32, 48, 64}** | 5-point sweep for Fig 3. BS=128 dropped — past convnextv2_nano's saturation knee (~48), adds wall-clock without adding physics. BS=48 added to anchor the saturation knee. |
| Optimizer | AdamW lr=1e-3, wd=5e-2 | Paper default |
| λ (reg weight) | 0.05 | Paper default |
| `accum_steps` (T) | 8 | Pool size = 8 × bs |
| `num_proj` | 1024 | Paper default |
| Multi-crop | V_g=2 @ **128** (CNN: V_l=0; ViT: V_l=6 @ 56) | **3× faster than paper's 224**, run sanity check at 224 |
| Memory format | `channels_last` | ~1.3× speedup |
| `torch.compile` | mode=`default` | ~1.4× speedup |

### Note on BatchNorm — why we default to LN-only backbones

The pooled 2-pass mechanism has shown instability with BatchNorm in our
experiments so far. We have not yet found a fix that fully resolves it, so
we default to LN-only architectures (ConvNeXtV2, ViTs) for now. BN-based
backbones (resnet18/34/50) may still work and are worth attempting, but
results should be checked for the variance-drift failure mode described below.

**The failure mode** (diagnosed in `idea_experimental/bn_diagnose.py`): the `inject_bn_stats` trick that makes Pass 1 and Pass 2 use the same normalization injects *constants* (Pass-1 batch stats captured under `no_grad`) into Pass 2's BN. That removes BN-train-mode's implicit gradient through batch statistics — the gradient that normally penalizes the encoder for changing the overall scale of its features. With that constraint gone, the encoder's pre-BN feature variance drifts upward monotonically:

```
epoch:    init   0    4    8    13
rvar_max: 117  120  357 1000 1340     (~12× growth, resnet18 / cifar10 / bs64 / accum=8)
```

Eventually the run hits a numerical instability and collapses (loss jumps from ~1.7 → ~3.5 around epoch 10–12 in the 20-epoch sweep). The wild val_acc oscillation that motivated this whole investigation was the early symptom of that drift, not an EMA staleness issue.

**Eval-side patches don't fix it.** We tried recalibrating BN before eval, evaluating with batch stats, freezing BN, and bumping BN momentum (all in `idea_experimental/continuation_fix.py`). They eliminate the *visible* dip in val_acc but plateau ~2 points below the standard baseline because the encoder is still drifting under the hood, and the underlying training trajectory still collapses around the same epoch.

**The fix is to drop BN entirely.** LayerNorm normalizes per-sample with no running stats and no train/eval distinction, so the 2-pass mechanism is train/eval consistent automatically. The existing `_BNStatCapture` and `inject_bn_stats` machinery in `experiments/pretrain/train_loops.py` becomes a harmless no-op on LN models (the loops only register on BN modules), so **no code changes to the training loop are required** — only the backbone choice.

We verified this empirically (`idea_experimental/layernorm_experiment.py`): convnextv2_nano continuation, 20 epochs, BS=64, accum=8 → pooled finishes at **0.544 val_acc vs standard 0.501 (+4.3 points)**, no dip, no collapse, encoder feature std stays in [0.79, 0.92] throughout. This is the "pooled wins" result the synthetic experiments predict — it shows up as soon as BN isn't masking it.

**GroupNorm** would also avoid the pathology (it normalizes per-sample-per-group, no running stats) and is a valid fallback for any backbone that lacks an LN variant. We default to LayerNorm because the ConvNeXtV2 family provides a clean 5-rung scale ladder of LN-only CNNs (atto 3.4M → femto 4.8M → pico 8.6M → nano 15M, plus convnext_tiny at 27.8M), exactly covering the "small fast CNN" niche resnet18 used to occupy. If a future architecture comparison needs a backbone outside the ConvNeXt family, the recommended path is to either pick a LN-native variant or wrap it in `nn.GroupNorm` substitution before training.

---

### NOTE: GPU saturation curve — why pooled is "almost free" at small batch sizes

This is the **key cost insight** for the paper. See `experiments/compute_cost/`
for the benchmark scripts, raw data tables, and plots.

**Empirical saturation curve** — `convnextv2_nano` forward at 128², bf16, eval mode (RTX 4090):

| n_imgs | time | ms/img | regime |
|---:|---:|---:|---|
| 1 | 3.49 ms | 3.49 | flat (under-utilized) |
| 8 | 3.43 ms | 0.428 | flat |
| 32 | 3.57 ms | 0.112 | flat |
| **48** | **4.92 ms** | **0.103** | **knee** |
| 64 | 5.99 ms | 0.094 | near-peak |
| 96 | 9.02 ms | 0.094 | near-peak |
| **128** | **13.53 ms** | **0.106** | **linear (compute-bound)** |
| 256 | 31.75 ms | 0.124 | linear |
| 512 | 62.46 ms | 0.122 | linear |
| 1024 | 124.32 ms | 0.121 | linear |

**Two regimes** (note: knee is ~3× earlier than the historical resnet18 curve):
- **n ≤ 48**: time is essentially constant. The GPU is under-utilized; adding more
  samples is free. convnextv2_nano at 128² has more work per image than resnet18
  did, so the GPU saturates at ~half the batch size.
- **n ≥ 128**: time scales linearly with n. The GPU is compute-bound; each extra
  sample adds proportional cost. This is exactly why we cap the BS sweep at 64.

Raw CSV: `experiments/compute_cost/results/saturation_NVIDIA_GeForce_RTX_4090_convnextv2_nano_128.csv`.

### Per-step cost: pool_nograd vs std vs full-gradient oracle

Per-step timing INCLUDES sample_batch + gpu_aug. RTX 4090, 128², T=8, V_g=2,
no local crops. `fullgrad` is a forward+backward on T·BS samples — the
"oracle" cost of getting the same regularizer information as `pool_nograd` if
you also wanted gradients on every sample. Raw CSV at
`experiments/compute_cost/results/pooled_overhead_NVIDIA_GeForce_RTX_4090_convnextv2_nano_128_T8.csv`.

**convnextv2_nano (V_g=2, no locals)**:

| BS | std | pool_nograd | fullgrad (T·BS) | pool/std | fullgrad/std | **pool/fullgrad** | std GB | pool GB | fullgrad GB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | 13.8 ms | **24.3 ms** | 61.0 ms (64) | **1.76×** | 4.43× | **0.40×** | 1.27 | 1.36 | 3.85 |
| 16 | 17.4 ms | 42.3 ms | 123.4 ms (128) | 2.44× | 7.11× | **0.34×** | 1.63 | 1.86 | 6.82 |
| 32 | 25.8 ms | 90.3 ms | 253.2 ms (256) | 3.50× | 9.81× | **0.36×** | 2.37 | 2.86 | 12.68 |
| **48** | 44.9 ms | 137.7 ms | 379.6 ms (384) | 3.07× | 8.46× | **0.36×** | 3.11 | 3.87 | 18.52 |
| **64** | 61.0 ms | **185.3 ms** | **OOM** (512) | 3.04× | — | — | 3.85 | 4.87 | **>24** |

**The headline finding**: pooled training adds only **1.76× wall clock** at BS=8
where the bias floor matters most, and stabilizes around 3× std past saturation.
Crucially, the `fullgrad` baseline that you'd otherwise need to get the same
regularizer-sample count **OOMs at BS=64 on a 24 GB GPU**, while pool fits
comfortably with under 5 GB. This gives the headline cost story:

> **Pool delivers the regularizer information of an equivalent full-gradient
> step at ~0.36× its wall clock and ~3–4× less peak memory; at the upper end
> of our BS sweep the full-gradient baseline does not even fit, while pool
> runs on every cell.**

This matches theory: pool_nograd ≈ T·BS forwards + BS forward+backward, while
fullgrad ≈ T·BS forward+backward. Theory predicts pool/fullgrad ≈
`(T+2)/(3T)` = 0.42 at T=8, ignoring saturation effects. We measure 0.34–0.40
across the sweep, *better* than the prediction at larger BS because std-on-BS
gets the GPU under-utilization discount and fullgrad-on-T·BS doesn't.

**Saturation also bounds the BS sweep itself**: per-pool-epoch wall clock on
cifar100 is essentially constant at ~135 s across BS={8..64} because pool's
cost is dominated by the (T-1)·BS no-grad pass, which is the same total samples
regardless of how you split it across optimizer steps. We don't measure BS=128
because:
1. fullgrad doesn't fit there at all on 24 GB,
2. pool/std and per-epoch cost both plateau, and
3. it would add ~30% Phase 1 wall clock without adding new accuracy or cost
   information.

**ViT-tiny (V_g=2 + V_l=6 multi-crop)** — historical resnet18-era measurements
kept here because the qualitative shape of the ViT curve hasn't changed; ViT
backbones were always LayerNorm-based. Re-measurement on the new nano default
of `pooled_overhead.py --backbone tiny --multicrop --resolution 128` is cheap
and pending:

| BS | std | pool_nograd | pool / std | std GB | pool GB |
|---:|---:|---:|---:|---:|---:|
| **8** | 16.8 ms | **23.2 ms** | **1.38×** | 0.91 | 0.95 |
| **16** | 17.2 ms | **23.5 ms** | **1.37×** | 1.08 | 1.16 |
| 32 | 17.4 ms | 30.5 ms | 1.75× | 1.42 | 1.58 |

ViT-tiny multi-crop stays cheap longer than nano because each BS sample expands
to 8 view-images (V_g=2 + V_l=6), keeping the GPU in its under-utilized regime
through BS=16 even at 128² resolution.

This is **exactly the regime where the bias floor matters most**: small batches
have high finite-sample bias for distributional regularizers, and that's where
pooling helps most. Pooling is cheapest precisely where it's most useful, and
the equivalent fullgrad baseline is most infeasible.

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

(Earlier drafts also cited BN-consistency complexity as a reason — that point
is now moot. With LN-only backbones, splitting Pass 1 into chunks has no
normalization-consistency cost; the only remaining objections are kernel-launch
overhead and the lack of need for memory savings.)

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
- Output: `runs/base/{dataset}_convnextv2_nano_sigreg_bs8_seed42/final.pt`

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

For each of `convnextv2_atto`, `convnextv2_nano`, `convnext_tiny`, `vit_tiny`,
train base on `cifar100` / `flowers102` / `pets`, then continue with 2 methods
(`sigreg` baseline + `w1_pooled` headline) at the headline cont bs.

This is a 4-rung scale ladder from 3.4M params (atto) → 27.8M (convnext_tiny),
plus a ViT for architecture-class breadth. **All four backbones are LayerNorm-only**
(see "Note on BatchNorm" in the default training config section); no BN
architectures are included in the sweep, by design.

- 4 archs × 3 ds × (1 base + 2 methods × 3 seeds) = 12 base + 72 cont
- − 18 reused (`convnextv2_nano` rows already done in Phase 1 since it is the
  default backbone) = **66 runs**

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
    --continuation runs/base/cifar100_convnextv2_nano_sigreg_bs8_seed42/final.pt \
    --encoder-scale convnextv2_nano \
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
