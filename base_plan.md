# Experiment Plan: Pooled Distributional Regularization for JEPAs

## Paper Title (working)
"Pooled Sorting Breaks the Batch Size Barrier in Sliced Distributional Regularization"

## Core Claims
1. Sliced distributional regularizers (W1, W2, EP) have irreducible positive bias under the null that scales as O(1/√n) or O(1/n) with sample size n.
2. Standard gradient accumulation (averaging independent losses) does NOT reduce this bias. Pooling samples before computing the test statistic DOES.
3. This applies to ALL test statistics: W1, W2, EP, CvM, KS. For EP, pooling means accumulating the ECF before squaring. For W1/W2, pooling means accumulating embeddings before sorting.
4. LeJEPA's dismissal of CDF-based tests (sorting is non-parallel, non-differentiable) is overstated. All-gather + local sort works, and gradients flow through sorted values natively.
5. W1 with pooled sorting is faster than EP and produces empirically better regularization.

---

## Phase 0: Setup

### 0.1 Data setup
- [ ] CIFAR-100: torchvision
- [ ] ImageNet-100: HuggingFace (`clane9/imagenet-100`) or symlink (`github.com/ellisbrown/IN100`)
- [ ] Flowers-102: torchvision (optional, tiny dataset stress test)
- [ ] LeWorldModel datasets: HuggingFace (`quentinll/lewm-tworooms`, `lewm-reacher`, `lewm-pusht`, `lewm-cube`)
- [ ] LeWorldModel checkpoints: HuggingFace (same repos)

### 0.2 Extract hyperparameters from references
- [ ] LeJEPA: configs for CIFAR-100, ImageNet (lr, wd, epochs, batch size, lambda, num_slices, augmentation, image resize for small datasets, architecture per dataset from Table 5)
- [ ] LeWorldModel: configs for each of 4 tasks (lr, wd, epochs/steps, batch size, lambda, encoder/predictor architecture, evaluation protocol, training time per task)
- [ ] Record in shared config file

### 0.3 Logging and evaluation
- [ ] wandb or tensorboard logging
- [ ] Linear probe evaluation (frozen backbone, train linear head)
- [ ] LeWorldModel evaluation protocol (CEM planning, success rate)

---

## Phase 1: Synthetic Validation (MAIN PAPER)

### Experiment 1.1: Full batch baseline — which loss is best?
- **Goal:** Establish W1/W2 vs SIGReg when batch size is not a constraint
- **Setup:** Data in R^d, linear projection to R^N (N > d), MLP maps back to R^d, target N(0, I_d)
- **Latent dim d:** 4
- **Projected dims N:** 8, 32
- **Batch size:** 256 (large, minimal finite-sample bias)
- **Methods:** SIGReg (EP), Sliced W1, Sliced W2
- **No accumulation, no pooling**
- **Metric:** Final W1 distance to N(0, I_d) after convergence
- **Output:** Table + convergence curves

### Experiment 1.2: Accumulation — standard vs pooled
- **Goal:** Show pooled breaks the bias floor, standard accumulation does not
- **Setup:** Same as 1.1, small micro-batch
- **Micro-batch size:** 8
- **Accumulation steps T:** 1, 2, 4, 8, 16, 32
- **Methods at each T:**
  - (a) SIGReg, standard grad accum (average T independent EP losses)
  - (b) SIGReg, pooled (accumulate ECF across T steps before squaring, detach previous steps)
  - (c) Sliced W1, standard grad accum (average T independent W1 losses)
  - (d) Sliced W1, pooled sort (accumulate embeddings across T steps, sort once, detach previous steps)
  - (e) Sliced W2, standard grad accum
  - (f) Sliced W2, pooled sort
- **Projected dims:** 8, 32
- **Metric:** Final W1 distance to N(0, I_d) after convergence
- **Output:** Table + plot of converged quality vs T for all 6 methods
- **Expected:** (a,c,e) plateau, (b,d,f) keep improving. All pooled methods improve. W1/W2 pooled best overall.

### Experiment 1.3: DDP simulation — local shards vs pooled
- **Goal:** Show that local-shard-only regularization degrades, and pooling fixes it, simulating multi-GPU without actually needing multiple GPUs
- **Setup:** Same synthetic task
- **Total batch size:** 256 (fixed)
- **Simulate D devices:** D = 1, 2, 4, 8, 16 (split batch into D shards of 256/D each)
- **For each simulated D:**
  - (a) SIGReg on local shard only (each shard computes EP independently, average losses). This simulates DDP WITHOUT all-reduce of ECF.
  - (b) SIGReg with all-reduce of ECF (compute global ECF from all shards, then square). This is what LeJEPA actually does.
  - (c) Sliced W1 on local shard only (sort 256/D samples per shard, average losses)
  - (d) Sliced W1 with all-gather + local sort (gather all 256 samples, sort on each shard)
  - (e) Sliced W1 local shard + pooled accumulation T=D (accumulate D steps of 256/D samples locally, sort once). Same effective sample size as (d) but no communication.
- **Metric:** Final W1 distance to N(0, I_d)
- **Output:** Table showing (a,c) degrade with D, (b,d,e) don't. Shows pooled accumulation matches all-gather without communication.
- **Note:** This is a single-GPU experiment that SIMULATES DDP by splitting the batch. No actual multi-GPU needed.

### Experiment 1.4: Timing comparison
- **Goal:** W1/W2 are faster than SIGReg per sample
- **Setup:** Wall-clock time for forward pass only
- **Sweep:** Batch B = 128 to 32768, Dimension D = 2 to 256
- **Methods:** SIGReg (EP, 17 knots, 1024 slices) vs Sliced W1 (1024 slices) vs Sliced W2 (1024 slices)
- **Output:** Heatmap of time ratios, per-sample cost plot

---

## Phase 2: Pretraining — Batch Size Sweep Across Datasets (MAIN PAPER)

### Common design for all datasets
- **Accumulation steps T:** 8 (fixed)
- **Methods at each batch size:**
  - (a) SIGReg, standard grad accum (LeJEPA baseline — average T independent EP losses)
  - (b) SIGReg, pooled (accumulate ECF across T steps before squaring)
  - (c) Sliced W1, standard grad accum (average T independent W1 losses)
  - (d) Sliced W1, pooled sort (accumulate embeddings, sort once)
- **Evaluation:** Frozen backbone linear probe, top-1 accuracy
- **Lambda:** LeJEPA default per dataset

### Experiment 2.1: CIFAR-100
- **Dataset:** CIFAR-100 (50K train, 10K test, 100 classes)
- **Image size:** Check LeJEPA configs
- **Architecture:** Match LeJEPA Table 5 (e.g., ResNet-18 11M)
- **Training:** Match LeJEPA defaults
- **Micro-batch sizes:** 32, 64, 128, 256
- **Total runs:** 4 batch sizes × 4 methods = 16 runs
- **Time estimate:** ~1-2 hours per run
- **Output:** One panel of Figure 4

### Experiment 2.2: ImageNet-100
- **Dataset:** ImageNet-100 (130K train, 100 classes)
- **Image size:** 224x224
- **Architecture:** ViT-Tiny (5.7M params)
- **Training:** Match LeJEPA ImageNet defaults (100 epochs)
- **Micro-batch sizes:** 64, 128, 256
- **Total runs:** 3 batch sizes × 4 methods = 12 runs
- **Time estimate:** ~2-4 hours per run
- **Output:** One panel of Figure 4

### Experiment 2.3: Flowers-102 (optional, for breadth)
- **Dataset:** Flowers-102 (1020 train, 102 classes)
- **Architecture:** Match LeJEPA Table 5
- **Micro-batch sizes:** 16, 32, 64, 128
- **Total runs:** 4 batch sizes × 4 methods = 16 runs
- **Time estimate:** Very fast (<1 hour per run)
- **Output:** One panel of Figure 4
- **Note:** Extremely small dataset, batch size sensitivity should be extreme

---

## Phase 3: LeWorldModel Application (MAIN PAPER)

### Setup
- **Code:** Clone LeWorldModel repo, integrate pooled W1 and pooled EP losses
- **Tasks:** Two-Room, Reacher, Push-T, OGBench-Cube
- **Extract from code/paper:**
  - [ ] Batch size per task
  - [ ] Training epochs/steps
  - [ ] lr, optimizer, scheduler
  - [ ] Lambda for SIGReg
  - [ ] Encoder + predictor architecture
  - [ ] Evaluation protocol
  - [ ] Training time per task on single GPU

### Experiment 3.1: Train from scratch
- **Goal:** Show pooled W1 improves world model from scratch
- **For each of 4 tasks:**
  - (a) SIGReg standard (reproduce their baseline)
  - (b) Sliced W1 pooled sort (ours)
- **Train full duration using their configs**
- **Total runs:** 4 tasks × 2 methods = 8 runs
- **Output:** Table of planning success rate

### Experiment 3.2: Continue training from checkpoints
- **Goal:** Show improvement even on a trained model
- **For each of 4 tasks:**
  - (a) Their checkpoint, no additional training (their published numbers)
  - (b) Continue K epochs with SIGReg (control)
  - (c) Continue K epochs with Sliced W1 pooled
- **K:** 10-20% of original training duration
- **Total runs:** 4 tasks × 2 continued methods = 8 runs
- **Output:** Table: (a) vs (b) vs (c)

---

## Phase 4: Supplementary Experiments

### Experiment 4.1: Accumulation depth sweep
- **Dataset:** CIFAR-100 or ImageNet-100 (pick one)
- **Micro-batch size:** 64
- **T:** 1, 2, 4, 8, 16
- **Methods:** W1 pooled, SIGReg pooled
- **Baselines:** SIGReg standard T=1, W1 standard T=16
- **Total runs:** 12
- **Output:** Accuracy vs T plot

### Experiment 4.2: Lambda sensitivity
- **Dataset:** CIFAR-100 or ImageNet-100 (pick one)
- **Micro-batch size:** 64, T=8
- **Methods:** SIGReg standard vs W1 pooled
- **Lambda:** 0.5x, 1x, 2x, 4x of LeJEPA default
- **Total runs:** 8-10
- **Output:** Accuracy vs lambda plot

### Experiment 4.3: Full synthetic tables
- All combinations from Experiments 1.1 + 1.2, formatted for supplementary

### Experiment 4.4: Training curves
- Save loss/accuracy curves from Phase 2 and 3, show convergence differences

### Experiment 4.5: DDP simulation on real data (optional)
- Same idea as Experiment 1.3 but on CIFAR-100
- Simulate D = 1, 2, 4, 8 devices by splitting batch
- Show local-only W1 pooled with T=D matches global-batch quality

### Experiment 4.6: No-grad forward pass for extra CDF samples
- **Goal:** Test whether running extra forward passes WITHOUT gradient (no activation storage) to gather more CDF samples improves regularization beyond what gradient accumulation provides
- **Motivation:** Forward pass without gradient costs FLOPs but NOT memory (no activation graph stored). This decouples CDF resolution from both batch size AND gradient accumulation depth.
- **Setup on synthetic task first, then CIFAR-100 if promising**
- **Micro-batch with gradient:** M = 64 (fixed, this is what fits in memory)
- **Additional no-grad forward passes:** E = 0, 1, 2, 4, 8 (extra batches of 64, no grad)
- **Total CDF samples:** M + E×M = 64, 128, 192, 320, 576
- **Procedure per update step:**
  1. Run E batches of M samples each with torch.no_grad() — store only the output embeddings (detached). No activation graph stored, so memory cost is just the embeddings. These are separate forward calls but cheap (no autograd overhead, no activation storage).
  2. Run 1 batch of M samples with gradient — activations stored for backward
  3. Concatenate all (E+1)×M embeddings
  4. Compute pooled W1 loss — gradient flows only through the M live embeddings
  5. Compute prediction loss on the M live samples only
  6. Backward, update parameters
- **Total forward passes per update:** E+1 (cannot be fused into one call because PyTorch stores activations for entire batch if any sample requires gradient)
- **Peak memory:** Same as batch size M (only one grad-enabled forward pass at a time)
- **Compare against:**
  - Standard grad accum with T=E+1 steps (same total forward passes, but also T backward passes for prediction loss)
  - Pooled grad accum with T=E+1 steps (same CDF resolution, but more backward passes)
- **Key tradeoff to measure:**
  - No-grad trick: fewer backward passes (only 1), but prediction loss uses only M samples
  - Grad accum: more backward passes (T), but prediction loss uses T×M samples
  - Is the CDF improvement from no-grad extra samples worth the prediction gradient quality loss?
- **Output:** Table comparing final quality, wall-clock time, peak memory for each approach
- **Note:** Can be combined with grad accum — do T accumulation steps for prediction loss, AND within each step run extra no-grad forward passes for CDF. This gives T×M prediction gradients AND (T×(E+1)×M) CDF samples. Maximum quality but maximum FLOPs.

---

## Main Paper Figures

### Figure 1: Method illustration
- Left: standard grad accum (T separate coarse CDFs/ECFs, averaged loss)
- Middle: pooled EP (accumulate ECF across T steps, square once)
- Right: pooled W1 (accumulate embeddings across T steps, sort once)
- Show gradient flows only through current batch in all cases
- Optionally: small panel showing no-grad variant where extra forward passes contribute to CDF without activation storage

### Figure 2: Synthetic — bias floor + DDP simulation
- Panel A (from Exp 1.2): converged quality vs T, standard (flat) vs pooled (improving), for W1, W2, EP
- Panel B (from Exp 1.3): converged quality vs simulated D devices, showing local-shard degrades, all-reduce/all-gather/pooled-accum don't

### Figure 3: Timing
- From Experiment 1.4
- W1/W2 faster than SIGReg

### Figure 4: Batch size sweep across datasets (KEY FIGURE)
- Multi-panel: CIFAR-100, ImageNet-100, (optionally Flowers-102)
- Each panel: linear probe accuracy vs micro-batch size
- Curves: SIGReg standard, SIGReg pooled, W1 standard, W1 pooled

## Main Paper Tables

### Table 1: Theoretical bias under null
| Test | Bias (batch n) | Bias (averaged T×n) | Bias (pooled Tn) |
|------|---------------|--------------------|--------------------|
| W1   | C/√n          | C/√n               | C/√(Tn)           |
| W2   | C/√n          | C/√n               | C/√(Tn)           |
| KS   | c/√n          | c/√n               | c/√(Tn)           |
| CvM  | 1/(6n)        | 1/(6n)             | 1/(6Tn)           |
| EP   | C_EP/n        | C_EP/n             | C_EP/(Tn)         |

### Table 2: LeWorldModel results
| Task | Published | SIGReg scratch | Ours scratch | SIGReg cont. | Ours cont. |
|------|-----------|---------------|-------------|-------------|-----------|

---

## DDP Discussion Strategy (in paper text, not experiment)

Do NOT claim DDP is a problem we uniquely solve. Instead:
- Acknowledge EP's all-reduce trick is efficient and gives global-batch ECF
- Show that EP can ALSO benefit from accumulation pooling (accumulate ECF before squaring)
- Show W1 works with all-gather + local sort OR pure local accumulation (no communication)
- Argue W1 is faster per sample and gives better regularization
- Note that LeJEPA's dismissal of CDF-based tests was based on concerns (global sorting, non-differentiability) that don't apply to W1 with all-gather or local accumulation
- Key sentence: "All-gather followed by local sorting is a standard operation that addresses the distributed sorting concern. Gradients through sorted values are natively supported in PyTorch, addressing the differentiability concern."

---

## Priority Order

1. **Phase 0** — setup, data, hyperparameters
2. **Exp 1.1 + 1.2** — synthetic baseline + accumulation comparison (fast)
3. **Exp 1.3** — DDP simulation (fast, single GPU)
4. **Exp 2.1** — CIFAR-100 batch sweep (~16-32 hours)
5. **Exp 2.2** — ImageNet-100 batch sweep (~24-48 hours)
6. **Exp 3.1** — LeWorldModel from scratch (time TBD)
7. **Exp 3.2** — LeWorldModel continuation
8. **Exp 1.4** — timing (fast)
9. **Exp 2.3** — Flowers-102 (fast, optional)
10. **Exp 4.1** — depth sweep (supplementary)
11. **Exp 4.2** — lambda sweep (supplementary)

---

## Open Questions To Resolve Early

- [ ] What image size does LeJEPA use for CIFAR-100?
- [ ] What architecture for CIFAR-100 from Table 5?
- [ ] LeWorldModel batch size per task? (critical)
- [ ] LeWorldModel: multi-crop or single frame?
- [ ] LeWorldModel training time per task?
- [ ] ImageNet-100 from HuggingFace sufficient or need ImageNet-1K?
- [ ] W1 vs W2: run both in synthetic, pick winner for real experiments
- [ ] LeJEPA default lambda per dataset?
- [ ] LeJEPA norm type per architecture (layer norm vs batch norm)?
- [ ] Does LeJEPA code all-reduce the ECF or compute EP on local shard? (verify our understanding)