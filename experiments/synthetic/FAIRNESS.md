# Fairness in Non-Full-Batch Synthetic Experiments

## The problem

The four synthetic experiments (`exp1_1` through `exp1_5`) compare regularization
losses (W1, W2, SIGReg) under different operating regimes. `exp1_1` uses
full-batch training and is unambiguous: every step processes the entire
dataset. The other three (`exp1_2`, `exp1_3`, `exp1_5`) draw mini-batches, so
"how do we make a fair comparison across sweep points?" becomes a real
question.

The original implementations ran every config for a fixed number of optimizer
steps (`--steps 10000`). Under that convention, configs with high accumulation
or many simulated devices saw vastly more raw data than the baseline:

| Config                                              | Samples/step | Total samples (10000 steps) | "Epochs" of K=16384 |
|-----------------------------------------------------|--------------|------------------------------|----------------------|
| exp1_2 T=1, BS=32                                   | 32           | 320K                         | 20                   |
| exp1_2 T=32, BS=32                                  | 1024         | 10.24M                       | 625                  |
| exp1_3 D=16, BS_per_device=32, T_accum=8 (pooled)   | 4096         | 40.96M                       | **2500**             |
| exp1_5 T_cur=8, T_fifo=8, BS=32                     | 2304         | 23.04M                       | 1406                 |

Any reported "T helps" or "D hurts" effect under this protocol is contaminated
by raw data exposure rather than reflecting the property the experiment is
supposed to measure. We need an epoch-based protocol.

## Two definitions of fairness

There are two reasonable definitions of fairness across configs that consume
data at different rates.

### Option A — grad-compute-matched

The unit of consumption is the **gradient batch**. An epoch is
`K // grad_BS` optimizer steps, where `grad_BS` is the batch size that carries
gradient on each opt step. No-grad context (e.g., the `(T-1)*BS` samples in
pooled accumulation, the FIFO contents, or the per-device CDF window) is
treated as "free context": it improves the regularizer but does not count
toward data consumption.

**The story this tells:** *"For a fixed number of gradient updates with a
fixed gradient batch size, does pooling / global communication / a longer FIFO
help?"*

This matches how a practitioner thinks about cost on a real cluster: backward
passes and optimizer steps dominate wall-clock and memory; the no-grad pass is
opportunistic reuse of cheap forward compute. It is also what the production
training loop in `experiments/pretrain/train_loops.py` does — `num_steps =
len(train_dataset) // cfg.batch_size`, where `batch_size` is the *grad* batch.

### Option B — total-data-matched

The unit of consumption is **any sample seen by the model**, with or without
gradient. An epoch is `K // total_BS` optimizer steps, where `total_BS` is the
*total* number of samples (grad + no-grad) consumed per opt step.

**The story this tells:** *"For a fixed total number of samples consumed, does
pooling them into a single CDF beat averaging T independent loss estimates?"*

This is the right question when the **information content** of data is what
the experiment is testing — i.e., when both methods are given the same data
budget and the only difference is how that data is aggregated into the loss.

## Why each experiment uses what it uses

### `exp1_2` (standard vs pooled vs big-batch accumulation) → **Option B**

`train_one_standard(T)`, `train_one_pooled(T)`, and `train_one_bigbatch(T)`
all consume exactly `T*BS` samples per opt step:

- Standard: T sub-batches of size BS, each producing an independent loss; the
  T losses are averaged and backpropagated. T forward+backward passes per opt
  step. Memory: BS activations (sequential).
- Pooled: (T-1) no-grad forward passes + 1 grad forward pass, all `T*BS`
  embeddings concatenated into one CDF, single backward pass through the
  final BS-slice. Memory: BS activations + `T*BS` detached embeddings (cheap).
- Big-batch (oracle): single forward through `T*BS` samples with grad, one
  loss on the union, single backward through all `T*BS`. Memory:
  `T*BS` activations.

The experiment exists to answer two coupled questions: *"Given T·BS samples,
(a) is averaging T independent losses worse than pooling them into one CDF?
and (b) does pooled — which only backprops through BS samples — actually
match the big-batch oracle that backprops through all T·BS?"* Counting
matched total data is the only honest way to ask either question. Counting
only gradient batches would penalize standard (which uses T grad batches per
opt step) against pooled and big-batch (which use 1) on a metric the
experiment is not trying to test.

The big-batch arm matters because without it, "pooled beats standard" is
ambiguous between "pooling is better" and "pooled has its own pathologies
that happen to be smaller than standard's pathologies." With big-batch as a
reference, pooled ≈ big-batch becomes a positive claim about pooling, and
standard < big-batch becomes a positive claim about averaging-T-losses being
fundamentally a different (worse) loss function.

**Concretely:** epoch = `K // (T*BS)` opt steps for all three modes with the
same T. Different T values do different numbers of opt steps per epoch but
see the same amount of data per epoch. At T=1 all three modes collapse to a
single forward+backward through BS samples and produce identical results, so
T=1 is trained once (as `standard`) and the result is copied to the
`pooled` and `bigbatch` cells.

### `exp1_3` (DDP simulation) → **Option A**

This experiment models a real DDP setup: each device contributes a fixed
shard of size BS_per_device, the effective batch grows linearly with the
device count D, and the comparison is between communication strategies (local
shards vs all-reduce vs all-gather vs pooled). The cost model that matters in
practice is **gradient passes and inter-device communication**; the no-grad
pooled context is a deployment optimization, not data consumption.

Counting no-grad samples as consumption would penalize pooled modes for using
the technique they exist to test, and would give D=16 pooled configs roughly
T_accum-times fewer opt steps per epoch than D=16 non-pooled configs. The
DDP-degradation story would no longer be visible because the configs would
not be comparable on a per-step basis.

**Concretely:** epoch = `K // (D * BS_per_device)` opt steps for all 8 modes.
Pooled modes additionally do `(T_accum - 1)` no-grad forward passes per
device per opt step, but those samples do not count toward the epoch budget.

A consequence of this choice: at fixed epoch count, D=16 does 16× fewer
optimizer steps than D=1 — exactly because its effective batch is 16× larger.
This matches the realistic DDP scenario where bigger effective batches mean
fewer updates per epoch, and the question of whether that hurts the loss
floor is itself part of what the experiment measures.

### `exp1_5` (FIFO buffer) → **Option A**

This experiment varies the *no-grad context size* (`T_cur` and `T_fifo`)
while holding the gradient batch fixed at BS. Counting no-grad samples as
consumption would defeat the experiment outright — it would force every
`(T_cur, T_fifo)` cell to do a different number of opt steps per epoch and
conflate the measurement we are trying to make.

**Concretely:** epoch = `K // BS` opt steps for every `(T_cur, T_fifo)` cell.
Every cell does the same number of optimizer updates per epoch; the only
thing that varies is how much detached context the regularizer can see when
computing each gradient.

## Why not Option A everywhere or Option B everywhere?

**Option A everywhere** would make `exp1_2` dishonest. Standard with T=8
would do 8× as many gradient updates per epoch as pooled and big-batch with
T=8 (because pooled and big-batch use 1 grad batch per opt step and standard
uses 8). The comparison would no longer be "T·BS samples → which loss
aggregation is better?" but rather "given a fixed opt-step budget, who
wins?" — and standard would automatically win on opt-step count, masking the
data-aggregation question the experiment is designed to isolate.

**Option B everywhere** would make `exp1_3` and `exp1_5` dishonest in the
opposite direction. It would treat the cheap no-grad forward passes that
pooling and FIFO rely on as if they were full data consumption, force pooled
configs to do a fraction of the optimizer updates of non-pooled configs, and
fail to reflect any realistic deployment cost model.

The right answer is to pick the definition that matches **what each
experiment is trying to isolate**, not to enforce a single global rule.

## Sampling: without replacement, per epoch

The original implementations sampled grad batches with replacement
(`torch.randint`). The new epoch-based implementations use a fresh permutation
per epoch (`torch.randperm`) and yield non-overlapping chunks via the helper
`src.sliced_gauss_reg.epoch_iter`. After one epoch every data point has been
seen exactly once (modulo the dropped final partial chunk when
`K % batch_size != 0`). This is what makes the epoch concept well-defined.

No-grad batches in pooled and FIFO configs continue to be drawn with
replacement via per-device generators — they are *context*, not the epoch
driver, so resampling them at each step is the correct behavior. The
gradient-carrying samples are the ones that have to be deduplicated within
an epoch.

## Gradient dilution compensation in pooled losses

Pooled losses compute the regularizer on `n_total = T·BS` embeddings but
let only `n_real = BS` of them carry gradient (the rest are detached
context). The "right" compensation depends on whether per-sample
gradient scales with `n_total`, and **W1/W2 and SIGReg behave differently
in this respect** — so they get different treatment.

### W1/W2: compensate by `n_total / n_real ≈ T`

W1's loss is `mean over n_total samples` of `|sorted - quantile|`, so
per-sample gradient is `O(1/n_total)`. With only `n_real = BS` samples
carrying gradient, the uncompensated total grad is `BS · (1/n_total) =
1/T` — diluted relative to standard accumulation (which has total grad
`1`) and to the big-batch oracle (also `1`). Multiplying the loss by
`n_total / n_real ≈ T` restores parity:

| W1/W2 mode      | Per-sample grad | Total grad |
|-----------------|-----------------|------------|
| Standard (T sub-batches /T) | `1/BS` per sub | `1` |
| Pooled (compensated × T)    | `1/BS`          | `1` |
| Big-batch                   | `1/(T·BS)`      | `1` |

All three arms match. Pooled is a true memory-efficient drop-in for
big-batch *and* a drop-in for standard accumulation under the same
lambda. This compensation lives in `PooledSlicedLoss.grad_step` (W1/W2
branch), `pooled_loss` (W1/W2 branch), and the manual `pooled_w1` /
`pooled_global_w1` paths in `exp1_3`.

### SIGReg: NO compensation

SIGReg's test statistic is `||residual||² · n`, where the `*n` is the
Epps-Pulley normalization that makes the loss *value* asymptotically
non-degenerate (~chi-squared, `O(1)` regardless of `n`). But that same
`*n` cancels the `1/n` from `c_bar = (1/n) Σ cos(t·xᵢ)` in the gradient
chain, leaving per-sample gradient `O(1)` — **independent of `n_total`**.

The consequence:

| SIGReg mode | Per-sample grad | Grad samples | Total grad |
|---|---|---|---|
| Standard (T sub-batches /T) | `O(1)/T` (after avg) | `T·BS` | `BS` |
| Pooled (uncompensated)      | `O(1)`               | `BS`   | `BS` |
| Pooled (× T compensation)   | `O(T)`               | `BS`   | `T·BS` |
| Big-batch                   | `O(1)`               | `T·BS` | `T·BS` |

**Standard and pooled-uncompensated already match at `BS`.** Compensating
pooled by T would push it to `T·BS`, breaking lambda parity with
standard (the way W1/W2 compensation does *not* break parity, because
W1/W2 ends up at `1` everywhere).

**You can match standard ↔ pooled OR pooled ↔ big-batch, but not both.**
There is no choice of compensation that aligns all three SIGReg modes,
because standard SIGReg literally throws away `T-1` worth of gradient
signal by averaging `T` independent test statistics — and the discarded
signal cannot be recovered by scaling. Standard SIGReg and big-batch
SIGReg are genuinely different loss functions.

We choose **standard ↔ pooled equivalence** (no compensation for
SIGReg). This makes pooled SIGReg a drop-in replacement for standard
SIGReg under the same lambda — which matters for the production pretrain
loop, where lambda is tuned against the non-pooled default. The cost is
that big-batch SIGReg in the synthetic exp1_2 is intrinsically `T×`
stronger than the other two SIGReg arms; that's an honest reflection of
SIGReg's test-statistic formulation, not a bug.

### Why W1/W2 don't have this problem

For W1/W2, the per-sample gradient `O(1/n)` cancels the sample count `n`
in the total, so all three arms (standard, pooled, big-batch) land at
the same total grad `1` regardless of how the samples are partitioned.
The T compensation makes pooled W1/W2 mathematically equivalent to both
standard and big-batch.

For SIGReg, the per-sample gradient is `O(1)` (no `n` cancellation), so
total grad scales linearly with the number of grad-carrying samples.
Standard has `T·BS` grad samples each weakened by `T`-averaging
(effective `BS`); pooled has `BS` grad samples at full strength
(`BS`); big-batch has `T·BS` at full strength (`T·BS`). These cannot
all be made equal by a single scalar compensation.

## Defaults and tuning

| Experiment | Default `--epochs` | Min opt steps (worst case)              | Max opt steps                    |
|------------|--------------------|------------------------------------------|----------------------------------|
| exp1_2     | 30                 | 480 (T=32, BS=32 → K/(T·BS)=16, ×30)     | 15360 (T=1, BS=32)               |
| exp1_3     | 100                | 3200 (D=16, BS=32 → K/(D·BS)=32, ×100)   | 51200 (D=1, BS=32)               |
| exp1_5     | 30                 | 15360 (constant: K/BS=512, ×30)          | same                             |

These are starting points. The min-opt-steps cell is the one to watch for
under-convergence: `exp1_3` D=16 with only ~3200 updates may need more than
100 epochs to fully reach its loss floor, and if it does that signal —
"large effective batch needs more epochs to converge" — is itself a
meaningful result. After a sanity run, bump `--epochs` for any specific
experiment that has not visibly plateaued.

`exp1_1` is unaffected by all of this — it is full-batch and `--steps` still
means "optimizer steps", which under full batch is also "epochs".
