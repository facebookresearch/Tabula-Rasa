# CountUniqueEstimator Deep Dive

This document provides a detailed analysis of `count_unique_estimator.py`, the core innovation of the Tabula Rasa method (paper Sec. 4.2.3).

## Problem Statement

A pixel value is a weighted sum of independent unit-variance Gaussians (Eq. 5):
```
P(x,t)(ξ) = Σ A_i · F_i(x,t)(ξ)
```
where `A_i` are fractional areas of each unique noise cell in the pixel footprint, and `F_i` are i.i.d. N(0,1). The resulting variance is `Σ A_i²` (Eq. 6), which is < 1. To restore unit variance, divide by `sqrt(Σ A_i²)` (Eq. 7). But we don't know the number of cells, their shapes, or their areas. `CountUniqueEstimator` estimates these weights via a histogram sketch.

## `count_distinct` — The Histogram Sketch (lines 11-52)

Four strategies are implemented; only `count_histo` is active (line 52).

### `count_histo` (lines 28-33) — Active method
```python
def count_histo(keys, weights):
    keys = keys % self.bin_count          # H_s: reduce to bin_count bins
    counts = jnp.zeros((self.bin_count,), dtype=jnp.float32)
    counts = counts.at[keys].add(weights) # scatter-add weights per bin
    counts = counts / jnp.sum(counts)     # normalize to categorical distribution
    return counts
```
This is the paper's small hash H_s. Each sample's big hash key (from `field.position_to_key`) is reduced modulo `bin_count` (default 256). Samples hitting the same noise cell hash to the same bin (modulo collisions), so the histogram captures the weight distribution.

The `weights` parameter matters: with `BlendedAdaptiveField`, each sample produces two keys (two MIP levels) with interpolation weights (e.g. `[0.3, 0.7]`). These weights, not just counts, are accumulated into the histogram, correctly accounting for tri-linear MIP blending in the variance estimate.

Paper quote: *"we apply a different ('small') hash function H_s... This hashing reduces the many discrete lattice coordinates to very few values, e.g., o=256. Few enough, so that we can maintain an approximate histogram of all of them in a per-GPU thread buffer."*

### `count_sorting` (lines 12-26) — Unused exact method
Uses `jnp.unique` + `segment_sum` for exact weight estimation (no hash collisions). O(n log n) and variable-sized output makes it impractical on GPU. Corresponds to the paper's ideal case of a histogram with n=2^32 entries.

### `count_linear` (lines 35-45) — Unused linear counting
Implements [Whang et al. 1990]: bitmap of 2^12 entries, each key sets a bit, unset fraction estimates cardinality via `-bin_count * log(fraction)`. Only estimates total unique count (not per-cell weights), so it produces uniform weights — a cruder approximation.

### `count_none` (lines 47-48) — Unused no-correction baseline
Returns uniform `1/n` weights. Equivalent to the naive estimator of Eq. 2 with no variance correction.

## `estimate` — Two-Pass Estimator (lines 90-127)

The paper notes a subtle bias when reusing samples for both value and area estimation (Sec. 4.2.3, final paragraph). The implementation addresses this with two independent passes.

### Pass 1 — Value estimation (lines 102-112)
```python
samples = Estimator.get_filter_samples(position, time, sample_key, ...)
value, mean_value = jax.vmap(field.position_to_value, ...)(
    samples["position"], samples["footprint"], field_key
)
result = jax.tree.map(lambda x: jnp.mean(x, axis=0), result)
```
Generates samples using `sample_key`, evaluates noise at each position, averages. This computes P_hat from Eq. 2.

### Pass 2 — Weight estimation (lines 113-125)
```python
sample_key_2 = jax.random.fold_in(sample_key, 69420)  # independent key
samples_2 = Estimator.get_filter_samples(position, time, sample_key_2, ...)
keys_2, weights_2 = jax.vmap(field.position_to_key)(
    samples_2["position"], samples_2["footprint"]
)
weights_2 = self.count_distinct(keys_2, weights_2)
```
Generates **independent** samples (different key via `fold_in`) and uses them solely to build the histogram sketch. `field.position_to_key` returns big hash keys and MIP-blend weights.

Paper quote: *"We could also eliminate the bias by using two independent sets of samples to estimate the value and the area separately, at the cost of some duplicated computation."*

The commented-out one-pass version (lines 54-87) shows the evolution — it used the same samples for both, which is the biased variant discussed in the paper.

## `finish` in `estimator.py` — Applying the Correction (lines 25-43)

After `jax.lax.scan` accumulates results across `batch_size` iterations in the renderer:
```python
def finish(self, accumulator):
    weights = frame["weights"]      # histogram, summed across batches
    value = frame["value"]          # sum of noise values across batches
    mean_value = frame["mean_value"]
    variance = 1 / jnp.sqrt(jnp.sum(weights**2, axis=-1))  # 1/sqrt(Σ A_i²)
    value = value * variance + mean_value
```

`weights` is the normalized histogram — each entry estimates an `A_i`. Squaring and summing gives `Σ A_i²` (variance of the weighted sum, Eq. 6). Dividing by `sqrt(Σ A_i²)` restores unit variance (Eq. 7).

`mean_value` is currently always `jnp.array([0])` from `BlendedAdaptiveField`, so this addition is a no-op in practice.

## Batch Accumulation Detail

The renderer's `accumulate_sample` (renderer.py:350-353) **sums** results across batches:
```python
new_carry = jax.tree.map(lambda c, s: c + s, carry, sample)
```

The `weights` histogram is summed (not averaged) across batches. Since `count_histo` normalizes each batch to sum to 1, after `batch_size` batches the histogram sums to `batch_size`. This does not affect correctness because `finish()` computes `weights**2` — the relative distribution matters, not the absolute magnitude. Both `value` and `weights` scale proportionally with batch count.

## Why `bin_count=256`

The paper's Fig. 7 (row 3, "Dist. var.") shows:
- Too few bins → hash collisions merge distinct cells → overestimates variance → under-correction
- Sweet spot around 64 bins, stabilizes by 256
- Beyond 256 → diminishing returns, wastes GPU thread memory
- the `bin-counts` experiment in `python/examples/run_experiments.py` sweeps 2^0 to 2^11 to validate empirically

## Connection to Paper's Fig. 4

The code directly implements Fig. 4's procedure:
1. For each sample, trace ray → world-space hit → `field.position_to_key` returns hash code (the "upright Roman A-D" labels in Fig. 4)
2. `count_histo` bins these codes via H_s (the "small hash" column)
3. Normalized histogram entries are the `A_i` estimates
4. `Σ A_i²` gives variance, `1/sqrt(Σ A_i²)` scales the result (last row of Fig. 4)

## Data Flow Summary

```
estimate() called per pixel per batch:
  Pass 1: sample_key → get_filter_samples → field.position_to_value → mean → {value, mean_value}
  Pass 2: sample_key_2 → get_filter_samples → field.position_to_key → count_distinct → {weights}

renderer accumulates across batch_size iterations via jax.lax.scan (summing)

finish() called once per pixel:
  variance = 1 / sqrt(sum(weights²))
  final_value = accumulated_value * variance + accumulated_mean_value
```
