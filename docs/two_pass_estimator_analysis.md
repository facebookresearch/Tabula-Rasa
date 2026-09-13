# Two-Pass vs One-Pass Estimator Analysis

This document analyzes whether the two-pass `CountUniqueEstimator` correctly uses independent samples to prevent bias, as intended by the paper (Sec. 4.2.3).

## Background: The Bias Problem

The paper identifies a "subtle form of bias" when the same samples are used to both:
1. Estimate the noise value (P_hat, Eq. 2)
2. Estimate the area weights (A_i) for variance correction (Eq. 7)

This is the same bias found in estimating gradients of squared loss functions [Nimier-David et al. 2022]. The paper proposes using two independent sets of samples as a fix.

## Implementation: Two-Pass Estimator (`count_unique_estimator.py:90-127`)

```python
# Pass 1: value estimation using sample_key
samples = Estimator.get_filter_samples(position, time, sample_key, sample_base, ...)
value, mean_value = jax.vmap(field.position_to_value, ...)(samples["position"], ...)

# Pass 2: weight estimation using sample_key_2
sample_key_2 = jax.random.fold_in(sample_key, 69420)  # derive independent key
samples_2 = Estimator.get_filter_samples(position, time, sample_key_2, sample_base, ...)
keys_2, weights_2 = jax.vmap(field.position_to_key)(samples_2["position"], ...)
weights_2 = self.count_distinct(keys_2, weights_2)
```

The two passes differ only in `sample_key` vs `sample_key_2`. All other arguments (`sample_base`, `sampler`, `filter`, `map`) are shared.

## Finding: Independence Depends on the Sampler

### HammersleySampler — BROKEN (samples are identical)

`hammersley_sampler.py:12-48`:
```python
def sample(self, sample_key, base):
    result = hammersley(self.sample_count)  # purely deterministic

    if False:                               # jitter is DISABLED
        jitter = jax.random.uniform(sample_key, ...)
        ...

    return result
```

The Hammersley sequence is deterministic — it depends only on `base` (batch index) and `sample_count`/`total_sample_count`. **The `sample_key` parameter is never read** (the `if False` block disables the only code path that would use it).

Consequence: Pass 1 and Pass 2 call `get_filter_samples` with different `sample_key` values but the sampler ignores them. Both passes generate **the exact same sample positions**. The two-pass design is functionally equivalent to the commented-out one-pass estimator (lines 54-87), completely defeating the purpose.

### RandomSampler — CORRECT (truly independent)

`random_sampler.py:10-12`:
```python
def sample(self, sample_key, sample_base):
    sample_key = jax.random.fold_in(sample_key, sample_base)
    return jax.random.uniform(sample_key, (self.sample_count, self.dimension))
```

Uses `sample_key` directly. Different keys → different positions → truly independent passes.

### LPGKSampler — CORRECT (truly independent)

`lpgk_sampler.py:162-174`:
```python
def sample(self, sample_key, sample_base):
    keys = jax.random.split(sample_key, 3)
    scramble1 = jax.random.bits(keys[0], dtype=jnp.uint32)
    scramble2 = jax.random.bits(keys[1], dtype=jnp.uint32)
    scramble3 = jax.random.bits(keys[2], dtype=jnp.uint32)
    ...
```

Uses `sample_key` for scrambling seeds. Different keys → differently scrambled QMC sequences → independent passes.

## Summary Table

| Sampler | Uses `sample_key`? | Two-pass independent? | Notes |
|---|---|---|---|
| `HammersleySampler` | No (jitter disabled) | **No** — identical samples | Bug: defeats two-pass design |
| `RandomSampler` | Yes | Yes | Correct |
| `LPGKSampler` | Yes (scrambling) | Yes | Correct |
| `JitteredSampler` | Yes | Yes | Correct |

## Impact on Current Code

- the `convergence` experiment in `python/examples/run_experiments.py` uses `RandomSampler` — **not affected**
- `test_suit()` uses `HammersleySampler` — **affected**
- `test_performance()` uses `HammersleySampler` — **affected**
- `test_flows()` uses `HammersleySampler` — **affected**

## The Commented-Out One-Pass Estimator (lines 54-87)

For reference, the one-pass version computed both value and weights from the same sample set:
```python
samples = get_filter_samples(position, time, sample_key, ...)
keys, weights = jax.vmap(field.position_to_key)(samples["position"], ...)
weights = self.count_distinct(keys, weights)
value, mean_value = jax.vmap(field.position_to_value, ...)(samples["position"], ...)
```

This has the bias the paper discusses but is computationally cheaper (one set of ray traces instead of two). With `HammersleySampler`, the active two-pass code is doing the same thing as this one-pass code but at 2x the cost.

## Potential Fix

Enable the Hammersley jitter (change `if False` to `if True` in `hammersley_sampler.py:43`):
```python
if True:  # was: if False
    jitter = jax.random.uniform(sample_key, (1, self.dimension))
    result = result + jitter
    result = jnp.modf(result)[0]
```

This applies Cranley-Patterson rotation using `sample_key`, making different keys produce different (but still low-discrepancy) sample positions. This is standard practice for randomized QMC and would restore the intended independence between passes while preserving the QMC convergence properties.
