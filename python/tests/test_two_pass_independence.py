# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Verify that two-pass CountUniqueEstimator produces independent sub-pixel samples.

Calls get_filter_samples directly for a single pixel — no vmap, no scan, no JIT.
Tests RandomSampler (should differ) and HammersleySampler (known bug: identical).

Usage:
    cd python
    python tests/test_two_pass_independence.py
"""

import os

import jax
import jax.numpy as jnp

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tabula_rasa import (
    BoxFilter,
    Estimator,
    HammersleySampler,
    RandomSampler,
    VaryingMap,
)

SHAPE = (128, 128)
FPS = 24
SAMPLE_COUNT = 16

position = jnp.array([0.0, 0.0])  # pixel center
time = 0.5
sample_base = 0
sample_key = jax.random.PRNGKey(42)
sample_key_2 = jax.random.fold_in(sample_key, 69420)  # same derivation as estimator

map_obj = VaryingMap()
filter_obj = BoxFilter(SHAPE, 0, FPS)

for name, sampler in [
    ("RandomSampler", RandomSampler(SAMPLE_COUNT, SAMPLE_COUNT)),
    ("HammersleySampler", HammersleySampler(SAMPLE_COUNT, SAMPLE_COUNT)),
]:
    print(f"\n=== {name} ===")

    samples_1 = Estimator.get_filter_samples(
        position, time, sample_key, sample_base, sampler, filter_obj, map_obj
    )
    samples_2 = Estimator.get_filter_samples(
        position, time, sample_key_2, sample_base, sampler, filter_obj, map_obj
    )

    pos1 = samples_1["position"]
    pos2 = samples_2["position"]

    print(f"  Pass 1 positions:\n{pos1}")
    print(f"  Pass 2 positions:\n{pos2}")
    print(f"  max|diff|: {float(jnp.max(jnp.abs(pos1 - pos2))):.8f}")
    print(f"  Identical: {bool(jnp.allclose(pos1, pos2))}")
