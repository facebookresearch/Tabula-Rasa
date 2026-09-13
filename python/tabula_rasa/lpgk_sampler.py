# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Larcher-Pillichshammer Gruenschloss-Keller (0,3) sequence sampler in JAX.

This is a (0,3) sequence in base 2. The first two dimensions are
the Larcher-Pillichshammer (0,2) sequence:

G. Larcher and F. Pillichshammer.
"Walsh series analysis of the L2-discrepancy of symmetrisized point sets."
Monatsheft Mathematik 132, 1–18, 2001.

and the third dimension completes the (0,3) sequence using the construction
by Gruenschloss & Keller from:

L. Gruenschloss and A. Keller.
"(t, m, s)-Nets and Maximized Minimum Distance, Part II",
in P. L'Ecuyer and A. Owen (eds.),
Monte Carlo and Quasi-Monte Carlo Methods 2008, Springer-Verlag, 2009.
"""

import jax
import jax.numpy as jnp
from .sampler import Sampler


def random_digit_scramble(f: jnp.ndarray, scramble: jnp.ndarray) -> jnp.ndarray:
    """Apply random digit scrambling to a float value.

    Converts float to integer representation, XORs with scramble value,
    and converts back to float in [0, 1).

    Args:
        f: Float value(s) in [0, 1)
        scramble: Unsigned 32-bit scramble value(s)

    Returns:
        Scrambled float value(s) in [0, 1)
    """
    # Use float constant (4294967296.0 = 2^32) to avoid integer overflow in JAX
    int_repr = (f * 4294967296.0).astype(jnp.uint32)
    scrambled = int_repr ^ scramble
    return scrambled.astype(jnp.float32) * 2.3283064365386962890625e-10


# Loop body from pbrt-v2, src/core/montecarlo.h::LarcherPillichshammer2 --
# pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys,
# BSD-2-Clause. https://github.com/mmp/pbrt-v2
# MODIFIED: rewritten as a 32-iteration jax.lax.scan rather than a for loop, and
# normalised over the full 32 bits (* 2^-32) instead of pbrt's 24-bit
# min(((scramble >> 8) & 0xffffff) / float(1 << 24), OneMinusEpsilon).
# Full notice in THIRD_PARTY_NOTICES.md §3.
def larcher_pillichshammer_riu(n: jnp.ndarray, scramble: jnp.ndarray) -> jnp.ndarray:
    """Compute Larcher-Pillichshammer radical inverse with scrambling.

    This generates the second dimension of the LP (0,2) sequence.

    Args:
        n: Sample index (unsigned integer)
        scramble: Scramble value for randomization

    Returns:
        Float value in [0, 1) for this dimension
    """

    def body_fn(carry, _):
        n_val, scramble_val, v = carry
        new_scramble = jnp.where(n_val & 1, scramble_val ^ v, scramble_val)
        new_v = v | (v >> 1)
        new_n = n_val >> 1
        return (new_n, new_scramble, new_v), None

    v_init = jnp.uint32(1 << 31)
    init_carry = (n.astype(jnp.uint32), scramble.astype(jnp.uint32), v_init)

    (_, final_scramble, _), _ = jax.lax.scan(body_fn, init_carry, None, length=32)

    return final_scramble.astype(jnp.float32) * 2.3283064365386962890625e-10


# Technique from pbrt-v2, src/core/montecarlo.h::Sobol2 -- pbrt source code
# Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys, BSD-2-Clause.
# https://github.com/mmp/pbrt-v2
# MODIFIED: pbrt's v ^= v >> 1 recurrence applied to the Gruenschloss-Keller
# third-dimension matrix (initial 3 << 30, accumulating v2 << 1); rewritten as a
# jax.lax.scan; normalised over the full 32 bits.
# Full notice in THIRD_PARTY_NOTICES.md §3.
def gruenschloss_keller_riu(n: jnp.ndarray, scramble: jnp.ndarray) -> jnp.ndarray:
    """Compute Gruenschloss-Keller radical inverse with scrambling.

    This generates the third dimension completing the (0,3) sequence.

    Args:
        n: Sample index (unsigned integer)
        scramble: Scramble value for randomization

    Returns:
        Float value in [0, 1) for this dimension
    """

    def body_fn(carry, _):
        n_val, scramble_val, v2 = carry
        new_scramble = jnp.where(n_val & 1, scramble_val ^ (v2 << 1), scramble_val)
        new_v2 = v2 ^ (v2 >> 1)
        new_n = n_val >> 1
        return (new_n, new_scramble, new_v2), None

    v2_init = jnp.uint32(3 << 30)
    init_carry = (n.astype(jnp.uint32), scramble.astype(jnp.uint32), v2_init)

    (_, final_scramble, _), _ = jax.lax.scan(body_fn, init_carry, None, length=32)

    return final_scramble.astype(jnp.float32) * 2.3283064365386962890625e-10


# Permutation adapted from A. Kensler, "Correlated Multi-Jittered Sampling",
# Pixar Technical Memo 13-01, 2013. See THIRD_PARTY_NOTICES.md §4.
def permute(i: jnp.ndarray, n: int, seed: jnp.ndarray) -> jnp.ndarray:
    """Permute sample index using hash-based permutation.

    This creates a permutation of indices for decorrelation across dimensions.

    Args:
        i: Sample index
        n: Number of samples
        seed: Seed value for permutation

    Returns:
        Permuted index in [0, n)
    """
    w = n - 1
    w |= w >> 1
    w |= w >> 2
    w |= w >> 4
    w |= w >> 8
    w |= w >> 16

    def body_fn(carry, _):
        idx = carry
        idx = idx ^ seed
        idx = idx * jnp.uint32(0xE170893D)
        idx = idx ^ (seed >> 16)
        idx = idx ^ ((idx & w) >> 4)
        idx = idx ^ (seed >> 8)
        idx = idx * jnp.uint32(0x0929EB3F)
        idx = idx ^ (seed >> 23)
        idx = idx ^ ((idx & w) >> 1)
        idx = idx * jnp.uint32(1 | seed >> 27)
        idx = idx * jnp.uint32(0x6935FA69)
        idx = idx ^ ((idx & w) >> 11)
        idx = idx * jnp.uint32(0x74DCB303)
        idx = idx ^ ((idx & w) >> 2)
        idx = idx * jnp.uint32(0x9E501CC3)
        idx = idx ^ ((idx & w) >> 2)
        idx = idx * jnp.uint32(0xC860A3DF)
        idx = idx & w
        return idx, None

    init_carry = i.astype(jnp.uint32)
    final_idx, _ = jax.lax.scan(body_fn, init_carry, None, length=8)

    return jnp.where(final_idx < n, final_idx, i.astype(jnp.uint32))


class LPGKSampler(Sampler):
    """Larcher-Pillichshammer Gruenschloss-Keller (0,3) sequence sampler.

    This sampler generates low-discrepancy samples using the (0,3) sequence
    in base 2. It cycles through 3 dimensions:
      - Dimension 0: Permuted index normalized by total samples (with digit scrambling)
      - Dimension 1: Larcher-Pillichshammer radical inverse
      - Dimension 2: Gruenschloss-Keller radical inverse

    For dimensions > 3, the pattern repeats with different scramble seeds.
    """

    def __init__(self, sample_count, total_sample_count):
        super().__init__(sample_count, total_sample_count)

    def sample(self, sample_key, sample_base):
        """Generate samples using the LP-GK (0,3) sequence.

        Args:
            sample_key: JAX random key for scrambling (consistent across batches)
            sample_base: Base index for batch offset into the full sequence

        Returns:
            Array of shape (sample_count, dimension) with samples in [0, 1)
        """
        # Do NOT fold_in sample_base - we want consistent scrambling across all batches
        # so that different batches form slices of the same scrambled sequence
        keys = jax.random.split(sample_key, 3)
        scramble1 = jax.random.bits(keys[0], dtype=jnp.uint32)
        scramble2 = jax.random.bits(keys[1], dtype=jnp.uint32)
        scramble3 = jax.random.bits(keys[2], dtype=jnp.uint32)

        # Offset indices by sample_base to get the correct slice of the sequence
        # When sample_base=0: indices=[0,1,...,sample_count-1]
        # When sample_base=1: indices=[sample_count, sample_count+1, ...]
        indices = sample_base * self.sample_count + jnp.arange(
            self.sample_count, dtype=jnp.uint32
        )
        inv = 1.0 / self.total_samples

        def compute_sample(i):
            result = jnp.zeros(self.dimension)

            def dim_body(d, res):
                dim_seed = jnp.uint32(0x68BC21EB) * jnp.uint32(d // 3 + 1)
                perm_key = jax.random.fold_in(sample_key, d)
                perm_seed = jax.random.bits(perm_key, dtype=jnp.uint32)
                s = permute(i, self.total_samples, perm_seed)

                dim_mod = d % 3
                val = jax.lax.switch(
                    dim_mod,
                    [
                        lambda: random_digit_scramble(
                            s.astype(jnp.float32) * inv, scramble1 * dim_seed
                        ),
                        lambda: larcher_pillichshammer_riu(s, scramble2 * dim_seed),
                        lambda: gruenschloss_keller_riu(s, scramble3 * dim_seed),
                    ],
                )
                return res.at[d].set(val)

            return jax.lax.fori_loop(0, self.dimension, dim_body, result)

        samples = jax.vmap(compute_sample)(indices)
        return samples
