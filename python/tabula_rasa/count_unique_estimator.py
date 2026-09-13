# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import jax
import jax.numpy as jnp

from .estimator import dict_add, Estimator


class CountUniqueEstimator(Estimator):
    def __init__(self, bin_count=256, two_pass=False):
        self.bin_count = bin_count
        self.two_pass = two_pass

    def count_distinct(self, keys, weights):
        def count_sorting(keys, weights):
            if True:

                def key_segment_sum(values, keys):
                    segs = jnp.unique(keys, size=keys.shape[0], return_inverse=True)[1]
                    return jax.ops.segment_sum(values, segs, num_segments=keys.shape[0])

                counts = key_segment_sum(keys, weights)
                counts = counts / jnp.sum(counts)
                print(counts.shape)
                return counts
            else:
                counts = jnp.unique(keys, size=keys.shape[0], return_counts=True)[1]
                weights = counts / jnp.sum(counts)
                return weights

        def count_histo(keys, weights):
            keys = keys % self.bin_count
            counts = jnp.zeros((self.bin_count,), dtype=jnp.float32)
            counts = counts.at[keys].add(weights)
            counts = counts / jnp.sum(counts)
            return counts

        def count_linear(keys):
            bin_count = 2**12
            keys = jnp.bitwise_and(keys, bin_count - 1)
            mask = jnp.ones(bin_count)
            mask = mask.at[keys].set(0)

            fraction = jnp.mean(mask)
            count = -bin_count * jnp.log(fraction) * 2

            weights = jnp.where(jnp.arange(0, keys.shape[0]) < count, 1, 0)
            return weights / jnp.sum(weights)

        def count_none(keys):
            return jnp.ones(keys.shape[0]) / keys.shape[0]

        keys = jnp.ravel(keys)
        weights = jnp.ravel(weights)
        return count_histo(keys, weights)

    def estimate(
        self,
        map,
        field,
        sampler,
        filter,
        sample_key,
        sample_base,
        field_key,
        position,
        time,
    ):
        samples = Estimator.get_filter_samples(
            position, time, sample_key, sample_base, sampler, filter, map
        )

        value, mean_value = jax.vmap(field.position_to_value, in_axes=(0, 0, None))(
            samples["position"], samples["footprint"], field_key
        )

        result = {"value": value, "mean_value": mean_value}
        result = dict_add(result, samples)
        result = jax.tree.map(lambda x: jnp.mean(x, axis=0), result)

        if self.two_pass:
            # Use independent samples for weight estimation to eliminate bias
            # from reusing samples in a nonlinear combination of estimators.
            # Only effective when the sampler uses sample_key for randomization.
            if type(sampler).__name__ == "HammersleySampler":
                warnings.warn(
                    "Two-pass estimation with HammersleySampler: samples may not be "
                    "independent because HammersleySampler ignores sample_key. "
                    "Consider using RandomSampler or LPGKSampler, or enabling "
                    "Cranley-Patterson jitter in HammersleySampler.",
                    stacklevel=2,
                )
            sample_key_2 = jax.random.fold_in(sample_key, 69420)
            samples_2 = Estimator.get_filter_samples(
                position, time, sample_key_2, sample_base, sampler, filter, map
            )
            keys, weights = jax.vmap(field.position_to_key)(
                samples_2["position"], samples_2["footprint"]
            )
        else:
            # One-pass: reuse the same samples for both value and weight estimation.
            # Introduces a subtle bias from the nonlinear combination, but the
            # estimator is still consistent and the bias is negligible in practice.
            keys, weights = jax.vmap(field.position_to_key)(
                samples["position"], samples["footprint"]
            )

        weights = self.count_distinct(keys, weights)
        result = dict_add({"weights": weights}, result)

        return result
