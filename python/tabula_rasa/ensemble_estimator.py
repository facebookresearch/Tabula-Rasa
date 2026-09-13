# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp

from .estimator import dict_add, Estimator


class EnsembleEstimator:
    def __init__(self, ensemble_size=1):
        self.ensemble_size = ensemble_size

    def estimate(
        self, map, field, sampler, filter, sample_key, field_key, position, time
    ):
        def convolve(sample_key, field_key):
            samples = Estimator.get_filter_samples(
                position, time, sample_key, sampler, filter, map
            )
            value, mean_value = jax.vmap(field.position_to_value, (0, 0, None))(
                samples["position"], samples["footprint"], field_key
            )
            result = {"value": value, "mean_value": mean_value}
            result = dict_add(result, samples)
            result = jax.tree.map(lambda x: jnp.mean(x, axis=0), result)
            return result

        sample_keys = jax.random.split(sample_key, self.ensemble_size)
        field_keys = jax.random.split(field_key, self.ensemble_size)
        ensemble = jax.vmap(convolve, (0, 0))(sample_keys, field_keys)

        expected_variance = jnp.var(ensemble["value"]) if self.ensemble_size > 1 else 1
        value = ensemble["value"][0] / jnp.sqrt(expected_variance)
        mean_value = ensemble["mean_value"][0]

        result = {"weights": jnp.ones((1,)), "value": value, "mean_value": mean_value}

        return result
