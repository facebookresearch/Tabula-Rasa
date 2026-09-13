# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import jax
import jax.numpy as jnp


def dict_add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


class Estimator:
    def estimate(
        self, map, field, sampler, filter, sample_key, field_key, position, time
    ):
        samples = Estimator.get_filter_samples(
            position, time, sample_key, sampler, filter, map
        )
        value = jax.vmap(field.position_to_value, (0, 0, None))(
            samples["position"], samples["footprint"], field_key
        )
        result = dict_add({"value": value}, samples)
        result = jax.tree.map(lambda x: jnp.mean(x, axis=0), result)
        return result

    def finish(self, accumulator):
        frame = copy.deepcopy(accumulator)
        weights = frame["weights"]
        value = frame["value"]
        mean_value = frame["mean_value"]

        variance = 1 / jnp.sqrt(jnp.sum(weights**2, axis=-1))
        if value.ndim == 3:
            variance = variance[..., None]
        value = value * variance + mean_value

        frame["value"] = value

        wanted_names = ["value", "id", "position", "normal", "footprint"]
        result = {}
        for wanted_name in wanted_names:
            if wanted_name in frame:
                result[wanted_name] = frame[wanted_name]
        return result

    @staticmethod
    def variance_from_weights(weights):
        return 1 / jnp.sqrt(jnp.sum(weights**2))

    @staticmethod
    def get_filter_samples(
        position, time, sample_key, sample_base, sampler, filter, map
    ):
        samples = sampler.sample(sample_key, sample_base)

        coords = samples[..., 0:2] * 2 - 1
        coords = jax.vmap(filter.filter)(coords)
        coords = coords + jnp.repeat(position[None, ...], sampler.sample_count, axis=0)

        times = filter.shutter / filter.fps * samples[..., 3] + time

        lens_coords = samples[..., 4:5]

        def get_footprint(coord, time, lens_coord):
            def position_map(coord):
                return map.map(coord, time, lens_coord)["position"]

            def cross(a, b):
                if a.shape[-1] == 3:
                    return jnp.linalg.cross(a, b)
                else:
                    return a[0] * b[1] - a[1] * b[0]

            jacobian = jax.jacfwd(position_map)(coord).T
            footprint = cross(jacobian[0], jacobian[1])
            footprint = jnp.linalg.norm(footprint)
            # footprint = jnp.nan_to_num(footprint)
            return footprint

        footprint = jax.vmap(get_footprint)(coords, times, lens_coords)
        map_result = jax.vmap(map.map)(coords, times, lens_coords)
        return dict_add(map_result, {"footprint": footprint})
