# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp

from .sampler import Sampler


class HammersleySampler(Sampler):
    def __init__(self, sample_count, total_sample_count):
        super().__init__(sample_count)
        self.total_sample_count = total_sample_count

    def sample(self, sample_key, base):
        def hammersley(num_points):
            def van_der_corput(index, base):
                def body_fun(i, x):
                    result, n, f = x
                    new_result = result + (n % base) * f
                    new_n = n // base
                    new_f = f / base
                    return jax.lax.cond(
                        n > 0,
                        lambda _: (new_result, new_n, new_f),
                        lambda _: (result, n, f),
                        operand=None,
                    )

                return jax.lax.fori_loop(0, 32, body_fun, (0.0, index, 1.0 / base))[0]

            indices = base * num_points + jnp.arange(0, num_points)

            result = jnp.empty((0, num_points))
            result = jnp.append(
                result, (indices / self.total_sample_count)[None, ...], axis=0
            )
            for i in range(0, self.dimension - 1):
                primes = [2, 3, 5, 7, 11, 13]
                slice = jax.vmap(van_der_corput, (0, None))(indices, primes[i])
                result = jnp.append(result, slice[None, ...], axis=0)
            return result.T

        result = hammersley(self.sample_count)

        if False:
            jitter = jax.random.uniform(sample_key, (1, self.dimension))
            result = result + jitter
            result = jnp.modf(result)[0]

        return result
