# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp


def hash_nd(x):
    x = jnp.atleast_1d(x)

    key = jax.random.PRNGKey(0)
    hash_dimensions = x.shape[0]
    for i in range(0, hash_dimensions):
        key = jax.random.fold_in(key, x[i])
    return jax.random.randint(key, (1,), 0, 2**30)[0]


class Field:
    def __init__(self, resolution):
        self.resolution = resolution
        self.value_dimension = 16

    def key_to_value(self, position_key, value_key):
        position_key = jax.random.PRNGKey(position_key)
        combined_key = position_key + value_key
        return jax.random.normal(combined_key, (self.value_dimension,))

    def position_to_key(self, x, footprint=None):
        x = (x + 1) / 2
        x *= self.resolution
        x = jnp.floor(x)
        x = jnp.array(x, dtype=int)
        return jnp.array([hash_nd(x)]), jnp.array([1])
