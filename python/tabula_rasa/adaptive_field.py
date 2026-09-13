# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp

from .field import Field, hash_nd


class AdaptiveField(Field):
    def footprint_to_level(self, footprint):
        footprint = 16 / (jnp.sqrt(footprint))
        level = jnp.log2(footprint)
        return level

    def position_to_key(self, position, footprint):
        level = self.footprint_to_level(footprint)
        discrete_level = jnp.array(jnp.floor(level), jnp.int32)

        def level_key(position, level):
            position /= 4
            position = (position + 1) / 2
            position *= jnp.array(self.resolution)
            position *= 2**level
            position = jnp.floor(position)
            position = jnp.array(position, dtype=int)
            return hash_nd(position)

        key0 = level_key(position, discrete_level + 0)
        key1 = level_key(position, discrete_level + 1)

        weight = level - discrete_level

        return jnp.array([key0, key1]), jnp.array([1 - weight, weight])

    def level_position_to_key(self, position, level, value_key):
        key = Field.position_to_key(self, position * (2**level) / 4)[0][0]
        level_value_key = jax.random.fold_in(value_key, level)
        return self.key_to_value(key, level_value_key)
