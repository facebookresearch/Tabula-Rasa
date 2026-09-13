# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp

from .adaptive_field import AdaptiveField


class BlendedAdaptiveField(AdaptiveField):
    def position_to_value(self, position, footprint, value_key):
        level = self.footprint_to_level(footprint)
        discrete_level = jnp.array(jnp.floor(level), jnp.int32)
        value0 = self.level_position_to_key(position, discrete_level + 0, value_key)
        value1 = self.level_position_to_key(position, discrete_level + 1, value_key)
        weight = level - discrete_level
        y = value0 * (1 - weight) + value1 * weight
        return y, jnp.array([0])
