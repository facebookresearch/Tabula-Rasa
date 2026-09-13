# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp

from .adaptive_field import AdaptiveField


class HardAdaptiveField(AdaptiveField):
    def position_to_value(self, position, footprint, value_key):
        level = self.footprint_to_level(footprint)
        discrete_level = jnp.array(level, dtype=jnp.int32)
        return self.level_position_to_key(position, discrete_level, value_key)
