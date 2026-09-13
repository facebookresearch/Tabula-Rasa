# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp

from .field import Field


class WhiteField(Field):
    def position_to_value(self, position, footprint, value_key):
        key = self.position_to_key(position)[0][0]
        value = self.key_to_value(key, value_key)
        return value, jnp.array([0])
