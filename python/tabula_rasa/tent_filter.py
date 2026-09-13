# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp

from .filter import Filter


class TentFilter(Filter):
    def filter(self, x):
        x = jnp.where(x > 0, 1 - jnp.sqrt(x), -1 + jnp.sqrt(-x))
        return 2 * x * self.width
