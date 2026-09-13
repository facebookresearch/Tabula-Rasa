# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp

from .filter import Filter


class GaussFilter(Filter):
    @staticmethod
    def probit(x):
        return jnp.sqrt(2) * jax.scipy.special.erfinv(2 * x - 1)

    def filter(self, x):
        return jnp.array(
            GaussFilter.probit(x + 0.5) / 2 * 2 * self.width, dtype=jnp.float32
        )
