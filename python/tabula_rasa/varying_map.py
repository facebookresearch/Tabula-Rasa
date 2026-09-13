# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp

from .map import Map


class VaryingMap(Map):
    def map(self, position, time, lens_coord):
        angle = 0.1 * time * 2 * jnp.pi
        matrix = jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
        )
        position = position @ matrix
        time /= 3
        position *= 2 * (1 - .95 * time)
        return {"position": position}
