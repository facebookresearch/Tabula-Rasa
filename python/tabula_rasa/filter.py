# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp


class Filter:
    def __init__(self, resolution, shutter=0, fps=20):
        self.size = 1 / jnp.array(resolution[0])
        self.shutter = shutter
        self.fps = fps
