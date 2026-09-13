# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import jax
import jax.numpy as jnp

from .grid_sampler import GridSampler
from .sampler import Sampler


class JitteredSampler(Sampler):
    def sample(self, key):
        assert self.dimension == 2

        resolution = int(math.sqrt(self.sample_count))
        grid = GridSampler(self.sample_count).sample(key)
        jitter = jax.random.uniform(key, (self.sample_count, 2)) / resolution
        result = grid + jitter
        result = jnp.modf(result)[0]
        return result
