# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import jax.numpy as jnp

from .sampler import Sampler


class GridSampler(Sampler):
    def sample(self, key):
        assert self.dimension == 2

        sample_count = int(math.sqrt(self.sample_count))
        samples = (jnp.arange(0, sample_count) + 0.5) / sample_count
        samples = jnp.array(jnp.meshgrid(samples, samples))
        samples = jnp.transpose(samples, (1, 2, 0))
        samples = jnp.reshape(samples, (-1, 2))

        return samples
