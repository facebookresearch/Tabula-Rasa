# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax

from .sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, sample_count, total_sample_count):
        super().__init__(sample_count, total_sample_count)

    def sample(self, sample_key, sample_base):
        sample_key = jax.random.fold_in(sample_key, sample_base)
        return jax.random.uniform(sample_key, (self.sample_count, self.dimension))
