# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Sampler:
    dimension = 5

    def __init__(self, sample_count, total_samples=None):
        self.sample_count = sample_count
        self.total_samples = (
            total_samples if total_samples is not None else sample_count
        )
