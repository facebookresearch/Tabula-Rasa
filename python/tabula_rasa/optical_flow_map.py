# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp
from safetensors.flax import load_file

from .map import Map


class OpticalFlowMap(Map):
    def __init__(self, file_name):
        data = load_file(file_name)
        self.flow = data["flows"]
        self.flow /= jnp.array([self.flow.shape[1], self.flow.shape[2]])[
            None, None, None, :
        ]
        fps = 24
        self.duration = self.flow.shape[0] / fps

    def map(self, position, time, lens_coord):
        time /= self.duration
        time_index = jnp.array(time * jnp.array(self.flow.shape[0]), jnp.int32)

        aspect = jnp.array(self.flow.shape[2]) / jnp.array(self.flow.shape[3])
        coord = (position + jnp.array([1, 1])) / jnp.array([2, 2 / aspect])
        coord = coord * jnp.array(self.flow.shape[2:])
        space_index = jnp.array(coord, dtype=jnp.int32)

        if True:
            w = jnp.modf(coord)[0]
            motion00 = self.flow[time_index, space_index[1] + 0, space_index[0] + 0]
            motion01 = self.flow[time_index, space_index[1] + 0, space_index[0] + 1]
            motion10 = self.flow[time_index, space_index[1] + 1, space_index[0] + 0]
            motion11 = self.flow[time_index, space_index[1] + 1, space_index[0] + 1]
            motion = (
                w[0] * w[1] * motion11
                + w[0] * (1 - w[1]) * motion10
                + (1 - w[0]) * w[1] * motion01
                + (1 - w[0]) * (1 - w[1]) * motion00
            )
        else:
            motion = self.flow[time_index, :, space_index[1], space_index[0]]

        return {"position": position + motion}
