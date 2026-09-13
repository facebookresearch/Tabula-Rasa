# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jax.numpy as jnp

from .field import Field
from PIL import Image


class ImageField(Field):
    def __init__(self, resolution):
        super(ImageField, self).__init__(resolution)
        self.image = jnp.array(Image.open("in/photo.jpg")) / 255 * 6 - 3
        # self.image = self.image[:512, :512]

    def position_to_value(self, position, footprint, value_key):
        key = self.position_to_key(position)[0][0]
        value = self.key_to_value(key, value_key)

        self.image_scale = 200
        index = jnp.array(jnp.floor(position * self.image_scale), jnp.int32)
        # index = jnp.clip(index, 0, self.image_scale)
        pixel_value = self.image[index[1], index[0]]

        return value, pixel_value
