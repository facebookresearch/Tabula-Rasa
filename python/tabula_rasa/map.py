# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Map:
    def __init__(self):
        self.range = 2

    def map(self, position, time, lens_coord):
        return {"position": position}
