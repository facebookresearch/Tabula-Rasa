# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time


class Timer:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def report(self):
        end_time = time.time()
        interval = end_time - self.start_time
        print(self.name + " took {:0.3f} seconds".format(interval))
