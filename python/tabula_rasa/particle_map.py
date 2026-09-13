# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import jax
import jax.numpy as jnp

from .map import Map

Paticle = namedtuple("Particle", "original_position current_position")


class ParticleMap(Map):
    def test_particles(self, position, particle):
        weight = position - particle.current_position
        weight = jnp.linalg.norm(weight)
        weight = jnp.exp(-((weight / 5.0) ** 2))
        motion = weight * (particle.original_position - particle.current_position)
        return motion, weight

    def get_particles(self, time):
        return Paticle(
            jnp.array([[0.5, 0.5], [0, 0]]),
            jnp.array([[0.5 + 0.25 * time, 0.5], [0, 0]]),
        )

    def map(self, position, time, lens_coord):
        particles = self.get_particles(time)
        test = jax.vmap(self.test_particles, in_axes=(None, 0))
        motion, weight = test(position, particles)
        motion = jnp.sum(motion, axis=0)
        weight = jnp.sum(weight, axis=0)
        motion /= weight
        return {"position": position + motion}
