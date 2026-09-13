# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import jax.numpy as jnp

from .map import Map

Ray = namedtuple("Ray", "position direction")


class RaytracingMap(Map):
    class Primitive:
        def __init__(self, position, velocity, inverse_mass):
            self.position_track = (position,)
            self.velocity_track = (velocity,)
            self.inverse_mass = inverse_mass

        def get_animated_position(self, time):
            path_sampling_rate = 20
            index = jnp.array(time * path_sampling_rate, dtype=jnp.int32)
            return self.position_track[index]

        def get_local_position(self, time):
            return self.get_animated_position(time) - self.get_animated_position(0)

    class Sphere(Primitive):
        def __init__(
            self,
            position,
            velocity,
            inverse_mass,
            radius,
        ):
            super().__init__(position, velocity, inverse_mass)
            self.radius = radius

        def intersect(self, ray, time):
            position_to_sphere = ray.position - self.get_animated_position(time)
            a = jnp.dot(ray.direction, ray.direction, precision="float32")
            b = 2 * jnp.dot(ray.direction, position_to_sphere, precision="float32")
            c = (
                jnp.dot(position_to_sphere, position_to_sphere, precision="float32")
                - self.radius**2
            )
            discriminant = b**2.0 - 4.0 * a * c

            sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 1e-7))

            t0 = (-b + sqrt_disc) / (2 * a)
            t1 = (-b - sqrt_disc) / (2 * a)
            return jnp.where(
                discriminant < 0.0, jnp.array([jnp.nan, jnp.nan]), jnp.array([t0, t1])
            )

        def get_normal(self, position, time):
            return (position - self.get_animated_position(time)) / self.radius

    class Plane(Primitive):
        def __init__(self, position, normal):
            super().__init__(position, jnp.array([0, 0, 0]), 0)
            self.normal = jnp.array(normal)

        def intersect(self, ray, time):
            position_to_plane = ray.position - self.get_animated_position(time)
            a = jnp.dot(ray.direction, self.normal, precision="float32")
            b = jnp.dot(position_to_plane, self.normal, precision="float32")
            t = -b / a
            return jnp.array([t])

        def get_normal(self, position, time):
            return self.normal

    def animate(self):
        for i in range(1, 5 * self.path_sampling_rate):
            for primitive in self.primitives:
                d_t = 1 / self.path_sampling_rate
                force = primitive.inverse_mass * jnp.array([0, -9.8, 0])

                if type(primitive) == self.Sphere:
                    if primitive.position_track[-1][1] - primitive.radius < 0:
                        force += -primitive.velocity_track[-1] * (2.0 - 0.1) / d_t

                primitive.velocity_track += (
                    primitive.velocity_track[-1] + d_t * force,
                )
                primitive.position_track += (
                    primitive.position_track[-1] + d_t * primitive.velocity_track[-1],
                )

        for primitive in self.primitives:
            primitive.position_track = jnp.array(primitive.position_track)
            primitive.velocity_track = jnp.array(primitive.velocity_track)

    def __init__(self, aperture_size=0, focal_distance=1):
        self.path_sampling_rate = 20
        self.aperture_size = aperture_size
        self.focal_distance = focal_distance
        self.primitives = [
            self.Sphere(jnp.array([0, 0, 0]), jnp.array([0, 0, 0]), 0, 8.0),
            self.Sphere(jnp.array([-0.3, 0.8, 0.1]), jnp.array([0, 0, 0]), 0.5, 0.2),
            self.Sphere(jnp.array([0.2, 0.5, -0.1]), jnp.array([0, 0, 0]), 0.4, 0.15),
            self.Plane(jnp.array([0, -0.2, 0]), jnp.array([0, 1, 0.1])),
        ]
        self.animate()

    def map(self, position, time, lens_coord):
        def intersect(ray, time):
            def get_best_t(
                primitive,
            ):
                ts = primitive.intersect(ray, time)

                t_best = 1000.0
                for t in ts:

                    def try_t(t, t_best):
                        t = jnp.where(t > 0, t, jnp.nan)
                        t = jnp.where(t < t_best, t, jnp.nan)
                        return jnp.where(jnp.isnan(t), t_best, t)

                    t_best = try_t(t, t_best)

                normal = primitive.get_normal(
                    ray.position + t_best * ray.direction, time
                )

                local_position = primitive.get_local_position(time)

                return t_best, normal, local_position

            ts = []
            normals = []
            local_positions = []
            for primitive in self.primitives:
                t, normal, local_position = get_best_t(primitive)
                ts.append(t)
                normals.append(normal)
                local_positions.append(local_position)

            ts = jnp.array(ts)
            normals = jnp.array(normals)
            local_positions = jnp.array(local_positions)

            id = jnp.argmin(ts)
            t = ts[id]
            normal = normals[id]
            local_position = local_positions[id]

            position = ray.position + t * ray.direction - local_position

            return position, id, normal

        def build_camera_ray(origin, target, pixel_coord):
            direction = target - origin
            direction /= jnp.linalg.norm(direction)

            up = jnp.array([0, 1, 0])
            u0 = jnp.cross(up, direction)
            u0 /= jnp.linalg.norm(u0)

            u1 = jnp.cross(direction, u0)
            u1 /= jnp.linalg.norm(u1)

            direction = 2.0 * direction - pixel_coord[0] * u0 - pixel_coord[1] * u1
            direction = direction / jnp.linalg.norm(direction)

            def disk_point(xi):
                r = jnp.sqrt(xi[0])
                theta = 2 * jnp.pi * xi[1]
                return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta)])

            offset = disk_point(lens_coord)
            lens_point = origin + (offset[0] * u0 + offset[1] * u1) * self.aperture_size
            convergence = origin + self.focal_distance * direction

            ray_origin = lens_point
            ray_direction = convergence - lens_point
            ray_direction /= jnp.linalg.norm(ray_direction)

            return Ray(ray_origin, ray_direction)

        time *= 1.5
        camera_position = jnp.array([-0.5, 0.5, -1]) + 0.5 * time * jnp.array(
            [0.5, 0, 0]
        )
        camera_target = jnp.array([0, 0.1, 0])
        camera_ray = build_camera_ray(camera_position, camera_target, position)
        intersection, id, normal = intersect(camera_ray, time)

        position = intersection

        return {"position": position, "id": id, "normal": normal}


def extract_camera_position_from_view_matrix(m):
    return m[3, :3]


def get_projection_matrix(fov=0.3, aspect_ratio=1.0, near=0.1, far=100.0):
    t = 1.0 / jnp.tan(fov / 2.0)
    return jnp.array(
        [
            [t / aspect_ratio, 0, 0, 0],
            [0, t, 0, 0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, (2 * far * near) / (near - far), 0],
        ]
    )


def get_view_matrix(
    camera_position=None,
    camera_target=None,
    camera_up=None,
):
    if camera_position is None:
        camera_position = jnp.array([1, 2, 3])
    if camera_target is None:
        camera_target = jnp.array([0, 0, 0])
    if camera_up is None:
        camera_up = jnp.array([0, 1, 0])
    z_axis = camera_position - camera_target
    z_axis = z_axis / jnp.linalg.norm(z_axis)
    x_axis = jnp.linalg.cross(camera_up, z_axis)
    y_axis = jnp.linalg.cross(z_axis, x_axis)

    return jnp.array(
        [
            [x_axis[0], y_axis[0], z_axis[0], 0],
            [x_axis[1], y_axis[1], z_axis[1], 0],
            [x_axis[2], y_axis[2], z_axis[2], 0],
            [
                -jnp.dot(x_axis, camera_position),
                -jnp.dot(y_axis, camera_position),
                -jnp.dot(z_axis, camera_position),
                1,
            ],
        ]
    )


def ray_test():
    pixel_coord = jnp.array([10, 20])
    screen_width = 100
    screen_height = 50
    projection_matrix = get_projection_matrix()
    view_matrix = get_view_matrix()

    x_ndc = (2.0 * pixel_coord[0]) / screen_width - 1.0
    y_ndc = 1.0 - (2.0 * pixel_coord[1]) / screen_height
    ndc = jnp.array([x_ndc, y_ndc, -1.0, 1.0])

    inv_proj = jnp.linalg.inv(projection_matrix)
    view_space = inv_proj @ ndc
    view_space /= view_space[3]

    inv_view = jnp.linalg.inv(view_matrix)
    world_space = inv_view @ view_space
    world_space /= world_space[3]

    origin = extract_camera_position_from_view_matrix(view_matrix)
    direction = world_space[:3] - origin
    direction /= jnp.linalg.norm(direction)

    print(origin)
    print(direction)


# ray_test()
