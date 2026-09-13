# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import time
import zipfile
from functools import partial

from matplotlib import pyplot as plt

os.environ["NCCL_DEBUG"] = "ERROR"

# import imath
import jax
import jax.lax
import jax.numpy as jnp
import mediapy as media
import numpy as np

# import OpenEXR
from safetensors.flax import save_file
from tqdm import trange

from .timer import Timer


def save_frame_as_exr(frame_data: np.ndarray, filepath: str) -> None:
    """Save a frame as OpenEXR with 32-bit float and lossless ZIP compression.

    Args:
        frame_data: HxW or HxWxC numpy array with HDR float values (can be negative).
        filepath: Output path for the EXR file.
    """
    frame_data = np.asarray(frame_data, dtype=np.float32)

    if frame_data.ndim == 2:
        height, width = frame_data.shape
        num_channels = 1
    else:
        height, width, num_channels = frame_data.shape

    header = OpenEXR.Header(width, height)
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)

    float_channel = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))

    if num_channels == 1:
        header["channels"] = {"Y": float_channel}
        channel_data = {"Y": frame_data.tobytes()}
    else:
        channel_names = ["R", "G", "B", "A"][:num_channels]
        header["channels"] = {name: float_channel for name in channel_names}
        channel_data = {
            name: np.ascontiguousarray(frame_data[:, :, i]).tobytes()
            for i, name in enumerate(channel_names)
        }

    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels(channel_data)
    out.close()


def dump_frames_to_exr_zip(frames: list, zip_path: str) -> None:
    """Dump a list of frames as OpenEXR images into a zip archive.

    Args:
        frames: List of frame dictionaries, each containing a "value" key with pixel data.
        zip_path: Output path for the zip archive.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(frames):
            frame_data = np.asarray(frame["value"])
            exr_filename = f"frame_{i:06d}.exr"
            exr_path = os.path.join(temp_dir, exr_filename)
            save_frame_as_exr(frame_data, exr_path)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for i in range(len(frames)):
                exr_filename = f"frame_{i:06d}.exr"
                exr_path = os.path.join(temp_dir, exr_filename)
                zf.write(exr_path, exr_filename)


def compute_histogram(data):
    data = jnp.ravel(data)
    bin_count = 128
    height = bin_count
    x = jnp.linspace(0, 1, bin_count)
    y = jnp.linspace(0, 1, height)
    g = jnp.array(jnp.meshgrid(x, y))
    g = jnp.transpose(g, (1, 2, 0))

    histo = jnp.linspace(-3, 3, bin_count)

    histo = jnp.repeat(histo[None, ...], data.shape[0], axis=0)
    histo -= data[..., None]
    histo = jnp.exp(-((20 * histo) ** 2))
    histo = jnp.mean(histo, axis=0) * 10
    result = jnp.where(g[:, :, 1] < 1 - histo[:], 0.0, 1)
    return result


def shapiro_wilk_test(data, max_samples=5000):
    data = jnp.ravel(data)
    n = max_samples
    data = data[:n]
    sorted_data = jnp.sort(data)

    mean = jnp.mean(sorted_data)
    ss = jnp.sum((sorted_data - mean) ** 2)

    def compute_m_values(n):
        i = jnp.arange(1, n + 1)
        m = jax.scipy.stats.norm.ppf((i - 0.375) / (n + 0.25))
        return m

    m = compute_m_values(n)

    m_squared_sum = jnp.sum(m**2)
    a = m / jnp.sqrt(m_squared_sum)
    half_n = n // 2
    indices = jnp.arange(half_n)
    a_coeffs = a[n - 1 - indices] - a[indices]
    x_diff = sorted_data[n - 1 - indices] - sorted_data[indices]
    b = jnp.sum(a_coeffs * x_diff)
    w = (b**2) / ss
    w = jnp.clip(w, 0.0, 1.0)
    return w


def make_empty_tree(f, *args):
    return jax.tree.map(
        lambda s: jnp.empty(s.shape, dtype=s.dtype),
        jax.eval_shape(f, *args),
    )


def write_performance_json(
    avg_compute_per_frame,
    resolution,
    bin_count,
    sample_count,
    map_class_name,
    field_class_name,
):
    os.makedirs("out", exist_ok=True)
    json_path = "out/performance.json"

    # Load existing data if file exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    key = f"{resolution}_{bin_count}_{sample_count}_{map_class_name}_{field_class_name}"
    data[key] = avg_compute_per_frame
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


class Renderer:
    def analayze_frame(self, frame):
        frame = frame["value"]
        histogram = compute_histogram(frame)
        shapiro_wilk_p = shapiro_wilk_test(frame)

        return {"shapiro_wilk_p": shapiro_wilk_p}

    def ship_result(self, path, frames, name):
        # wanted_names = ["value", "id", "position", "normal"]
        wanted_names = ["value"]
        frame_dict = {
            wanted_name: jnp.stack([frame[wanted_name] for frame in frames])
            for wanted_name in wanted_names
        }
        os.makedirs("out", exist_ok=True)
        save_file(frame_dict, path + name + ".safetensors")

    def visualize(self, frame):
        def draw_histogram(data):
            result = compute_histogram(data)
            result = jnp.repeat(result[..., None], 3, axis=-1)
            return result

        def draw_spectrum(image):
            def blur(image, axis, taps=1):
                result = 0
                weights = jnp.array([1, 2, 1]) / 4
                for i in range(-taps, taps + 1):
                    result += weights[i + taps] * jnp.roll(image, i, axis=axis)
                return result

            def downsample(image, steps=1):
                if steps == 0:
                    return image
                if steps > 1:
                    image = downsample(image, steps - 1)
                image = blur(image, -1)[:, ::2]
                image = blur(image, -2)[::2, :]
                return image

            spectrum = jnp.fft.fft2(image[..., 0])
            power = jnp.abs(spectrum)
            power = downsample(power, 3)
            power = power / jnp.percentile(power, 99)
            return jnp.repeat(power[..., None], 3, axis=-1)

        def draw_histogram_pair(frame):
            h0 = draw_histogram(frame)
            h1 = draw_histogram(jax.random.normal(jax.random.PRNGKey(1), (100000,)))
            return h0 * 0.5 + h1 * 0.5

        def insert(target, inset, position):
            if target.shape[0] < 256:
                return target
            target = target.at[
                position[0] : position[0] + inset.shape[0],
                position[1] : position[1] + inset.shape[1],
            ].set(inset)
            return target

        frame = frame["value"]

        # frame = jnp.flip(frame, axis =)
        if frame.ndim == 2:
            frame = jnp.repeat(frame[..., None], 3, axis=-1)
        elif frame.ndim == 3:
            frame = frame[:, :, :3]
        else:
            print("Oerroer")

        # if jnp.any(jnp.isnan(frame)):
        #  print("Frame with nans")
        frame = jnp.nan_to_num(frame)

        tonemapped = jnp.clip(frame / 3 / 2 + 0.5, 0, 1)
        # tonemapped = jnp.clip(frame / 20, 0, 1)
        result = tonemapped

        def up_to(x, resolution):
            up = resolution // x.shape[0]
            if up > 1:
                x = jnp.repeat(x, up, axis=0)
                x = jnp.repeat(x, up, axis=1)
            return x

        result = up_to(result, 512)

        if False:
            gap = 10
            side = 200

            canvas = (
                jnp.ones((result.shape[0], result.shape[1] + side + 2 * gap, 3)) / 2
            )
            result = insert(canvas, result, (0, 0))

            histo_vis = draw_histogram_pair(jnp.ravel(frame))
            result = insert(
                result, histo_vis, (gap, result.shape[1] - side - 2 * gap + 10)
            )

            spectrum = draw_spectrum(frame)
            spectrum = up_to(spectrum, 128)
            result = insert(
                result,
                spectrum,
                (gap + side + gap, result.shape[1] - side - 2 * gap + 10),
            )

        return result

    def coord_grid(self, shape):
        if len(shape) == 2:
            x = jnp.arange(shape[1])
            y = jnp.arange(shape[0])
            x, y = jnp.meshgrid(x, y)
            x = jnp.array([x, y])
            x = jnp.transpose(x, (2, 1, 0))
        else:
            x = jnp.arange(shape[0])[..., None]
        x += 0.5
        x /= jnp.array([1, shape[0] / shape[1]])
        x /= jnp.array(shape)
        x = x * 2 - 1
        return x

    def spread(self, f, shape, sample_key, sample_base, field_key, time):
        positions = self.coord_grid(shape)
        for i in range(len(shape)):
            f = jax.vmap(f, in_axes=(None, None, None, i, None))
        return f(sample_key, sample_base, field_key, positions, time)

    def get_description(self, config):
        def get_name(c):
            s = type(c).__name__
            for i in range(len(s) - 1, 0, -1):
                if s[i].isupper():
                    return s[:i]

        return (
            get_name(config.estimator)
            + "_"
            + get_name(config.map)
            + "_"
            + get_name(config.field)
            + "_"
            + get_name(config.sampler)
            + "_"
            + get_name(config.filter)
        )

    def render(
        self,
        config,
        shape,
        frame_count,
        batch_size=1,
        variant_count=1,
        sampling_count=1,
        ship=True,
        use_visualization=True,
        dump_exr_zip=False,
    ):
        # Check if the JSON key already exists - skip compute if so
        json_path = "out/performance.json"
        key = f"{shape}_{config.estimator.bin_count}_{config.sampler.sample_count}_{type(config.map).__name__}_{type(config.field).__name__}"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing_data = json.load(f)
            if key in existing_data:
                print(f"Skipping render: key '{key}' already exists in {json_path}")
                return

        def make_frame(estimate, frame_index, variant_index, sampling_index):
            def compute_single_sample(sample_base):
                time_coord = frame_index / fps

                main_field_key = jax.random.PRNGKey(2333)
                variant_field_key = jax.random.fold_in(main_field_key, variant_index)

                main_sample_key = jax.random.PRNGKey(3545)
                sample_key = jax.random.fold_in(main_sample_key, sampling_index)

                return estimate(sample_key, sample_base, variant_field_key, time_coord)

            def accumulate_sample(carry, sample_base):
                sample = compute_single_sample(sample_base)
                new_carry = jax.tree.map(lambda c, s: c + s, carry, sample)
                return new_carry, None

            init_carry = jax.tree.map(
                lambda x: jnp.zeros(x.shape, x.dtype),
                jax.eval_shape(compute_single_sample, 0),
            )

            frame, _ = jax.lax.scan(
                accumulate_sample, init_carry, jnp.arange(batch_size)
            )
            frame = config.estimator.finish(frame)
            return frame

        description = self.get_description(config)
        path = "out/" + config.name + "/"
        os.makedirs(path, exist_ok=True)

        print("Rendering '" + description + "'")

        fps = 24
        use_chunks = True
        if use_chunks:
            frame_chunk_size = jax.local_device_count()
        else:
            frame_chunk_size = 1
        print("Rendering with " + str(frame_chunk_size) + " frames per chunk")

        print("Compiling '" + description + "'")

        compile_timer = Timer("Compile")

        estimate = partial(config.estimator.estimate, *config[2:])
        estimate = partial(self.spread, estimate, shape)
        estimate = partial(make_frame, estimate)
        estimate = jax.pmap(estimate, in_axes=(0, None, None))
        visualize = jax.pmap(self.visualize)
        analyze = jax.pmap(self.analayze_frame)

        empty_index_range = jnp.arange(0, frame_chunk_size)
        empty_estimte_parameters = empty_index_range, 0, 0
        empty_frame = make_empty_tree(estimate, *empty_estimte_parameters)
        jax.block_until_ready(estimate(*empty_estimte_parameters))
        jax.block_until_ready(visualize(empty_frame))
        jax.block_until_ready(analyze(empty_frame))

        compile_timer.report()

        for variant_index in range(0, variant_count):
            render_timer = Timer("Render")
            visualization = []
            result = []
            compute_times = []
            transfer_times = []
            analysis_log = make_empty_tree(analyze, empty_frame)

            for sampling_index in range(0, sampling_count):
                chunk_count = frame_count // frame_chunk_size
                for frame_chunk_index in trange(chunk_count, ncols=80):
                    frame_indices = jnp.arange(
                        frame_chunk_index * frame_chunk_size,
                        frame_chunk_index * frame_chunk_size + frame_chunk_size,
                    )

                    compute_start = time.time()
                    chunk = estimate(frame_indices, variant_index, sampling_index)
                    jax.block_until_ready(chunk)
                    compute_times.append(time.time() - compute_start)

                    transfer_start = time.time()

                    chunk_analysis = analyze(chunk)
                    analysis_log = jax.tree.map(
                        lambda x, y: jnp.concatenate([x, y]),
                        chunk_analysis,
                        analysis_log,
                    )

                    if use_visualization:
                        chunk_visualizations = visualize(chunk)
                        for chunk_visualization in chunk_visualizations:
                            visualization.append(chunk_visualization)

                    if ship:
                        for chunk_index in range(0, frame_chunk_size):
                            frame = jax.tree.map(lambda x: x[chunk_index], chunk)
                            result.append(frame)

                    transfer_times.append(time.time() - transfer_start)

                render_timer.report()

                avg_compute_per_chunk = 1000 * sum(compute_times) / chunk_count
                avg_transfer_per_chunk = 1000 * sum(transfer_times) / chunk_count
                avg_compute_per_frame = avg_compute_per_chunk / frame_chunk_size
                avg_transfer_per_frame = avg_transfer_per_chunk / frame_chunk_size

                write_performance_json(
                    avg_compute_per_frame=avg_compute_per_frame,
                    resolution=shape[0],
                    bin_count=config.estimator.bin_count,
                    sample_count=config.sampler.sample_count * batch_size,
                    map_class_name=type(config.map).__name__,
                    field_class_name=type(config.field).__name__,
                )

                print(
                    f"Compute: {sum(compute_times):.3f}s total, "
                    f"{avg_compute_per_chunk:.2f}ms avg per chunk, "
                    f"{avg_compute_per_frame:.2f}ms avg per frame"
                )
                print(
                    f"Transfer: {sum(transfer_times):.3f}s total, "
                    f"{avg_transfer_per_chunk:.2f}ms avg per chunk, "
                    f"{avg_transfer_per_frame:.2f}ms avg per frame"
                )

            post_timer = Timer("Postprocessing")

            variant_description = (
                description + "_" + str(variant_index) + "_" + str(sampling_index)
            )

            if ship:
                self.ship_result(
                    path,
                    result,
                    variant_description,
                )

            if use_visualization:
                # codec = 'gif'
                codec = "h264"
                media.write_video(
                    path + variant_description + ".mp4",
                    visualization,
                    fps=fps,
                    codec=codec,
                )

                if dump_exr_zip and result:
                    exr_zip_path = path + variant_description + "_frames.zip"
                    print(f"Dumping {len(result)} frames to {exr_zip_path}")
                    dump_frames_to_exr_zip(result, exr_zip_path)

            post_timer.report()

            if False:
                plt.figure()
                plt.plot(analysis_log["shapiro_wilk_p"])
                plt.savefig(path + variant_description + "_shapiro_wilk_p.jpg")
                plt.close()
