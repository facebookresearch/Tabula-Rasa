# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bias/Variance decomposition experiment for one-pass vs two-pass CountUniqueEstimator.

For each combination of {map, mode, sample_count}, this script:
1. Computes a high-sample-count reference (fixed field seed)
2. Generates R independent estimates (varying sampling seed, same field seed)
3. Decomposes MSE = Bias² + Variance per pixel, then averages over pixels
4. Plots the results as a 2x3 figure

Usage:
    cd python
    python tests/test_bias_variance.py
"""

import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tabula_rasa import (
    BlendedAdaptiveField,
    BoxFilter,
    CountUniqueEstimator,
    RandomSampler,
    RaytracingMap,
    VaryingMap,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FPS = 24
RESOLUTION = (128, 128)
REPETITIONS = 128
REFERENCE_SAMPLE_COUNT = 2**15
SAMPLE_COUNTS = [2**0, 2**1, 2**2, 2**4, 2**6, 2**8, 2**10, 2**12]
BIN_COUNTS = [4, 16, 64, 256]
FIELD_KEY = jax.random.PRNGKey(2026)
TIME = 0.5


def coord_grid(shape):
    x = jnp.arange(shape[1])
    y = jnp.arange(shape[0])
    x, y = jnp.meshgrid(x, y)
    x = jnp.array([x, y])
    x = jnp.transpose(x, (2, 1, 0))
    x += 0.5
    x /= jnp.array([1, shape[0] / shape[1]])
    x /= jnp.array(shape)
    x = x * 2 - 1
    return x


def render_frame(
    estimator, map, field, sampler, filter, shape, batch_size, field_key, sample_key
):
    """Render a single frame and return the value array."""
    estimate_fn = partial(estimator.estimate, map, field, sampler, filter)

    def pixel_fn(position):
        def compute_single_sample(sample_base):
            return estimate_fn(sample_key, sample_base, field_key, position, TIME)

        def accumulate(carry, sample_base):
            sample = compute_single_sample(sample_base)
            return jax.tree.map(lambda c, s: c + s, carry, sample), None

        init = jax.tree.map(
            lambda x: jnp.zeros(x.shape, x.dtype),
            jax.eval_shape(compute_single_sample, 0),
        )
        accumulated, _ = jax.lax.scan(accumulate, init, jnp.arange(batch_size))
        frame = estimator.finish(accumulated)
        return frame["value"]

    positions = coord_grid(shape)
    # Match renderer.spread: vmap with in_axes=0 then in_axes=1
    # positions is [W, H, 2]. After both vmaps, each pixel_fn call gets [2].
    # Output is [H, W, ...] (outer vmap axis=1 → H, inner vmap axis=0 → W).
    f = pixel_fn
    for i in range(len(shape)):
        f = jax.vmap(f, in_axes=i)
    return f(positions)


def compute_reference(map_obj, field, filter, shape):
    """Compute a high-sample-count reference for a given map."""
    print(
        f"    Computing reference (samples={REFERENCE_SAMPLE_COUNT})...",
        end=" ",
        flush=True,
    )
    t0 = time.perf_counter()
    ref_sampler = RandomSampler(1, REFERENCE_SAMPLE_COUNT)
    ref_estimator = CountUniqueEstimator(two_pass=False)
    ref_fn = jax.jit(
        partial(
            render_frame,
            ref_estimator,
            map_obj,
            field,
            ref_sampler,
            filter,
            shape,
            REFERENCE_SAMPLE_COUNT,
        )
    )
    reference = ref_fn(FIELD_KEY, jax.random.PRNGKey(99999))
    jax.block_until_ready(reference)
    print(f"{time.perf_counter() - t0:.1f}s")
    return reference


def run_single_config(map_obj, two_pass, sample_count, bin_count, shape, reference):
    """Run the full bias/variance experiment for one configuration.

    Returns dict with bias_sq, variance, mse scalars.
    """
    field = BlendedAdaptiveField(shape[0])
    filter = BoxFilter(shape, 0, FPS)

    estimator = CountUniqueEstimator(bin_count=bin_count, two_pass=two_pass)

    # Build and JIT the render function
    # RandomSampler: sample_count=1 per call, total=sample_count for the sequence
    sampler = RandomSampler(1, sample_count)

    render_fn = jax.jit(
        partial(
            render_frame,
            estimator,
            map_obj,
            field,
            sampler,
            filter,
            shape,
            sample_count,
        )
    )

    # Warmup / compile
    print(
        f"    Compiling (samples={sample_count}, two_pass={two_pass}, bins={bin_count})...",
        end=" ",
        flush=True,
    )
    t0 = time.perf_counter()
    _ = render_fn(FIELD_KEY, jax.random.PRNGKey(0))
    jax.block_until_ready(_)
    compile_time = time.perf_counter() - t0
    print(f"{compile_time:.1f}s")

    # Repetitions: different sampling seeds, same field seed
    print(f"    Running {REPETITIONS} repetitions...", end=" ", flush=True)
    t0 = time.perf_counter()
    estimates = []
    for r in range(REPETITIONS):
        sample_key = jax.random.fold_in(jax.random.PRNGKey(7777), r)
        est = render_fn(FIELD_KEY, sample_key)
        estimates.append(est)

    # Stack into [R, H, W, ...] and block until ready
    estimates = jnp.stack(estimates)
    jax.block_until_ready(estimates)
    total_time = time.perf_counter() - t0
    avg_time = total_time / REPETITIONS
    print(f"{total_time:.1f}s ({avg_time * 1000:.2f}ms per rep)")

    # Use only the first channel if multi-channel
    ref = reference
    if estimates.ndim == 4:
        estimates = estimates[..., 0]
        ref = ref[..., 0]

    # Decompose
    mean_estimate = jnp.mean(estimates, axis=0)  # [H, W]
    bias_per_pixel = mean_estimate - ref  # [H, W]
    bias_sq = float(jnp.mean(bias_per_pixel**2))

    var_per_pixel = jnp.var(estimates, axis=0)  # [H, W]
    variance = float(jnp.mean(var_per_pixel))

    mse_per_pixel = jnp.mean((estimates - ref[None]) ** 2, axis=0)  # [H, W]
    mse = float(jnp.mean(mse_per_pixel))

    print(
        f"    bias²={bias_sq:.6f}  var={variance:.6f}  mse={mse:.6f}  "
        f"(bias²+var={bias_sq + variance:.6f})"
    )

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "mse": mse,
        "avg_time": avg_time,
        "compile_time": compile_time,
    }


def run_experiment():
    maps = [
        ("VaryingMap", VaryingMap()),
        ("RaytracingMap", RaytracingMap()),
    ]
    modes = [
        ("one_pass", False),
        ("two_pass", True),
    ]

    results = {}

    for map_name, map_obj in maps:
        results[map_name] = {}
        field = BlendedAdaptiveField(RESOLUTION[0])
        filter = BoxFilter(RESOLUTION, 0, FPS)
        print(f"\n--- {map_name}: computing reference ---")
        reference = compute_reference(map_obj, field, filter, RESOLUTION)
        for bc in BIN_COUNTS:
            bc_key = str(bc)
            results[map_name][bc_key] = {}
            for mode_name, two_pass in modes:
                results[map_name][bc_key][mode_name] = {}
                print(f"\n=== {map_name} / bins={bc} / {mode_name} ===")
                for sc in SAMPLE_COUNTS:
                    print(f"  sample_count={sc}")
                    metrics = run_single_config(
                        map_obj, two_pass, sc, bc, RESOLUTION, reference
                    )
                    results[map_name][bc_key][mode_name][str(sc)] = metrics

    return results


def plot_results(results):
    map_names = list(results.keys())
    bin_count_keys = list(results[map_names[0]].keys())
    metric_names = [("bias_sq", "Bias²"), ("variance", "Variance"), ("mse", "MSE")]

    bin_colors = {
        bin_count_keys[0]: "C0",
        bin_count_keys[1]: "C1",
        bin_count_keys[2]: "C2",
        bin_count_keys[3]: "C3",
    }
    mode_styles = {"one_pass": "-", "two_pass": "--"}

    fig, axes = plt.subplots(
        len(map_names),
        len(metric_names),
        figsize=(16, 5 * len(map_names)),
        squeeze=False,
    )

    for row, map_name in enumerate(map_names):
        for col, (metric_key, metric_label) in enumerate(metric_names):
            ax = axes[row, col]

            for bc_key in bin_count_keys:
                for mode_name in ["one_pass", "two_pass"]:
                    mode_data = results[map_name][bc_key][mode_name]
                    xs = [int(sc) for sc in mode_data.keys()]
                    ys = [mode_data[sc][metric_key] for sc in mode_data.keys()]

                    label = f"{mode_name.replace('_', '-')} (bins={bc_key})"
                    ax.plot(
                        xs,
                        ys,
                        mode_styles[mode_name],
                        color=bin_colors[bc_key],
                        marker="o",
                        markersize=4,
                        label=label,
                    )

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Sample count")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{map_name}: {metric_label}")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = "out/bias_variance"
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "bias_variance_decomposition.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {plot_path}")


def plot_performance(results):
    map_names = list(results.keys())
    bin_count_keys = list(results[map_names[0]].keys())

    bin_colors = {
        bin_count_keys[0]: "C0",
        bin_count_keys[1]: "C1",
        bin_count_keys[2]: "C2",
        bin_count_keys[3]: "C3",
    }

    fig, axes = plt.subplots(
        1,
        len(map_names),
        figsize=(8 * len(map_names), 5),
        squeeze=False,
    )

    for col, map_name in enumerate(map_names):
        ax = axes[0, col]

        for bc_key in bin_count_keys:
            one_pass = results[map_name][bc_key]["one_pass"]
            two_pass = results[map_name][bc_key]["two_pass"]
            sc_keys = list(one_pass.keys())
            xs = [int(sc) for sc in sc_keys]
            ratios = []
            for sc in sc_keys:
                t_one = one_pass[sc].get("avg_time")
                t_two = two_pass[sc].get("avg_time")
                if t_one and t_two and t_one > 0:
                    ratios.append(t_two / t_one)
                else:
                    ratios.append(float("nan"))

            ax.plot(
                xs,
                ratios,
                "-",
                color=bin_colors[bc_key],
                marker="o",
                markersize=4,
                label=f"bins={bc_key}",
            )

        ax.axhline(y=2.0, color="gray", linestyle=":", linewidth=1, label="2.0x")
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sample count")
        ax.set_ylabel("Time ratio (two-pass / one-pass)")
        ax.set_title(f"{map_name}: Relative performance overhead")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = "out/bias_variance"
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "performance_ratio.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    print(f"Devices: {jax.devices()}")
    print(f"Resolution: {RESOLUTION}")
    print(f"Repetitions: {REPETITIONS}")
    print(f"Sample counts: {SAMPLE_COUNTS}")
    print(f"Bin counts: {BIN_COUNTS}")
    print(f"Reference sample count: {REFERENCE_SAMPLE_COUNT}")

    results = run_experiment()

    # Save raw data
    out_dir = "out/bias_variance"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    plot_results(results)
    plot_performance(results)
