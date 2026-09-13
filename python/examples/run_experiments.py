# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Experiments from the paper.

Each experiment is a function below; pick one on the command line:

    python run_experiments.py suite
    python run_experiments.py convergence
    python run_experiments.py performance
    python run_experiments.py flows
    python run_experiments.py bin-counts

Input is read from ./in and output is written to ./out, both relative to the
directory you run this from.
"""

import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from tabula_rasa import (
    BlendedAdaptiveField,
    BoxFilter,
    Config,
    CountUniqueEstimator,
    HammersleySampler,
    OpticalFlowMap,
    RandomSampler,
    RaytracingMap,
    Renderer,
    VaryingMap,
    WhiteField,
)

# can this be part of the Config?
fps = 24


def test_suit():
    main_resolution = 480, 720
    batch_size = 1
    sample_count = 10**2
    field = BlendedAdaptiveField(main_resolution[0] // 1)
    sampler = HammersleySampler(sample_count, sample_count * batch_size)
    filter = BoxFilter(main_resolution, 0, fps)

    configs = []

    configs.append(
        Config(
            "varying",
            CountUniqueEstimator(),
            VaryingMap(),
            field,
            sampler,
            filter,
        )
    )

    renderer = Renderer()
    for config in configs:
        renderer.render(
            config,
            main_resolution,
            2 * fps,
            batch_size=batch_size,
            variant_count=1,
            sampling_count=1,
            ship=True,
            use_visualization=True,
        )




def test_convergence():
    # main_resolution = 512, 512
    main_resolution = 480, 720
    # frame_count = 4
    frame_count = 1 * 20
    estimator = CountUniqueEstimator()
    # map = VaryingMap()
    map = RaytracingMap()
    # map = OpticalFlowMap("in/flow/cotracker.safetensors")
    field = BlendedAdaptiveField(main_resolution[0] // 1)
    # field = WhiteField(main_resolution[0] * 4)
    filter = BoxFilter(main_resolution, 0, fps)
    renderer = Renderer()
    reference_sample_count = 2**16

    config = Config(
        "convergence_raytracing_separate/reference",
        estimator,
        map,
        field,
        # HammersleySampler(1, reference_sample_count),
        RandomSampler(1, reference_sample_count),
        # LPGKSampler(1, reference_sample_count),
        filter,
    )
    renderer.render(
        config,
        main_resolution,
        frame_count=frame_count,
        batch_size=2**16,
        variant_count=1,
        sampling_count=1,
        ship=True,
        use_visualization=True,
        dump_exr_zip=True,
    )
    for i in range(0, 16):
        batch_size = 2**i
        config = Config(
            "convergence_raytracing_separate/" + str(i),
            estimator,
            map,
            field,
            # HammersleySampler(1, batch_size),
            RandomSampler(1, batch_size),
            # LPGKSampler(1, reference_sample_count),
            filter,
        )
        renderer.render(
            config,
            main_resolution,
            frame_count=frame_count,
            batch_size=batch_size,
            variant_count=1,
            sampling_count=1,
            ship=True,
            use_visualization=True,
            dump_exr_zip=True,
        )


def analyze_convergence():
    name = "CountUnique_Raytracing_White_Hammersley_Box_0_0"

    def load(iteration):
        path = "out/convergence/"
        reference = load_file(path + iteration + "/" + name + ".safetensors")
        result = reference["value"]
        result = jnp.nan_to_num(result, nan=0.0)
        return result

    reference = load("reference")
    print(reference.shape)

    log = []
    for i in range(0, 16):
        iteration = load(str(i))
        error = jnp.mean(jnp.abs(iteration - reference))
        print(error)
        log.append(error)

    plt.plot(log)
    plt.savefig("out/convergence/" + name + "convergence.jpg")
    plt.close()


# analyze_convergence()



def make_plan(effective_sample_count):
    batch_size = 1
    while effective_sample_count > 16:
        effective_sample_count = effective_sample_count // 2
        batch_size *= 2
    return effective_sample_count, batch_size


def test_performance():
    resolutions = 256, 512, 1024
    bin_counts = 1, 16, 256
    sample_counts = 16, 256, 1024
    maps = (
        RaytracingMap(),
        VaryingMap(),
        OpticalFlowMap("in/flow/cotracker.safetensors"),
    )

    renderer = Renderer()

    for resolution in resolutions:
        fields = (
            BlendedAdaptiveField(resolution),
            # AdaptiveField(resolution),
            WhiteField(4 * resolution),
        )
        filter = BoxFilter((resolution, resolution), 0, fps)
        for bin_count in bin_counts:
            estimator = CountUniqueEstimator(bin_count)
            for sample_count in sample_counts:
                effective_sample_count, batch_size = make_plan(sample_count)
                sampler = HammersleySampler(effective_sample_count, sample_count)
                print(
                    "Plan is",
                    effective_sample_count,
                    "samples in",
                    batch_size,
                    "batches",
                )
                for map in maps:
                    for field in fields:
                        print(
                            resolution,
                            estimator.bin_count,
                            sample_count,
                            type(map).__name__,
                            type(field).__name__,
                        )
                        # continue
                        config = Config(
                            "performance",
                            estimator,
                            map,
                            field,
                            sampler,
                            filter,
                        )

                        renderer.render(
                            config,
                            (resolution, resolution),
                            frame_count=24,
                            batch_size=batch_size,
                            variant_count=1,
                            sampling_count=1,
                            ship=True,
                            use_visualization=True,
                        )




def test_flows():
    def enumerate_safetensor_files(base_dir="in/flow"):
        safetensor_files = []
        for root, _dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".safetensors"):
                    safetensor_files.append(os.path.join(root, file))
        return safetensor_files

    main_resolution = 120, 180
    batch_size = 10
    sample_count = 10**2
    field = BlendedAdaptiveField(main_resolution[0] // 1)
    sampler = HammersleySampler(sample_count, sample_count * batch_size)
    filter = BoxFilter(main_resolution, 0, fps)
    renderer = Renderer()

    filenames = enumerate_safetensor_files()
    for filename in filenames:
        name = os.path.dirname(filename).removeprefix("in/")

        print(filename)
        print(name)

        # continue

        config = Config(
            name,
            CountUniqueEstimator(),
            OpticalFlowMap(filename),
            field,
            sampler,
            filter,
        )
        renderer.render(
            config,
            main_resolution,
            2 * fps,
            batch_size=batch_size,
            variant_count=1,
            sampling_count=1,
            ship=True,
            use_visualization=True,
            dump_exr_zip=True,
        )




def test_bin_counts():
    # main_resolution = 480, 720
    main_resolution = 512, 512
    # frame_count = 1 * 20
    frame_count = 4
    sample_count = 1024

    # map = VaryingMap()
    # map = RaytracingMap()
    map = OpticalFlowMap("in/flow/cotracker.safetensors")
    # field = WhiteField(main_resolution[0] * 4)
    field = BlendedAdaptiveField(main_resolution[0] // 1)
    # sampler = HammersleySampler(1, sample_count)
    sampler = RandomSampler(1, sample_count)
    filter = BoxFilter(main_resolution, 0, fps)
    renderer = Renderer()

    # Test bin counts from 1 to 2048 (powers of 2: 2^0 to 2^11)
    for i in range(12):
        bin_count = 2**i
        estimator = CountUniqueEstimator(bin_count)

        config = Config(
            f"bin_count_opticalflow_random/{bin_count}",
            estimator,
            map,
            field,
            sampler,
            filter,
        )

        print(f"Testing bin_count={bin_count}")

        renderer.render(
            config,
            main_resolution,
            frame_count=frame_count,
            batch_size=sample_count,
            variant_count=1,
            sampling_count=1,
            ship=True,
            use_visualization=True,
            dump_exr_zip=True,
        )


EXPERIMENTS = {
    "suite": test_suit,
    "convergence": test_convergence,
    "analyze-convergence": analyze_convergence,
    "performance": test_performance,
    "flows": test_flows,
    "bin-counts": test_bin_counts,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS), help="which experiment to run")
    args = parser.parse_args()
    EXPERIMENTS[args.experiment]()


if __name__ == "__main__":
    main()
