# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tabula Rasa: Monte Carlo estimation of Gaussian noise with unit variance and
controlled spatio-temporal correlation.

The pipeline is a five-layer composition, assembled with :class:`Config`:

    Sampler -> Filter -> Map -> Field -> Estimator -> variance-corrected noise

- **Samplers** generate 5D sample points (2D position, time, 2D lens).
- **Filters** scale those sample coordinates by the pixel size.
- **Maps** turn 2D screen coordinates into the world-space positions that define
  the correlation structure.
- **Fields** supply hash-based procedural noise with MIP-level selection.
- **Estimators** restore unit variance after the weighted combination.
"""

from collections import namedtuple

from .adaptive_field import AdaptiveField
from .blended_adaptive_field import BlendedAdaptiveField
from .box_filter import BoxFilter
from .count_unique_estimator import CountUniqueEstimator
from .ensemble_estimator import EnsembleEstimator
from .estimator import Estimator
from .field import Field
from .filter import Filter
from .gauss_filter import GaussFilter
from .grid_sampler import GridSampler
from .hammersley_sampler import HammersleySampler
from .hard_adaptive_field import HardAdaptiveField
from .image_field import ImageField
from .jittered_sampler import JitteredSampler
from .lpgk_sampler import LPGKSampler
from .map import Map
from .optical_flow_map import OpticalFlowMap
from .particle_map import ParticleMap
from .random_sampler import RandomSampler
from .raytracing_map import RaytracingMap
from .renderer import Renderer
from .sampler import Sampler
from .tent_filter import TentFilter
from .timer import Timer
from .uniform_map import UniformMap
from .varying_map import VaryingMap
from .white_field import WhiteField

#: One rendering configuration: a name plus one component per pipeline layer.
Config = namedtuple("Config", "name estimator map field sampler filter")

__all__ = [
    "Config",
    # Base classes, one per layer
    "Sampler",
    "Filter",
    "Map",
    "Field",
    "Estimator",
    # Samplers
    "GridSampler",
    "HammersleySampler",
    "JitteredSampler",
    "LPGKSampler",
    "RandomSampler",
    # Filters
    "BoxFilter",
    "GaussFilter",
    "TentFilter",
    # Maps
    "OpticalFlowMap",
    "ParticleMap",
    "RaytracingMap",
    "UniformMap",
    "VaryingMap",
    # Fields
    "AdaptiveField",
    "BlendedAdaptiveField",
    "HardAdaptiveField",
    "ImageField",
    "WhiteField",
    # Estimators
    "CountUniqueEstimator",
    "EnsembleEstimator",
    # Rendering
    "Renderer",
    "Timer",
]
