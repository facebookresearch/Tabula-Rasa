// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// =============================================================================
// Common Shader Definitions
// Structs shared between the mesh rasterizer and the shaders that read its
// G-buffer. Params must stay byte-identical to its copies in noise.wgsl and
// tonemap.wgsl.
// =============================================================================

#pragma once

// =============================================================================
// Structs
// =============================================================================

struct Params {
    resolutionX: f32,
    resolutionY: f32,
    fieldResolution: f32,
    sampleCount: f32,
    seed: f32,
    outputMode: f32,      // 0 = Final Noise; 3-9 select a G-buffer channel
    fieldType: f32,       // 0 = White, 1 = Blended Adaptive, 2 = Adaptive
    rngType: f32,         // 0 = Threefry (JAX-compatible), 1 = PCG (fast)
    estimatorMode: f32,   // 0 = Biased (N samples), 1 = Unbiased (2N samples)
    histogramBinCount: f32,     // Number of bins for the weighted histogram (1-256)
    meshScale: f32,
    // Mesh normalization, so skinned vertices land in the same space as static ones
    meshNormCenterX: f32,
    meshNormCenterY: f32,
    meshNormCenterZ: f32,
    meshNormScale: f32,
    centerBackgroundCube: f32, // Whether background geometry is centered within field cells
}

struct Camera {
    position: vec3<f32>,
    _pad0: f32,
    direction: vec3<f32>,
    _pad1: f32,
    right: vec3<f32>,
    _pad2: f32,
    up: vec3<f32>,
    fov: f32,
}

struct FragmentOutput {
    @location(0) position: vec4<f32>,
    @location(1) footprint: vec4<f32>,
}
