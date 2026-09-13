// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// =============================================================================
// Tabula Rasa - WebGPU compute shader
// Pass 2: Count-Unique Estimator with histogram-based variance reduction
// Reads position and footprint from rasterizer pass, outputs noise
// =============================================================================

// Override constant for adaptive workgroup sizing
// Can be set at pipeline creation time: 32, 64, 128, 256, 512, or 1024
override WORKGROUP_SIZE: u32 = 256;

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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba32float, write>;      // Noise channels 0-3
@group(0) @binding(2) var positionTexture: texture_2d<f32>;    // From rasterizer: XY = warped pos, ZW = original pos
@group(0) @binding(3) var footprintTexture: texture_2d<f32>;   // From rasterizer: X = footprint, YZW = derivatives
@group(0) @binding(7) var outputTexture1: texture_storage_2d<rgba32float, write>;     // Noise channels 4-7
@group(0) @binding(8) var outputTexture2: texture_storage_2d<rgba32float, write>;     // Noise channels 8-11
@group(0) @binding(9) var outputTexture3: texture_storage_2d<rgba32float, write>;     // Noise channels 12-15

// Camera uniform for world-space position reconstruction
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
@group(0) @binding(5) var<uniform> camera: Camera;

// Generate a ray for a given pixel position using the camera
fn generateRayForPixel(pixelPos: vec2<f32>) -> Ray {
    var ray: Ray;
    ray.origin = camera.position;

    let aspect = params.resolutionX / params.resolutionY;
    let tanHalfFov = tan(camera.fov * 0.5);

    let rayDirLocal = normalize(vec3<f32>(
        pixelPos.x * tanHalfFov * aspect,
        pixelPos.y * tanHalfFov,
        1.0
    ));

    ray.direction = normalize(
        rayDirLocal.x * camera.right +
        rayDirLocal.y * camera.up +
        rayDirLocal.z * camera.direction
    );

    return ray;
}

// Ray structure used for world-space position reconstruction
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

// =============================================================================
// Python DOF Raytracing Functions (matching rasterizer.wgsl)
// =============================================================================

// Get animated center for a sphere by indexing into pre-computed position track

// Get local_position (displacement from initial) for a primitive


// Python plane intersection

// Python camera ray generation with DOF

// Full Python scene intersection for noise shader

const PI: f32 = 3.14159265359;

// Value type for 16-channel noise (stored as 4 vec4s for efficiency)
struct Value16 {
    v0: vec4<f32>,  // channels 0-3
    v1: vec4<f32>,  // channels 4-7
    v2: vec4<f32>,  // channels 8-11
    v3: vec4<f32>,  // channels 12-15
}

fn value16Zero() -> Value16 {
    var v: Value16;
    v.v0 = vec4<f32>(0.0);
    v.v1 = vec4<f32>(0.0);
    v.v2 = vec4<f32>(0.0);
    v.v3 = vec4<f32>(0.0);
    return v;
}

fn value16Add(a: Value16, b: Value16) -> Value16 {
    var v: Value16;
    v.v0 = a.v0 + b.v0;
    v.v1 = a.v1 + b.v1;
    v.v2 = a.v2 + b.v2;
    v.v3 = a.v3 + b.v3;
    return v;
}

fn value16Scale(a: Value16, s: f32) -> Value16 {
    var v: Value16;
    v.v0 = a.v0 * s;
    v.v1 = a.v1 * s;
    v.v2 = a.v2 * s;
    v.v3 = a.v3 * s;
    return v;
}

fn value16Lerp(a: Value16, b: Value16, t: f32) -> Value16 {
    var v: Value16;
    v.v0 = mix(a.v0, b.v0, t);
    v.v1 = mix(a.v1, b.v1, t);
    v.v2 = mix(a.v2, b.v2, t);
    v.v3 = mix(a.v3, b.v3, t);
    return v;
}

// Workgroup shared memory for the per-pixel weighted histogram.
// A separate array per LOD level (matching Algorithm 2 from the paper) prevents bin
// collisions between levels, giving a more accurate variance estimate.
var<workgroup> weightedHistogram0: array<atomic<u32>, 256>;  // Coarser level (level L)
var<workgroup> weightedHistogram1: array<atomic<u32>, 256>;  // Finer level (level L+1)
// Sized at 256 so Value16 storage fits the 32KB workgroup memory limit:
// 256×4 + 256×4 + 256×64 = 18KB
var<workgroup> sampleValues: array<Value16, 256>;  // Full 16-channel storage

const WEIGHT_SCALE: f32 = 1024.0;  // Fixed-point scale for weighted histogram

// =============================================================================
// Threefry2x32 counter-based PRNG
// =============================================================================
// Algorithm from:
//   J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw, "Parallel Random
//   Numbers: As Easy as 1, 2, 3", Proc. SC'11.
// Reference implementation: Random123,
//   Copyright (c) 2010-2011, D. E. Shaw Research. All rights reserved.
//   Licensed under BSD-3-Clause; full text in THIRD_PARTY_NOTICES.md.
//
// The rotation constants and SKEIN_KS_PARITY originate in the Threefish/Skein
// block cipher on which Threefry is based.
//
// MODIFIED FROM JAX (Apache-2.0):
//   Copyright 2021 The JAX Authors.
//   Licensed under the Apache License, Version 2.0. A copy is included at
//   third_party_licenses/Apache-2.0.txt; you may also obtain one at
//   https://www.apache.org/licenses/LICENSE-2.0
//
//   SKEIN_KS_PARITY, getRotation and threefry2x32 below are a translation into
//   WGSL of the Threefry-2x32 implementation in jax/_src/prng.py
//   (_threefry2x32_lowering, rotate_list), reproducing its round structure,
//   rotation schedule and key-schedule parity so this shader matches jax.random
//   bit-for-bit. Changes: translated from Python/NumPy to WGSL while retaining
//   a 20-round loop; JAX tracing and lowering machinery removed.
//   No other behavioural change is intended. See THIRD_PARTY_NOTICES.md §1.
//
// 20 rounds, ~100 ops per call.
// =============================================================================

const SKEIN_KS_PARITY: u32 = 0x1BD11BDAu;

fn getRotation(round: u32) -> u32 {
    switch (round % 8u) {
        case 0u: { return 13u; }
        case 1u: { return 15u; }
        case 2u: { return 26u; }
        case 3u: { return 6u; }
        case 4u: { return 17u; }
        case 5u: { return 29u; }
        case 6u: { return 16u; }
        case 7u: { return 24u; }
        default: { return 13u; }
    }
}

fn rotl32(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
    let ks0 = key.x;
    let ks1 = key.y;
    let ks2 = ks0 ^ ks1 ^ SKEIN_KS_PARITY;

    var x0 = ctr.x + ks0;
    var x1 = ctr.y + ks1;

    for (var r = 0u; r < 20u; r++) {
        x0 = x0 + x1;
        x1 = rotl32(x1, getRotation(r)) ^ x0;

        if ((r + 1u) % 4u == 0u) {
            let subkey = (r + 1u) / 4u;
            switch (subkey % 3u) {
                case 0u: { x0 = x0 + ks0; x1 = x1 + ks1 + subkey; }
                case 1u: { x0 = x0 + ks1; x1 = x1 + ks2 + subkey; }
                case 2u: { x0 = x0 + ks2; x1 = x1 + ks0 + subkey; }
                default: {}
            }
        }
    }

    return vec2<u32>(x0, x1);
}

// =============================================================================
// PCG 32-bit hash
// =============================================================================
// Copyright (c) 2014 M. E. O'Neill / pcg-random.org
// Licensed under the Apache License, Version 2.0. A copy is included at
// third_party_licenses/Apache-2.0.txt; you may also obtain one at
// https://www.apache.org/licenses/LICENSE-2.0
//
// MODIFIED: translated from GLSL to WGSL and renamed from `pcg` to `pcgHash`.
// Constants, shift schedule and operation order are unchanged.
//
// See THIRD_PARTY_NOTICES.md §2.
//
// ~6 ops per call, sufficient statistical quality for graphics.
// =============================================================================

fn pcgHash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate two 32-bit random values from seed

// PCG-based key generation
fn pcgKey(seed: u32) -> vec2<u32> {
    return vec2<u32>(pcgHash(seed), pcgHash(seed ^ 0x9E3779B9u));
}

// PCG fold-in: mix additional data into key
fn pcgFoldIn(key: vec2<u32>, data: u32) -> vec2<u32> {
    let mixed = pcgHash(key.x ^ data);
    return vec2<u32>(mixed, pcgHash(mixed ^ key.y));
}

// PCG random bits from state
fn pcgRandomBits(key: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(pcgHash(key.x ^ key.y), pcgHash(key.y ^ (key.x * 1664525u)));
}

// =============================================================================
// Unified PRNG Interface - selects between Threefry and PCG
// =============================================================================

fn prngKey(seed: u32) -> vec2<u32> {
    if (params.rngType > 0.5) {
        return pcgKey(seed);
    }
    return threefry2x32(vec2<u32>(0u, 0u), vec2<u32>(0u, seed));
}

fn foldIn(key: vec2<u32>, data: u32) -> vec2<u32> {
    if (params.rngType > 0.5) {
        return pcgFoldIn(key, data);
    }
    return threefry2x32(key, vec2<u32>(data, 0u));
}

fn randomBits(key: vec2<u32>) -> vec2<u32> {
    if (params.rngType > 0.5) {
        return pcgRandomBits(key);
    }
    return threefry2x32(key, vec2<u32>(0u, 0u));
}


fn randomUniform2(key: vec2<u32>) -> vec2<f32> {
    let bits = randomBits(key);
    return vec2<f32>(
        f32(bits.x) * 2.3283064365386963e-10,
        f32(bits.y) * 2.3283064365386963e-10
    );
}

// =============================================================================
// Hash and Field Functions
// =============================================================================

// 3D hash (matching Python's hash_nd for 3D positions)
fn hashNd3(x: vec3<f32>) -> u32 {
    let ix = vec3<i32>(floor(x));
    var key = prngKey(0u);
    key = foldIn(key, u32(ix.x));
    key = foldIn(key, u32(ix.y));
    key = foldIn(key, u32(ix.z));
    return randomBits(key).x % (1u << 30u);
}

fn boxMuller(u: vec2<f32>) -> vec2<f32> {
    let r = sqrt(-2.0 * log(max(u.x, 1e-10)));
    let theta = 2.0 * PI * u.y;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// 3D position to key (matching Python's position_to_key for 3D)
fn positionToFullKey3D(pos: vec3<f32>, fieldRes: f32) -> u32 {
    let normalized = (pos + 1.0) * 0.5;
    let gridPos = floor(normalized * fieldRes);
    return hashNd3(gridPos);
}

// =============================================================================
// Adaptive Field Functions (matching Python adaptive_field.py)
// =============================================================================

// Convert footprint (Jacobian determinant) to LOD level
// Matches: footprint = 16 / sqrt(footprint); level = log2(footprint)
fn footprintToLevel(footprint: f32) -> f32 {
    let scaledFootprint = 16.0 / sqrt(max(footprint, 0.0001));
    return log2(max(scaledFootprint, 1.0));
}

// Get key at a specific LOD level for HISTOGRAM (3D version)
// Matches Python level_key: pos/4 -> normalize -> ×res -> ×2^level
fn levelHistogramKey3D(pos: vec3<f32>, level: i32, fieldRes: f32) -> u32 {
    var scaledPos = pos / 4.0;                        // 1. Divide by 4 first
    let normalized = (scaledPos + 1.0) * 0.5;         // 2. Normalize to [0, 1]
    let scaleFactor = pow(2.0, f32(level));
    let gridPos = floor(normalized * fieldRes * scaleFactor);  // 3. Scale by res, then by 2^level
    return hashNd3(gridPos);
}

// Get key at a specific LOD level for VALUE lookup (3D version)
// Matches Python level_position_to_key: Field.position_to_key(pos * 2^level / 4)
// Which is: ((pos * 2^level / 4) + 1) / 2 * res
fn levelPositionToKey3D(pos: vec3<f32>, level: i32, fieldRes: f32) -> u32 {
    let scaleFactor = pow(2.0, f32(level));
    let scaledPos = pos * scaleFactor / 4.0;          // 1. Scale by 2^level, then divide by 4
    let normalized = (scaledPos + 1.0) * 0.5;         // 2. Normalize to [0, 1]
    let gridPos = floor(normalized * fieldRes);       // 3. Scale by res only (level already applied)
    return hashNd3(gridPos);
}

// Get value at a specific LOD level with level-specific value key (3D version) - 16 channels
fn levelPositionToValue16_3D(pos: vec3<f32>, level: i32, fieldRes: f32, valueKey: f32) -> Value16 {
    let key = levelPositionToKey3D(pos, level, fieldRes);
    // Fold in level to get different noise at each level
    var prngKeyVal = prngKey(key);
    prngKeyVal = foldIn(prngKeyVal, u32(level));
    prngKeyVal = foldIn(prngKeyVal, bitcast<u32>(valueKey));

    var result: Value16;

    // Generate 16 Gaussian values (8 pairs via Box-Muller)
    for (var i = 0u; i < 8u; i++) {
        let keyPair = foldIn(prngKeyVal, i);
        let u = randomUniform2(keyPair);
        let g = boxMuller(u);

        switch (i) {
            case 0u: { result.v0.x = g.x; result.v0.y = g.y; }
            case 1u: { result.v0.z = g.x; result.v0.w = g.y; }
            case 2u: { result.v1.x = g.x; result.v1.y = g.y; }
            case 3u: { result.v1.z = g.x; result.v1.w = g.y; }
            case 4u: { result.v2.x = g.x; result.v2.y = g.y; }
            case 5u: { result.v2.z = g.x; result.v2.w = g.y; }
            case 6u: { result.v3.x = g.x; result.v3.y = g.y; }
            case 7u: { result.v3.z = g.x; result.v3.w = g.y; }
            default: {}
        }
    }

    return result;
}

// =============================================================================
// 16-Channel Field Functions (for export)
// =============================================================================

// Adaptive Field 3D: single discrete LOD level, no blending - 16 channels
fn adaptiveFieldValue16_3D(pos: vec3<f32>, footprint: f32, fieldRes: f32, valueKey: f32) -> Value16 {
    let level = footprintToLevel(footprint);
    let discreteLevel = clamp(i32(floor(level)), 0, 8);
    return levelPositionToValue16_3D(pos, discreteLevel, fieldRes, valueKey);
}

// Blended Adaptive Field: trilinear interpolation between LOD levels (3D version) - 16 channels
fn blendedAdaptiveFieldValue16_3D(pos: vec3<f32>, footprint: f32, fieldRes: f32, valueKey: f32) -> Value16 {
    let level = footprintToLevel(footprint);
    let discreteLevel = i32(floor(level));

    // Clamp level to reasonable range (0 to 8 levels)
    let clampedLevel0 = clamp(discreteLevel, 0, 8);
    let clampedLevel1 = clamp(discreteLevel + 1, 0, 8);

    // Get noise values at two adjacent LOD levels
    let value0 = levelPositionToValue16_3D(pos, clampedLevel0, fieldRes, valueKey);
    let value1 = levelPositionToValue16_3D(pos, clampedLevel1, fieldRes, valueKey);

    // Interpolation weight (fractional part of level)
    let weight = level - floor(level);

    // Trilinear blend between levels
    return value16Lerp(value0, value1, weight);
}

// White Field 3D: fixed resolution, ignores footprint - 16 channels
fn whiteFieldValue16_3D(pos: vec3<f32>, fieldRes: f32, valueKey: f32) -> Value16 {
    let key = positionToFullKey3D(pos, fieldRes);
    return keyToValue16(key, valueKey);
}

// =============================================================================
// Field Selection (White, Blended Adaptive, or Adaptive) - 3D version
// =============================================================================


// 16-channel version for export
fn getFieldValue16_3D(pos: vec3<f32>, footprint: f32, fieldRes: f32, valueKey: f32) -> Value16 {
    let fieldType = u32(params.fieldType);

    switch (fieldType) {
        case 1u: {
            // Blended Adaptive Field - trilinear LOD blending (no variance compensation)
            return blendedAdaptiveFieldValue16_3D(pos, footprint, fieldRes, valueKey);
        }
        case 2u: {
            // Adaptive Field - single discrete LOD level, no blending
            return adaptiveFieldValue16_3D(pos, footprint, fieldRes, valueKey);
        }
        default: {
            // White Field - fixed resolution, ignores footprint
            return whiteFieldValue16_3D(pos, fieldRes, valueKey);
        }
    }
}

// Get TWO histogram keys and weights for adaptive fields (matching Python's position_to_key)
// Returns: x = key0 bin, y = key1 bin, z = weight0, w = weight1
fn getHistogramKeysAndWeights3D(pos: vec3<f32>, footprint: f32, fieldRes: f32) -> vec4<f32> {
    let fieldType = u32(params.fieldType);

    // Adaptive fields (1 = Blended, 2 = Adaptive) use LOD-based key with two levels
    if (fieldType == 1u || fieldType == 2u) {
        let level = footprintToLevel(footprint);
        let discreteLevel = i32(floor(level));
        let clampedLevel0 = clamp(discreteLevel, 0, 8);
        let clampedLevel1 = clamp(discreteLevel + 1, 0, 8);

        // Weight for blending (matches Python: weight = level - discrete_level)
        let weight = level - f32(discreteLevel);

        // Get keys for both levels
        let key0 = levelHistogramKey3D(pos, clampedLevel0, fieldRes);
        let key1 = levelHistogramKey3D(pos, clampedLevel1, fieldRes);

        let bin0 = f32(key0 % u32(params.histogramBinCount));
        let bin1 = f32(key1 % u32(params.histogramBinCount));

        // Python: returns [1 - weight, weight] as weights
        return vec4<f32>(bin0, bin1, 1.0 - weight, weight);
    }

    // White field uses fixed resolution key (single key, full weight)
    let key = positionToFullKey3D(pos, fieldRes);
    let bin = f32(key % u32(params.histogramBinCount));
    return vec4<f32>(bin, bin, 1.0, 0.0);
}

// Get histogram key based on field type (3D version) - backward compatible

// Histogram key at a specific LOD level - matches Python's level_key() in position_to_key
// Python order: pos/4 -> (pos+1)/2 -> *res -> *2^level -> floor -> hash

// Get histogram key for adaptive fields - returns key from primary LOD level
// This matches Python's AdaptiveField.position_to_key behavior

// Get histogram key based on field type

// Get TWO histogram keys and weights from UV coordinates (matching Python's position_to_key)
// This is used for screen-space jittering where we jitter UV coords for histogram keys
// but use 3D positions for noise values
// Returns: x = key0 bin, y = key1 bin, z = weight0, w = weight1


// Generate 16 Gaussian values from a key (matching Python's field.value_dimension = 16)
fn keyToValue16(fullKey: u32, valueKey: f32) -> Value16 {
    var key = prngKey(fullKey);
    key = foldIn(key, bitcast<u32>(valueKey));

    var result: Value16;

    // Generate 16 Gaussian values (8 pairs via Box-Muller)
    // Channels 0-1
    let key0 = foldIn(key, 0u);
    let u0 = randomUniform2(key0);
    let g0 = boxMuller(u0);
    result.v0.x = g0.x;
    result.v0.y = g0.y;

    // Channels 2-3
    let key1 = foldIn(key, 1u);
    let u1 = randomUniform2(key1);
    let g1 = boxMuller(u1);
    result.v0.z = g1.x;
    result.v0.w = g1.y;

    // Channels 4-5
    let key2 = foldIn(key, 2u);
    let u2 = randomUniform2(key2);
    let g2 = boxMuller(u2);
    result.v1.x = g2.x;
    result.v1.y = g2.y;

    // Channels 6-7
    let key3 = foldIn(key, 3u);
    let u3 = randomUniform2(key3);
    let g3 = boxMuller(u3);
    result.v1.z = g3.x;
    result.v1.w = g3.y;

    // Channels 8-9
    let key4 = foldIn(key, 4u);
    let u4 = randomUniform2(key4);
    let g4 = boxMuller(u4);
    result.v2.x = g4.x;
    result.v2.y = g4.y;

    // Channels 10-11
    let key5 = foldIn(key, 5u);
    let u5 = randomUniform2(key5);
    let g5 = boxMuller(u5);
    result.v2.z = g5.x;
    result.v2.w = g5.y;

    // Channels 12-13
    let key6 = foldIn(key, 6u);
    let u6 = randomUniform2(key6);
    let g6 = boxMuller(u6);
    result.v3.x = g6.x;
    result.v3.y = g6.y;

    // Channels 14-15
    let key7 = foldIn(key, 7u);
    let u7 = randomUniform2(key7);
    let g7 = boxMuller(u7);
    result.v3.z = g7.x;
    result.v3.w = g7.y;

    return result;
}

// =============================================================================
// Hammersley Sampling
// =============================================================================

fn vanDerCorput(idx: f32, base: f32) -> f32 {
    var result = 0.0;
    var f = 1.0 / base;
    var n = idx;

    for (var i = 0; i < 32; i++) {
        if (n <= 0.0) { break; }
        result += (n % base) * f;
        n = floor(n / base);
        f /= base;
    }
    return result;
}

fn hammersley2D(index: f32, numSamples: f32, sampleBase: f32) -> vec2<f32> {
    let idx = sampleBase * numSamples + index;
    return vec2<f32>(index / numSamples, vanDerCorput(idx, 2.0));
}

fn boxFilter(sample: vec2<f32>, filterSize: vec2<f32>) -> vec2<f32> {
    return sample * filterSize;
}

// =============================================================================
// Process a single pixel (called sequentially for each pixel in tile)
// =============================================================================

fn processPixel(
    pixelX: u32,
    pixelY: u32,
    sampleIdx: u32,
    maxSamples: u32
) {
    // Read this pixel's data from rasterizer output
    let pixelCoord = vec2<i32>(i32(pixelX), i32(pixelY));
    let posData = textureLoad(positionTexture, pixelCoord, 0);
    let footprintData = textureLoad(footprintTexture, pixelCoord, 0);

    // Extract 3D warped position, depth, footprint, and objectId from rasterizer output
    // posData: XYZ = 3D warped position, W = depth
    // footprintData: X = footprint (from dpdx/dpdy in fragment shader), Y = objectId,
    //                ZW = world-space normal XY
    var warpedPos3D = posData.xyz;
    let depth = posData.w;
    let footprint = footprintData.x;  // Now computed via dpdx/dpdy in fragment shader
    let objectId = footprintData.y;

    // Pixel size on each axis. Both axes are needed: jitterY scales by
    // resolutionY below, so a single resolutionX-derived width only cancels on a
    // square render and otherwise jitters vertically by aspect/2 pixels.
    let filterWidth = vec2<f32>(1.0 / params.resolutionX, 1.0 / params.resolutionY);
    // Use a constant value key - the same across all frames
    // Uniqueness comes from the position hash, not the value key
    // This matches Python where field_key is constant
    let valueKey = params.seed;  // Seed from UI control

    // Estimator mode: 0 = Biased (single pass), 1 = Unbiased (two passes with independent samples)
    let isUnbiased = u32(params.estimatorMode) == 1u;
    let passCount = select(1u, 2u, isUnbiased);

    var myValue16 = value16Zero();  // Full 16-channel value

    // Two-pass loop for unbiased estimation:
    // Pass 0: Generate samples for histogram (variance correction)
    // Pass 1: Generate independent samples for accumulator (unbiased mode only)
    for (var passIdx = 0u; passIdx < passCount; passIdx++) {
        // Use different sample base for each pass to ensure independence
        // Pass 0: sampleBase = 0, Pass 1: sampleBase = maxSamples (offset Hammersley sequence)
        let currentSampleBase = select(0.0, f32(maxSamples), passIdx == 1u);

        if (sampleIdx < maxSamples) {
            // 2D Hammersley sampling for jittering around the warped position
            let spatialSample = hammersley2D(
                f32(sampleIdx),
                f32(maxSamples),
                currentSampleBase,
            ) * 2.0 - 1.0;

            // Spatial offset from box filter (small jitter for anti-aliasing)
            // Python applies this to UV coords BEFORE mapping to 3D position
            let offset = boxFilter(spatialSample, filterWidth);

            // For rasterization modes (physics balls, custom mesh, etc.), use screen-space texture lookup
            // with bilinear interpolation. This avoids expensive per-sample raytracing by sampling
            // from the pre-computed G-buffer (position texture from rasterizer pass).
            //
            // IMPORTANT: Both histogram keys AND noise values must come from the SAME jittered 3D position
            // to ensure variance correction is accurate (matching Python behavior)

            // Convert jitter offset to pixel units
            // The offset is in normalized screen space [-filterWidth, filterWidth],
            // one pixel wide on each axis, so this recovers +/-0.5 pixel on both.
            // In raytracing, offset is added to screenPos which is in [-1, 1], so 1 pixel = 2/resolution
            // To match: offset * (resolution / 2) gives the equivalent pixel offset
            // But we want offset * resolution to convert from normalized to pixel space
            // The factor of 2 difference comes from raytracing using [-1,1] range vs rasterization using [0,width]
            let jitterX = f32(pixelX) + offset.x * params.resolutionX * 0.5;
            let jitterY = f32(pixelY) + offset.y * params.resolutionY * 0.5;

            // Compute bilinear interpolation coordinates
            let x0 = i32(floor(jitterX));
            let y0 = i32(floor(jitterY));
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            // Clamp to texture bounds
            let maxX = i32(params.resolutionX) - 1;
            let maxY = i32(params.resolutionY) - 1;
            let cx0 = clamp(x0, 0, maxX);
            let cy0 = clamp(y0, 0, maxY);
            let cx1 = clamp(x1, 0, maxX);
            let cy1 = clamp(y1, 0, maxY);

            // Fractional part for interpolation
            let fx = fract(jitterX);
            let fy = fract(jitterY);

            // Sample position texture at 4 neighboring pixels
            let p00 = textureLoad(positionTexture, vec2<i32>(cx0, cy0), 0).xyz;
            let p10 = textureLoad(positionTexture, vec2<i32>(cx1, cy0), 0).xyz;
            let p01 = textureLoad(positionTexture, vec2<i32>(cx0, cy1), 0).xyz;
            let p11 = textureLoad(positionTexture, vec2<i32>(cx1, cy1), 0).xyz;

            // Bilinear interpolation for jittered 3D position
            let samplePos3D = mix(mix(p00, p10, fx), mix(p01, p11, fx), fy);

            // Use the SAME jittered 3D position for BOTH histogram key AND noise value
            // This matches Python where both use map.map(jittered_uv)
            let keysAndWeights = getHistogramKeysAndWeights3D(samplePos3D, footprint, params.fieldResolution);
            let bin0 = u32(keysAndWeights.x);
            let bin1 = u32(keysAndWeights.y);
            let weight0 = keysAndWeights.z;
            let weight1 = keysAndWeights.w;

            // Compute full 16-channel value
            myValue16 = getFieldValue16_3D(samplePos3D, footprint, params.fieldResolution, valueKey);

            // Store for histogram algorithm
            sampleValues[sampleIdx] = myValue16;

            // 2D histogram - each LOD level gets its own histogram array
            // This prevents bin collisions between levels for more accurate variance
            // In unbiased mode (pass 1), skip histogram updates - use independent samples for accumulator only
            if (passIdx == 0u) {
                let scaledWeight0 = u32(weight0 * WEIGHT_SCALE);
                let scaledWeight1 = u32(weight1 * WEIGHT_SCALE);
                atomicAdd(&weightedHistogram0[bin0], scaledWeight0);  // Coarser level
                atomicAdd(&weightedHistogram1[bin1], scaledWeight1);  // Finer level
            }
        }
    }  // End of two-pass loop

    // Ensure all writes to shared memory are visible
    workgroupBarrier();
    storageBarrier();

    // Thread 0 computes the final result
    if (sampleIdx == 0u) {
        let outputMode = u32(params.outputMode);

        // All modes output RAW values - tonemapping is done in a separate pass
        var rawOutput = vec3<f32>(0.0);

        switch (outputMode) {
            case 0u: {
                // Mode 0: Final Noise - RAW N(0,1) values (16 channels)
                // Accumulate all 16 channels (now stored in sampleValues as Value16)
                var accumulator16 = value16Zero();
                for (var s = 0u; s < maxSamples; s++) {
                    accumulator16 = value16Add(accumulator16, sampleValues[s]);
                }
                let invMaxSamples = 1.0 / f32(maxSamples);
                accumulator16 = value16Scale(accumulator16, invMaxSamples);

                // Count-unique variance correction using 2D WEIGHTED histogram (matching Algorithm 2)
                // Each LOD level has its own histogram to prevent bin collisions between levels
                // Total weight = maxSamples * WEIGHT_SCALE (since each sample contributes weight sum of 1.0)
                let totalWeight = f32(maxSamples) * WEIGHT_SCALE;

                // Compute sum of squared normalized weights from BOTH histogram levels
                var weightSquaredSum = 0.0;
                let binCount = u32(params.histogramBinCount);

                // Level 0 (coarser level) histogram
                for (var k = 0u; k < binCount; k++) {
                    let scaledCount = f32(atomicLoad(&weightedHistogram0[k]));
                    if (scaledCount > 0.0) {
                        let normalizedWeight = scaledCount / totalWeight;
                        weightSquaredSum += normalizedWeight * normalizedWeight;
                    }
                }

                // Level 1 (finer level) histogram
                for (var k = 0u; k < binCount; k++) {
                    let scaledCount = f32(atomicLoad(&weightedHistogram1[k]));
                    if (scaledCount > 0.0) {
                        let normalizedWeight = scaledCount / totalWeight;
                        weightSquaredSum += normalizedWeight * normalizedWeight;
                    }
                }

                let varianceCorrection = 1.0 / sqrt(weightSquaredSum);

                // Apply variance correction to all 16 channels
                let finalNoise16 = value16Scale(accumulator16, varianceCorrection);

                // Write all 16 channels to 4 output textures
                var outY = i32(pixelY);
                textureStore(outputTexture, vec2<i32>(i32(pixelX), outY), finalNoise16.v0);   // Channels 0-3
                textureStore(outputTexture1, vec2<i32>(i32(pixelX), outY), finalNoise16.v1);  // Channels 4-7
                textureStore(outputTexture2, vec2<i32>(i32(pixelX), outY), finalNoise16.v2);  // Channels 8-11
                textureStore(outputTexture3, vec2<i32>(i32(pixelX), outY), finalNoise16.v3);  // Channels 12-15
            }
            case 3u: {
                // Mode 3: Footprint - RAW Jacobian determinant
                rawOutput = vec3<f32>(footprint, footprint, footprint);
            }
            case 9u: {
                // Mode 9: Footprint for LOD level visualization (same as mode 3)
                // Tonemap shader will convert to LOD level
                rawOutput = vec3<f32>(footprint, footprint, footprint);
            }
            case 4u: {
                // Mode 4: Depth - RAW depth value (ray travel distance)
                rawOutput = vec3<f32>(depth, depth, depth);
            }
            case 5u: {
                // Mode 5: Object ID - Output raw ID directly
                // IDs: -1 = background, -2 = floor, 0+ = balls/objects
                rawOutput = vec3<f32>(objectId, objectId, objectId);
            }
            case 6u: {
                // Mode 6: Object-Space Position - RAW XYZ coordinates
                // Position relative to the primitive's local coordinate system
                // (matches Python: ray.position + t * ray.direction - local_position)
                rawOutput = warpedPos3D;
            }
            case 7u: {
                // Mode 7: World-Space Normals - RAW normal vector [-1, 1]
                // Normals in world space (matches Python: primitive.get_normal())
                let objectId = footprintData.y;

                if (objectId > -1.5 && objectId < -0.5) {
                    // Background - default facing-camera normal in world space
                    rawOutput = vec3<f32>(0.0, 0.0, -1.0);
                } else {
                    // Reconstruct world-space normal from XY components stored in footprintData
                    let nx = footprintData.z;
                    let ny = footprintData.w;
                    let nzSq = 1.0 - nx * nx - ny * ny;
                    let nz = sqrt(max(nzSq, 0.0));
                    rawOutput = vec3<f32>(nx, ny, nz);
                }
            }
            case 8u: {
                // Mode 8: Camera-Space (World-Space) Position - RAW XYZ coordinates
                // This is the actual hit point in world space: ray.origin + t * ray.direction
                // Use ray + depth to compute world-space position (works for all warp modes)
                // depth is already extracted from posData.w at the start of processPixel

                // Convert pixel coordinates back to normalized [-1, 1] UV space
                let uvX = (f32(pixelX) + 0.5) / params.resolutionX * 2.0 - 1.0;
                let uvY = (f32(pixelY) + 0.5) / params.resolutionY * 2.0 - 1.0;
                let pixelUV = vec2<f32>(uvX, uvY);

                // Generate ray for this pixel and compute world-space hit point
                let ray = generateRayForPixel(pixelUV);
                var worldSpacePos = ray.origin + ray.direction * depth;
                // Negate Y to match Python's coordinate convention
                worldSpacePos.y = -worldSpacePos.y;
                rawOutput = worldSpacePos;
            }
            default: {
                rawOutput = vec3<f32>(0.0, 0.0, 0.0);
            }
        }

        // For mode 0 (noise), textures are already written inside the case block with all 16 channels
        // For other modes, write the 3-channel rawOutput to the primary texture
        if (outputMode != 0u) {
            textureStore(outputTexture, vec2<i32>(i32(pixelX), i32(pixelY)), vec4<f32>(rawOutput, 1.0));
        }
    }
}

// =============================================================================
// Main Compute Shader - One workgroup per pixel
// Workgroup size is configurable via WORKGROUP_SIZE override constant
// =============================================================================

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>
) {
    let resolutionX = u32(params.resolutionX);
    let resolutionY = u32(params.resolutionY);
    let pixelX = workgroupId.x;
    let pixelY = workgroupId.y;
    let sampleIdx = localId.x;
    let sampleCount = u32(params.sampleCount);
    // Cap samples at 256 to fit Value16 storage in 32KB workgroup memory limit
    let maxSamples = min(min(sampleCount, WORKGROUP_SIZE), 256u);

    if (pixelX >= resolutionX || pixelY >= resolutionY) {
        return;
    }

    // Clear the histograms and sample storage. All three arrays hold a fixed 256
    // entries, which can exceed WORKGROUP_SIZE, so stride the clear across whatever
    // threads exist rather than assuming one thread per entry. Clearing only index
    // sampleIdx left entries at and above WORKGROUP_SIZE untouched, and the histogram
    // reduction below reads up to histogramBinCount of them: with fewer samples than
    // bins that read reached workgroup memory WGSL does not guarantee is zeroed.
    for (var i = sampleIdx; i < 256u; i += WORKGROUP_SIZE) {
        atomicStore(&weightedHistogram0[i], 0u);
        atomicStore(&weightedHistogram1[i], 0u);
        sampleValues[i] = value16Zero();
    }
    workgroupBarrier();

    // Process this pixel
    processPixel(pixelX, pixelY, sampleIdx, maxSamples);
}
