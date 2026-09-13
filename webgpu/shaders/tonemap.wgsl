// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// =============================================================================
// Tonemap Shader - Converts raw values to displayable [0,1] range
// Each output mode has its own tonemapping for visualization
// Raw values are preserved in the noise texture for export
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// Viridis-like colormap (matches Python debug_footprint.py)
fn viridisColormap(t: f32) -> vec3<f32> {
    let r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
    let g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
    let b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let dims = textureDimensions(inputTexture);
    if (globalId.x >= dims.x || globalId.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(globalId.xy);
    let rawValue = textureLoad(inputTexture, coord, 0);

    var finalColor: vec3<f32>;
    let outputMode = u32(params.outputMode);

    switch (outputMode) {
        case 0u: {
            // Noise: N(0,1) -> [0,1] for display
            // Maps approximately ±3σ to [0,1] range
            finalColor = rawValue.rgb * 0.166666 + 0.5;
        }
        case 3u: {
            // Footprint magnitude: normalize and apply viridis colormap
            // Match Python's per-frame min/max normalization
            let footprint = rawValue.r;
            // Use same range as Python tends to produce (approximately)
            // Python normalizes min to max per frame - we approximate with fixed range
            let minFootprint = 0.0;
            let maxFootprint = 50.0;  // Typical max footprint value at horizon
            let normalizedFootprint = clamp((footprint - minFootprint) / (maxFootprint - minFootprint), 0.0, 1.0);
            finalColor = viridisColormap(normalizedFootprint);
        }
        case 9u: {
            // LOD Level visualization - matches Python debug_footprint.py
            // Python: level = log2(16 / sqrt(footprint))
            // Python LOD level stats: min=0, max=5.3
            let footprint = rawValue.r;

            // Compute LOD level (inverse relationship to footprint)
            let scaledFootprint = 16.0 / sqrt(max(footprint, 0.0001));
            let level = log2(max(scaledFootprint, 1.0));

            // Normalize to Python's range: 0 to 5.3
            let minLevel = 0.0;
            let maxLevel = 5.5;
            let normalizedLevel = clamp((level - minLevel) / (maxLevel - minLevel), 0.0, 1.0);

            finalColor = viridisColormap(normalizedLevel);
        }
        case 4u: {
            // Depth: normalize to [0,1] range
            // Raw depth is in world units, scale for visualization
            // Assuming depth range roughly [1, 20]
            let normalizedDepth = (rawValue.r - 1.0) / 19.0;
            finalColor = vec3<f32>(clamp(normalizedDepth, 0.0, 1.0));
        }
        case 5u: {
            // Object ID: visualization with color gradient
            // Raw IDs: -1 = background, -2 = floor, 0+ = balls/objects
            let objId = rawValue.r;

            // Background (-1) should be black
            if (objId > -1.5 && objId < -0.5) {
                finalColor = vec3<f32>(0.0, 0.0, 0.0);
            } else if (objId < -1.5) {
                // Floor (-2) - distinct color (gray)
                finalColor = vec3<f32>(0.4, 0.4, 0.4);
            } else {
                // Objects (0+) - color gradient
                let normalizedId = (objId + 1.0) / 10.0;
                finalColor = vec3<f32>(
                    fract(normalizedId * 3.0),
                    fract(normalizedId * 5.0 + 0.33),
                    fract(normalizedId * 7.0 + 0.66)
                );
            }
        }
        case 6u: {
            // Object-space position: normalize for visualization
            // Position relative to primitive's local coordinate system
            finalColor = rawValue.rgb * 0.5 + 0.5;
        }
        case 7u: {
            // World-space normals: [-1,1] -> [0,1] for display
            finalColor = rawValue.rgb * 0.5 + 0.5;
        }
        case 8u: {
            // Camera-space (world-space) position: raw values clamped to [0,1]
            // No offset transformation - negative values become black
            // This matches Python's direct position-to-color mapping
            finalColor = clamp(rawValue.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        default: {
            finalColor = rawValue.rgb;
        }
    }

    textureStore(outputTexture, coord, vec4<f32>(finalColor, 1.0));
}
