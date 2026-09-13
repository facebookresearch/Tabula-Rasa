// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// =============================================================================
// Tabula Rasa - WebGPU render shader
// Fullscreen quad to display the output texture
// =============================================================================

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texCoord: vec2<f32>,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    // Fullscreen triangle (more efficient than quad)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
    output.texCoord = positions[vertexIndex] * 0.5 + 0.5;
    return output;
}

@group(0) @binding(0) var outputTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

@fragment
fn fragmentMain(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(outputTexture, textureSampler, texCoord);
}
