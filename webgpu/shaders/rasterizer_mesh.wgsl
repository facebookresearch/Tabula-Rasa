// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// =============================================================================
// Mesh Rasterizer Shader
// True GPU rasterization of the loaded mesh plus the background cube.
// Pass 1: Outputs position map and footprint/derivative map (G-buffer),
// using dpdx/dpdy for true GPU derivatives. Skeletal animation is applied
// on the GPU via linear-blend skinning.
// =============================================================================

#include "common.wgsl"

// =============================================================================
// Bindings
// =============================================================================

// Joint matrices for skeletal animation (max 128 joints)
// Each joint has a 4x4 matrix = 16 floats
// We pack them as array of mat4x4 - must be 256 byte aligned for uniform
const MAX_JOINTS: u32 = 128u;

struct JointMatrices {
    matrices: array<mat4x4<f32>, 128>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(6) var<uniform> jointMatrices: JointMatrices;


// =============================================================================
// Skinned Mesh Vertex Input (with joint indices and weights)
// =============================================================================

struct SkinnedVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) joints: vec4<u32>,     // Joint indices (packed as 4 u8 -> u32)
    @location(4) weights: vec4<f32>,    // Joint weights
}

// =============================================================================
// Vertex Shader for Skinned Custom Mesh with Skeletal Animation
// Input vertices are in ORIGINAL GLTF coordinates (not normalized)
// The shader applies skinning, then normalizes the result
// =============================================================================

// Normalization params passed via push constants or additional uniforms
// For now, we pass them via params buffer - meshScale contains the normalization scale

@vertex
fn vertexMainSkinned(input: SkinnedVertexInput) -> MeshVertexOutput {
    var output: MeshVertexOutput;

    // Apply skeletal skinning in original GLTF space
    var skinnedPosition = vec3<f32>(0.0);
    var skinnedNormal = vec3<f32>(0.0);
    var totalWeight = 0.0;

    // Sum weighted contributions from up to 4 joints
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let jointIdx = input.joints[i];
        let weight = input.weights[i];

        if (weight > 0.0 && jointIdx < MAX_JOINTS) {
            let jointMatrix = jointMatrices.matrices[jointIdx];

            // Transform position
            let transformedPos = jointMatrix * vec4<f32>(input.position, 1.0);
            skinnedPosition = skinnedPosition + transformedPos.xyz * weight;

            // Transform normal (use upper 3x3 of matrix)
            let transformedNormal = mat3x3<f32>(
                jointMatrix[0].xyz,
                jointMatrix[1].xyz,
                jointMatrix[2].xyz
            ) * input.normal;
            skinnedNormal = skinnedNormal + transformedNormal * weight;

            totalWeight = totalWeight + weight;
        }
    }

    // Fallback if no weights applied
    if (totalWeight < 0.001) {
        skinnedPosition = input.position;
        skinnedNormal = input.normal;
    }

    // Normalize the skinned normal
    skinnedNormal = normalize(skinnedNormal);

    // The skinned position is in original GLTF coordinates
    // Apply normalization: center, scale, and Y flip (same as static mesh loader does)
    let normCenter = vec3<f32>(params.meshNormCenterX, params.meshNormCenterY, params.meshNormCenterZ);
    let normScale = params.meshNormScale;

    // Normalize: (pos - center) * scale, with Y flip
    let centeredPos = skinnedPosition - normCenter;
    let normalizedPos = vec3<f32>(
        centeredPos.x * normScale,
        -centeredPos.y * normScale,  // Y flip
        centeredPos.z * normScale
    );

    // Flip normal Y to match
    let normalizedNormal = vec3<f32>(skinnedNormal.x, -skinnedNormal.y, skinnedNormal.z);

    // Camera controls do not alter mesh geometry.
    let meshScale = params.meshScale;
    output.worldPosition = normalizedPos * meshScale;
    output.worldNormal = normalize(normalizedNormal);

    // Object-space position for stable noise sampling
    // Use the ORIGINAL (canonical) input position, not the skinned position
    // This ensures noise stays fixed to the mesh surface during animation
    // Do NOT apply rotation - that's a world-space transform
    let canonicalCentered = input.position - normCenter;
    let canonicalNormalized = vec3<f32>(
        canonicalCentered.x * normScale,
        -canonicalCentered.y * normScale,  // Y flip
        canonicalCentered.z * normScale
    );
    // Only apply scale (not rotation) - scale affects noise LOD but rotation shouldn't
    output.objectSpacePosition = canonicalNormalized * meshScale;

    output.objectId = 1.0;  // Custom mesh has object ID 1

    // Project to clip space using camera matrices
    let viewPos = output.worldPosition - camera.position;
    let viewSpacePos = vec3<f32>(
        -dot(viewPos, camera.right),  // Negate to match raytracer
        -dot(viewPos, camera.up),     // Negate to match raytracer
        dot(viewPos, camera.direction)
    );

    // Perspective projection
    let aspect = params.resolutionX / params.resolutionY;
    let tanHalfFov = tan(camera.fov * 0.5);
    let near = 0.1;
    let far = 100.0;

    let x = viewSpacePos.x / (tanHalfFov * aspect);
    let y = viewSpacePos.y / tanHalfFov;
    let z = far * (viewSpacePos.z - near) / (far - near);

    output.clipPosition = vec4<f32>(x, y, z, viewSpacePos.z);

    return output;
}

// =============================================================================
// Mesh Vertex Input/Output Structures
// =============================================================================

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct MeshVertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) worldPosition: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) objectSpacePosition: vec3<f32>,
    @location(4) @interpolate(flat) objectId: f32,
}

// =============================================================================
// Vertex Shader for Background Cube
// Cube positioned in world space - same projection as floor/balls
// =============================================================================

@vertex
fn vertexMainCube(input: MeshVertexInput) -> MeshVertexOutput {
    var output: MeshVertexOutput;

    // The background is an axis-aligned cube, so a face can coincide exactly with
    // a 3D field-cell boundary. Small interpolation differences can then select
    // different hashed cells on different GPUs. When enabled, uniformly resize the
    // actual cube so its faces sit halfway between adjacent field planes.
    let bgScale = 4.0;  // Visual scale factor
    var worldPos = input.position;
    if (params.centerBackgroundCube > 0.5) {
        let fieldRes = max(params.fieldResolution, 1.0);
        let faceHalfSize = dot(abs(input.position), abs(input.normal));
        let faceGridCoord = (faceHalfSize + 1.0) * 0.5 * fieldRes;
        let centeredGridCoord = floor(faceGridCoord) + 0.5;
        let centeredHalfSize = centeredGridCoord * 2.0 / fieldRes - 1.0;
        worldPos *= centeredHalfSize / faceHalfSize;
    }

    output.worldPosition = worldPos;
    output.worldNormal = input.normal;
    output.objectSpacePosition = worldPos;
    output.objectId = 0.0;  // Background cube has object ID 0

    // Project to clip space
    // Scale camera position DOWN to make the cube appear larger/further away
    // This is equivalent to scaling the cube up but keeps object-space coords small
    let scaledCameraPos = camera.position / bgScale;
    let viewPos = output.worldPosition - scaledCameraPos;
    let viewSpacePos = vec3<f32>(
        -dot(viewPos, camera.right),  // Negate to match raytracer
        -dot(viewPos, camera.up),     // Negate to match raytracer
        dot(viewPos, camera.direction)
    );

    let aspect = params.resolutionX / params.resolutionY;
    let tanHalfFov = tan(camera.fov * 0.5);
    let near = 0.1 / bgScale;  // Scale near/far to match
    let far = 100.0 / bgScale;

    let x = viewSpacePos.x / (tanHalfFov * aspect);
    let y = viewSpacePos.y / tanHalfFov;
    let z = far * (viewSpacePos.z - near) / (far - near);

    output.clipPosition = vec4<f32>(x, y, z, viewSpacePos.z);

    return output;
}

// =============================================================================
// Vertex Shader - Transform mesh vertices to clip space
// =============================================================================

@vertex
fn vertexMain(input: MeshVertexInput) -> MeshVertexOutput {
    var output: MeshVertexOutput;

    // Camera controls leave mesh geometry unchanged.
    let meshScale = params.meshScale;
    output.worldPosition = input.position * meshScale;
    output.worldNormal = normalize(input.normal);

    // Object-space position is the unrotated position, so the noise field stays
    // stable as the camera orbits. Y is flipped to match the Python convention.
    output.objectSpacePosition = vec3<f32>(input.position.x * meshScale, -input.position.y * meshScale, input.position.z * meshScale);

    output.objectId = 1.0;

    // Project to clip space using camera matrices
    // NOTE: The raytracer uses "2.0 * direction - pixelPos.x * u0 - pixelPos.y * u1"
    // The SUBTRACTION means positive pixel coords go in NEGATIVE u0/u1 direction
    // So we need to negate right and up to match the raytracer's convention
    let viewPos = output.worldPosition - camera.position;
    let viewSpacePos = vec3<f32>(
        -dot(viewPos, camera.right),  // Negate to match raytracer
        -dot(viewPos, camera.up),     // Negate to match raytracer
        dot(viewPos, camera.direction)
    );

    // Perspective projection
    let aspect = params.resolutionX / params.resolutionY;
    let tanHalfFov = tan(camera.fov * 0.5);
    let near = 0.1;
    let far = 100.0;

    // Proper perspective projection with w = z
    // After perspective divide: NDC.z = clip.z / clip.w should be in [0, 1]
    let x = viewSpacePos.x / (tanHalfFov * aspect);
    let y = viewSpacePos.y / tanHalfFov;
    let z = far * (viewSpacePos.z - near) / (far - near);

    output.clipPosition = vec4<f32>(x, y, z, viewSpacePos.z);

    return output;
}

// =============================================================================
// Fragment Shader - Output G-buffer data
// =============================================================================

@fragment
fn fragmentMain(input: MeshVertexOutput) -> FragmentOutput {
    // Compute depth from world position
    let viewVec = input.worldPosition - camera.position;
    let depth = length(viewVec);

    // Compute footprint using GPU derivatives
    let dPdx3D = dpdx(input.objectSpacePosition);
    let dPdy3D = dpdy(input.objectSpacePosition);

    // Footprint = magnitude of cross product (area of parallelogram in 3D)
    let crossProduct = cross(dPdx3D, dPdy3D);
    let rawFootprint = length(crossProduct);

    // Convert from per-pixel² to per-UV²
    let pixelsPerUvX = params.resolutionX / 2.0;
    let pixelsPerUvY = params.resolutionY / 2.0;
    let footprint = rawFootprint * pixelsPerUvX * pixelsPerUvY;

    var output: FragmentOutput;
    output.position = vec4<f32>(input.objectSpacePosition, depth);

    // The G-buffer has two spare channels, and the noise pass needs the world-space
    // normal in them (see noise.wgsl outputMode 7 and the exported `normal` tensor).
    // Z is recovered as sqrt(1 - x^2 - y^2), so only XY are stored.
    let n = normalize(input.worldNormal);

    // Store: footprint, objectId, normal.x, normal.y
    output.footprint = vec4<f32>(footprint, input.objectId, n.x, n.y);

    return output;
}
