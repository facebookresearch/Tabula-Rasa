/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// =============================================================================
// Tabula Rasa - OBJ file parser
// Parses Wavefront OBJ mesh files
// =============================================================================

/**
 * Parse an OBJ file text and return mesh data
 * @param {string} text - The OBJ file content
 * @returns {Object} Mesh data with vertices, normals, indices, triangleCount, and boundingBox
 */
function parseOBJ(text) {
    const vertices = [];
    const normals = [];
    const faces = [];

    const lines = text.split('\n');

    for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length === 0) continue;

        const type = parts[0];

        if (type === 'v') {
            // Vertex position: v x y z
            vertices.push([
                parseFloat(parts[1]) || 0,
                parseFloat(parts[2]) || 0,
                parseFloat(parts[3]) || 0
            ]);
        } else if (type === 'vn') {
            // Vertex normal: vn x y z
            normals.push([
                parseFloat(parts[1]) || 0,
                parseFloat(parts[2]) || 0,
                parseFloat(parts[3]) || 0
            ]);
        } else if (type === 'f') {
            // Face: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
            // Can also be: f v1 v2 v3 or f v1//vn1 v2//vn2 v3//vn3
            const faceVerts = [];
            for (let i = 1; i < parts.length; i++) {
                const indices = parts[i].split('/');
                faceVerts.push({
                    v: parseInt(indices[0]) - 1,  // OBJ indices are 1-based
                    vn: indices[2] ? parseInt(indices[2]) - 1 : -1
                });
            }

            // Triangulate faces (fan triangulation for convex polygons)
            for (let i = 1; i < faceVerts.length - 1; i++) {
                faces.push([faceVerts[0], faceVerts[i], faceVerts[i + 1]]);
            }
        }
    }

    // Build the final mesh data
    // We need to flatten vertex/normal data per triangle for GPU
    const triangleVertices = [];
    const triangleNormals = [];
    const triangleIndices = [];

    // Compute bounding box for normalization
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const v of vertices) {
        minX = Math.min(minX, v[0]); maxX = Math.max(maxX, v[0]);
        minY = Math.min(minY, v[1]); maxY = Math.max(maxY, v[1]);
        minZ = Math.min(minZ, v[2]); maxZ = Math.max(maxZ, v[2]);
    }

    // Center and scale to fit in unit cube
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);
    const scale = maxSize > 0 ? 1.0 / maxSize : 1.0;

    for (let faceIdx = 0; faceIdx < faces.length; faceIdx++) {
        const face = faces[faceIdx];

        // Get triangle vertices
        const v0 = vertices[face[0].v];
        const v1 = vertices[face[1].v];
        const v2 = vertices[face[2].v];

        // Normalize to unit cube centered at origin, flip Y to match coordinate convention
        const normalizedV0 = [
            (v0[0] - centerX) * scale,
            -(v0[1] - centerY) * scale,
            (v0[2] - centerZ) * scale
        ];
        const normalizedV1 = [
            (v1[0] - centerX) * scale,
            -(v1[1] - centerY) * scale,
            (v1[2] - centerZ) * scale
        ];
        const normalizedV2 = [
            (v2[0] - centerX) * scale,
            -(v2[1] - centerY) * scale,
            (v2[2] - centerZ) * scale
        ];

        triangleVertices.push(...normalizedV0, ...normalizedV1, ...normalizedV2);

        // Get or compute normals (flip Y to match vertex flip)
        let n0, n1, n2;
        if (normals.length > 0 && face[0].vn >= 0) {
            const rawN0 = normals[face[0].vn];
            const rawN1 = normals[face[1].vn];
            const rawN2 = normals[face[2].vn];
            n0 = [rawN0[0], -rawN0[1], rawN0[2]];
            n1 = [rawN1[0], -rawN1[1], rawN1[2]];
            n2 = [rawN2[0], -rawN2[1], rawN2[2]];
        } else {
            // Compute face normal
            const edge1 = [normalizedV1[0] - normalizedV0[0], normalizedV1[1] - normalizedV0[1], normalizedV1[2] - normalizedV0[2]];
            const edge2 = [normalizedV2[0] - normalizedV0[0], normalizedV2[1] - normalizedV0[1], normalizedV2[2] - normalizedV0[2]];
            const crossProduct = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0]
            ];
            const len = Math.sqrt(crossProduct[0] ** 2 + crossProduct[1] ** 2 + crossProduct[2] ** 2);
            const faceNormal = len > 0 ? [crossProduct[0] / len, crossProduct[1] / len, crossProduct[2] / len] : [0, 1, 0];
            n0 = n1 = n2 = faceNormal;
        }

        triangleNormals.push(...n0, ...n1, ...n2);
        triangleIndices.push(faceIdx * 3, faceIdx * 3 + 1, faceIdx * 3 + 2);
    }

    return {
        vertices: new Float32Array(triangleVertices),
        normals: new Float32Array(triangleNormals),
        indices: new Uint32Array(triangleIndices),
        triangleCount: faces.length,
        boundingBox: { minX, minY, minZ, maxX, maxY, maxZ }
    };
}
