/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// =============================================================================
// Tabula Rasa - glTF/GLB file parser
// Parses GLTF 2.0 JSON and GLB binary mesh files with animation support
// =============================================================================

/**
 * Parse a GLTF or GLB file and return mesh data with animations
 * @param {ArrayBuffer} arrayBuffer - The file content as ArrayBuffer
 * @param {string} filename - The filename (to determine format)
 * @returns {Object} Mesh data with vertices, normals, indices, triangleCount, boundingBox, and animations
 */
async function parseGLTF(arrayBuffer, filename) {
    const isGLB = filename.toLowerCase().endsWith('.glb');

    let gltf;
    let binaryChunk = null;

    if (isGLB) {
        // Parse GLB binary format
        const result = parseGLBBinary(arrayBuffer);
        gltf = result.json;
        binaryChunk = result.binaryChunk;
    } else {
        // Parse GLTF JSON format
        const text = new TextDecoder().decode(arrayBuffer);
        gltf = JSON.parse(text);
    }

    // Extract animations first so mesh normalization can include their world-space extent.
    const animations = extractGLTFAnimations(gltf, binaryChunk);

    // Extract mesh data from GLTF structure
    const meshData = extractGLTFMesh(gltf, binaryChunk, animations);
    meshData.animations = animations;
    meshData.nodes = gltf.nodes || [];
    meshData.skins = gltf.skins || [];

    // Store the raw GLTF data for animation playback
    meshData.gltf = gltf;
    meshData.binaryChunk = binaryChunk;

    return meshData;
}

/**
 * Parse GLB binary container
 * @param {ArrayBuffer} arrayBuffer - The GLB file content
 * @returns {Object} Parsed JSON and binary chunk
 */
function parseGLBBinary(arrayBuffer) {
    const view = new DataView(arrayBuffer);

    // GLB Header (12 bytes)
    const magic = view.getUint32(0, true);
    if (magic !== 0x46546C67) { // 'glTF' in little-endian
        throw new Error('Invalid GLB file: wrong magic number');
    }

    const version = view.getUint32(4, true);
    if (version !== 2) {
        throw new Error(`Unsupported GLB version: ${version}`);
    }

    // const length = view.getUint32(8, true);

    // Parse chunks
    let offset = 12;
    let json = null;
    let binaryChunk = null;

    while (offset < arrayBuffer.byteLength) {
        const chunkLength = view.getUint32(offset, true);
        const chunkType = view.getUint32(offset + 4, true);

        if (chunkType === 0x4E4F534A) { // 'JSON' in little-endian
            const jsonBytes = new Uint8Array(arrayBuffer, offset + 8, chunkLength);
            const jsonText = new TextDecoder().decode(jsonBytes);
            json = JSON.parse(jsonText);
        } else if (chunkType === 0x004E4942) { // 'BIN' in little-endian
            binaryChunk = new Uint8Array(arrayBuffer, offset + 8, chunkLength);
        }

        offset += 8 + chunkLength;
    }

    if (!json) {
        throw new Error('Invalid GLB file: missing JSON chunk');
    }

    return { json, binaryChunk };
}

/**
 * Extract mesh data from GLTF JSON structure
 * @param {Object} gltf - The GLTF JSON object
 * @param {Uint8Array|null} binaryChunk - The binary chunk (for GLB files)
 * @param {Array} animations - Parsed animations used to calculate full motion bounds
 * @returns {Object} Mesh data with vertices, normals, indices, triangleCount, boundingBox, and skinning data
 */
function extractGLTFMesh(gltf, binaryChunk, animations = []) {
    if (!gltf.meshes || gltf.meshes.length === 0) {
        throw new Error('GLTF file contains no meshes');
    }

    const identityMatrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const maxJoints = 128;
    const nodes = gltf.nodes || [];
    const restWorldMatrices = nodes.length > 0 ? computeNodeWorldMatrices(nodes) : [];

    // Resolve mesh instances from the active scene. If scene metadata is absent,
    // retain permissive fallback behavior for incomplete GLTF files.
    const meshInstances = [];
    if (nodes.length > 0) {
        if (gltf.scenes && gltf.scenes.length > 0) {
            const sceneIdx = gltf.scene !== undefined ? gltf.scene : 0;
            const scene = gltf.scenes[sceneIdx] || gltf.scenes[0];
            const visitNode = (nodeIdx, ancestors = new Set()) => {
                if (ancestors.has(nodeIdx)) {
                    console.warn(`Cycle detected at GLTF node ${nodeIdx}, skipping recursive instance`);
                    return;
                }
                const node = nodes[nodeIdx];
                if (!node) return;

                if (node.mesh !== undefined && gltf.meshes[node.mesh]) {
                    meshInstances.push({
                        meshIndex: node.mesh,
                        nodeIndex: nodeIdx,
                        worldMatrix: restWorldMatrices[nodeIdx] || identityMatrix
                    });
                }

                if (node.children) {
                    const childAncestors = new Set(ancestors);
                    childAncestors.add(nodeIdx);
                    for (const childIdx of node.children) {
                        visitNode(childIdx, childAncestors);
                    }
                }
            };

            for (const rootNodeIdx of scene.nodes || []) {
                visitNode(rootNodeIdx);
            }
        } else {
            for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
                const node = nodes[nodeIdx];
                if (node.mesh !== undefined && gltf.meshes[node.mesh]) {
                    meshInstances.push({
                        meshIndex: node.mesh,
                        nodeIndex: nodeIdx,
                        worldMatrix: restWorldMatrices[nodeIdx] || identityMatrix
                    });
                }
            }
        }
    } else {
        for (let meshIndex = 0; meshIndex < gltf.meshes.length; meshIndex++) {
            meshInstances.push({ meshIndex, nodeIndex: null, worldMatrix: identityMatrix });
        }
    }

    // Preserve the real skin's joint ordering, then add identity-bind rigid joints
    // for active-scene mesh nodes that are not already part of that skin.
    const baseSkin = gltf.skins && gltf.skins.length > 0
        ? extractGLTFSkin(gltf, binaryChunk)
        : null;
    const resolvedJoints = baseSkin ? [...baseSkin.joints] : [];
    const resolvedInverseBindMatrices = baseSkin
        ? baseSkin.inverseBindMatrices.map(matrix => [...matrix])
        : [];
    const jointIndexByNode = new Map();
    for (let i = 0; i < resolvedJoints.length; i++) {
        jointIndexByNode.set(resolvedJoints[i], i);
    }
    for (const instance of meshInstances) {
        if (instance.nodeIndex !== null && !jointIndexByNode.has(instance.nodeIndex)) {
            jointIndexByNode.set(instance.nodeIndex, resolvedJoints.length);
            resolvedJoints.push(instance.nodeIndex);
            resolvedInverseBindMatrices.push([...identityMatrix]);
        }
    }

    const resolvedSkin = resolvedJoints.length > 0 ? {
        joints: resolvedJoints,
        skeleton: baseSkin ? baseSkin.skeleton : undefined,
        inverseBindMatrices: resolvedInverseBindMatrices,
        jointCount: resolvedJoints.length
    } : null;
    if (resolvedSkin) {
        // Keep renderer plumbing unchanged: extractGLTFSkin() resolves this table
        // when main.js follows its existing gltf.skins-based path.
        gltf.skins = gltf.skins || [];
        gltf._resolvedSkin = resolvedSkin;
    }
    const jointTableFits = !resolvedSkin || resolvedSkin.jointCount <= maxJoints;
    if (!jointTableFits) {
        console.warn(
            `GLTF joint table has ${resolvedSkin.jointCount} entries, exceeding the ${maxJoints}-joint shader limit. ` +
            'Falling back to baked static rendering.'
        );
    }
    const canAnimateWithJoints = animations.length > 0 && Boolean(resolvedSkin) && jointTableFits;

    const allLocalVertices = [];
    const allWorldVertices = [];
    const allLocalNormals = [];
    const allWorldNormals = [];
    const allJoints = [];
    const allWeights = [];
    const allIndices = [];
    const boundsSources = [];
    let currentVertexOffset = 0;

    for (const instance of meshInstances) {
        const mesh = gltf.meshes[instance.meshIndex];
        for (const primitive of mesh.primitives || []) {
            const positionAccessorIdx = primitive.attributes.POSITION;
            if (positionAccessorIdx === undefined) {
                console.warn('Primitive missing POSITION attribute, skipping');
                continue;
            }

            const positions = getAccessorData(gltf, positionAccessorIdx, binaryChunk);
            const indices = primitive.indices !== undefined
                ? getAccessorData(gltf, primitive.indices, binaryChunk)
                : createSequentialIndices(positions.length / 3);
            const normals = primitive.attributes.NORMAL !== undefined
                ? getAccessorData(gltf, primitive.attributes.NORMAL, binaryChunk)
                : computeVertexNormals(positions, indices);
            const joints = primitive.attributes.JOINTS_0 !== undefined && primitive.attributes.WEIGHTS_0 !== undefined
                ? getAccessorData(gltf, primitive.attributes.JOINTS_0, binaryChunk)
                : null;
            const weights = joints
                ? getAccessorData(gltf, primitive.attributes.WEIGHTS_0, binaryChunk)
                : null;
            const syntheticJointIdx = instance.nodeIndex !== null
                ? jointIndexByNode.get(instance.nodeIndex)
                : undefined;
            const vertexCount = positions.length / 3;

            for (let vertexIdx = 0; vertexIdx < vertexCount; vertexIdx++) {
                const localPosition = [
                    positions[vertexIdx * 3],
                    positions[vertexIdx * 3 + 1],
                    positions[vertexIdx * 3 + 2]
                ];
                const localNormal = [
                    normals[vertexIdx * 3],
                    normals[vertexIdx * 3 + 1],
                    normals[vertexIdx * 3 + 2]
                ];
                const worldPosition = transformPosition(instance.worldMatrix, localPosition);
                const worldNormal = transformNormal(instance.worldMatrix, localNormal);

                allLocalVertices.push(...localPosition);
                allWorldVertices.push(...worldPosition);
                allLocalNormals.push(...localNormal);
                allWorldNormals.push(...worldNormal);

                if (joints && weights) {
                    allJoints.push(
                        joints[vertexIdx * 4], joints[vertexIdx * 4 + 1],
                        joints[vertexIdx * 4 + 2], joints[vertexIdx * 4 + 3]
                    );
                    allWeights.push(
                        weights[vertexIdx * 4], weights[vertexIdx * 4 + 1],
                        weights[vertexIdx * 4 + 2], weights[vertexIdx * 4 + 3]
                    );
                } else if (canAnimateWithJoints && syntheticJointIdx !== undefined) {
                    allJoints.push(syntheticJointIdx, 0, 0, 0);
                    allWeights.push(1, 0, 0, 0);
                } else {
                    allJoints.push(0, 0, 0, 0);
                    allWeights.push(1, 0, 0, 0);
                }
            }

            for (let i = 0; i < indices.length; i++) {
                allIndices.push(indices[i] + currentVertexOffset);
            }
            boundsSources.push({
                positions,
                joints,
                weights,
                nodeIndex: instance.nodeIndex
            });
            currentVertexOffset += vertexCount;
        }
    }

    if (allLocalVertices.length === 0) {
        throw new Error('GLTF file contains no renderable mesh primitives');
    }

    const bounds = createEmptyBounds();
    includePositionsInBounds(bounds, allWorldVertices);

    // Use every keyframe time from every clip. Linear TRS channels attain their
    // extrema at keyframes, so this produces one stable normalization that covers
    // the rest pose and the requested animation samples.
    if (nodes.length > 0) {
        for (const animation of animations) {
            const keyframeTimes = new Set();
            for (const channel of animation.channels) {
                for (const time of channel.times) keyframeTimes.add(time);
            }
            for (const time of keyframeTimes) {
                const animatedTransforms = sampleAnimation(animation, time, false);
                const animatedWorldMatrices = computeNodeWorldMatrices(nodes, animatedTransforms);
                const animatedJointMatrices = baseSkin
                    ? computeJointMatrices(nodes, baseSkin, animatedTransforms)
                    : null;
                for (const source of boundsSources) {
                    if (source.joints && source.weights && animatedJointMatrices) {
                        includeSkinnedPositionsInBounds(bounds, source, animatedJointMatrices);
                    } else {
                        const worldMatrix = source.nodeIndex !== null
                            ? animatedWorldMatrices[source.nodeIndex] || identityMatrix
                            : identityMatrix;
                        includeTransformedPositionsInBounds(bounds, source.positions, worldMatrix);
                    }
                }
            }
        }
    }

    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    const maxSize = Math.max(
        bounds.maxX - bounds.minX,
        bounds.maxY - bounds.minY,
        bounds.maxZ - bounds.minZ
    );
    const scale = maxSize > 0 ? 1.0 / maxSize : 1.0;

    const worldVertices = new Float32Array(allWorldVertices);
    const localVertices = new Float32Array(allLocalVertices);
    const indices = new Uint32Array(allIndices);
    const jointsData = new Uint8Array(allJoints);
    const weightsData = new Float32Array(allWeights);
    const triangleVertices = [];
    const triangleNormals = [];
    const triangleIndices = [];
    const triangleJoints = [];
    const triangleWeights = [];
    const triangleCount = indices.length / 3;

    for (let t = 0; t < triangleCount; t++) {
        const triangleVertexIndices = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
        for (let v = 0; v < 3; v++) {
            const vertexIdx = triangleVertexIndices[v];
            triangleVertices.push(
                (worldVertices[vertexIdx * 3] - centerX) * scale,
                -(worldVertices[vertexIdx * 3 + 1] - centerY) * scale,
                (worldVertices[vertexIdx * 3 + 2] - centerZ) * scale
            );
            triangleNormals.push(
                allWorldNormals[vertexIdx * 3],
                -allWorldNormals[vertexIdx * 3 + 1],
                allWorldNormals[vertexIdx * 3 + 2]
            );
            triangleJoints.push(
                jointsData[vertexIdx * 4], jointsData[vertexIdx * 4 + 1],
                jointsData[vertexIdx * 4 + 2], jointsData[vertexIdx * 4 + 3]
            );
            triangleWeights.push(
                weightsData[vertexIdx * 4], weightsData[vertexIdx * 4 + 1],
                weightsData[vertexIdx * 4 + 2], weightsData[vertexIdx * 4 + 3]
            );
            triangleIndices.push(t * 3 + v);
        }
    }

    return {
        vertices: new Float32Array(triangleVertices),
        normals: new Float32Array(triangleNormals),
        indices: new Uint32Array(triangleIndices),
        joints: new Uint8Array(triangleJoints),
        weights: new Float32Array(triangleWeights),
        hasSkinning: Boolean(canAnimateWithJoints),
        skin: jointTableFits ? resolvedSkin : null,
        triangleCount,
        boundingBox: bounds,
        normalization: { centerX, centerY, centerZ, scale },
        originalVertices: localVertices,
        originalNormals: new Float32Array(allLocalNormals),
        originalIndices: indices
    };
}

function createSequentialIndices(vertexCount) {
    const indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) indices[i] = i;
    return indices;
}

function computeVertexNormals(positions, indices) {
    const normals = new Float32Array(positions.length);
    for (let i = 0; i + 2 < indices.length; i += 3) {
        const i0 = indices[i] * 3;
        const i1 = indices[i + 1] * 3;
        const i2 = indices[i + 2] * 3;
        const edge1 = [
            positions[i1] - positions[i0],
            positions[i1 + 1] - positions[i0 + 1],
            positions[i1 + 2] - positions[i0 + 2]
        ];
        const edge2 = [
            positions[i2] - positions[i0],
            positions[i2 + 1] - positions[i0 + 1],
            positions[i2 + 2] - positions[i0 + 2]
        ];
        const faceNormal = [
            edge1[1] * edge2[2] - edge1[2] * edge2[1],
            edge1[2] * edge2[0] - edge1[0] * edge2[2],
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        ];
        for (const offset of [i0, i1, i2]) {
            normals[offset] += faceNormal[0];
            normals[offset + 1] += faceNormal[1];
            normals[offset + 2] += faceNormal[2];
        }
    }
    for (let i = 0; i < normals.length; i += 3) {
        const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]);
        if (length > 0) {
            normals[i] /= length;
            normals[i + 1] /= length;
            normals[i + 2] /= length;
        } else {
            normals[i + 1] = 1;
        }
    }
    return normals;
}

function transformPosition(matrix, position) {
    return [
        matrix[0] * position[0] + matrix[4] * position[1] + matrix[8] * position[2] + matrix[12],
        matrix[1] * position[0] + matrix[5] * position[1] + matrix[9] * position[2] + matrix[13],
        matrix[2] * position[0] + matrix[6] * position[1] + matrix[10] * position[2] + matrix[14]
    ];
}

function transformNormal(matrix, normal) {
    const column0 = [matrix[0], matrix[1], matrix[2]];
    const column1 = [matrix[4], matrix[5], matrix[6]];
    const column2 = [matrix[8], matrix[9], matrix[10]];
    const cofactor0 = cross3(column1, column2);
    const cofactor1 = cross3(column2, column0);
    const cofactor2 = cross3(column0, column1);
    const determinant = dot3(column0, cofactor0);
    if (Math.abs(determinant) < 1e-12) return normalize3(normal);

    return normalize3([
        (cofactor0[0] * normal[0] + cofactor1[0] * normal[1] + cofactor2[0] * normal[2]) / determinant,
        (cofactor0[1] * normal[0] + cofactor1[1] * normal[1] + cofactor2[1] * normal[2]) / determinant,
        (cofactor0[2] * normal[0] + cofactor1[2] * normal[1] + cofactor2[2] * normal[2]) / determinant
    ]);
}

function cross3(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function dot3(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize3(vector) {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    return length > 0
        ? [vector[0] / length, vector[1] / length, vector[2] / length]
        : [0, 1, 0];
}

function createEmptyBounds() {
    return {
        minX: Infinity, minY: Infinity, minZ: Infinity,
        maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity
    };
}

function includePositionsInBounds(bounds, positions) {
    for (let i = 0; i < positions.length; i += 3) {
        includePointInBounds(bounds, positions[i], positions[i + 1], positions[i + 2]);
    }
}

function includeTransformedPositionsInBounds(bounds, positions, matrix) {
    for (let i = 0; i < positions.length; i += 3) {
        const position = transformPosition(matrix, [positions[i], positions[i + 1], positions[i + 2]]);
        includePointInBounds(bounds, position[0], position[1], position[2]);
    }
}

function includeSkinnedPositionsInBounds(bounds, source, jointMatrices) {
    const jointCount = jointMatrices.length / 16;
    for (let vertexIdx = 0; vertexIdx < source.positions.length / 3; vertexIdx++) {
        const localPosition = [
            source.positions[vertexIdx * 3],
            source.positions[vertexIdx * 3 + 1],
            source.positions[vertexIdx * 3 + 2]
        ];
        const skinnedPosition = [0, 0, 0];
        let totalWeight = 0;
        for (let influence = 0; influence < 4; influence++) {
            const jointIdx = source.joints[vertexIdx * 4 + influence];
            const weight = source.weights[vertexIdx * 4 + influence];
            if (weight <= 0 || jointIdx >= jointCount) continue;

            const matrixOffset = jointIdx * 16;
            const matrix = jointMatrices.subarray(matrixOffset, matrixOffset + 16);
            const transformedPosition = transformPosition(matrix, localPosition);
            skinnedPosition[0] += transformedPosition[0] * weight;
            skinnedPosition[1] += transformedPosition[1] * weight;
            skinnedPosition[2] += transformedPosition[2] * weight;
            totalWeight += weight;
        }
        if (totalWeight < 0.001) {
            includePointInBounds(bounds, localPosition[0], localPosition[1], localPosition[2]);
        } else {
            includePointInBounds(bounds, skinnedPosition[0], skinnedPosition[1], skinnedPosition[2]);
        }
    }
}

function includePointInBounds(bounds, x, y, z) {
    bounds.minX = Math.min(bounds.minX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.minZ = Math.min(bounds.minZ, z);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.maxY = Math.max(bounds.maxY, y);
    bounds.maxZ = Math.max(bounds.maxZ, z);
}
/**
 * Get typed array data from a GLTF accessor
 * @param {Object} gltf - The GLTF JSON object
 * @param {number} accessorIdx - The accessor index
 * @param {Uint8Array|null} binaryChunk - The binary chunk (for GLB files)
 * @returns {TypedArray} The accessor data
 */
function getAccessorData(gltf, accessorIdx, binaryChunk) {
    const accessor = gltf.accessors[accessorIdx];
    const bufferViewIdx = accessor.bufferView;
    const bufferView = gltf.bufferViews[bufferViewIdx];

    // Get the buffer data
    let bufferData;
    if (binaryChunk) {
        bufferData = binaryChunk;
    } else {
        // For non-GLB files, we need to handle buffer data
        const buffer = gltf.buffers[bufferView.buffer];

        if (!buffer.uri) {
            // No URI means the buffer data should be in the GLB binary chunk
            throw new Error('GLTF file has no buffer URI and no binary chunk. The file may be corrupted.');
        }

        if (buffer.uri.startsWith('data:')) {
            // Embedded base64 data
            const base64Match = buffer.uri.match(/^data:[^;]*;base64,(.*)$/);
            if (!base64Match) {
                throw new Error('Invalid data URI format in GLTF buffer');
            }
            const base64 = base64Match[1];
            const binaryString = atob(base64);
            bufferData = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bufferData[i] = binaryString.charCodeAt(i);
            }
        } else {
            // External buffer URI - can't load in browser without additional file access
            throw new Error(
                `External buffer files not supported in browser.\n\n` +
                `This GLTF file references an external file: "${buffer.uri}"\n\n` +
                `Solutions:\n` +
                `1. Use GLB format instead (recommended) - it's a single self-contained file\n` +
                `2. Convert your GLTF to GLB using: npx gltf-pipeline -i model.gltf -o model.glb\n` +
                `3. Use a GLTF file with embedded base64 data`
            );
        }
    }

    // Calculate offsets
    const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
    const count = accessor.count;

    // Get component count based on type
    const typeComponentCounts = {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16
    };
    const componentCount = typeComponentCounts[accessor.type] || 1;

    // Create appropriate typed array based on component type
    const componentTypes = {
        5120: Int8Array,    // BYTE
        5121: Uint8Array,   // UNSIGNED_BYTE
        5122: Int16Array,   // SHORT
        5123: Uint16Array,  // UNSIGNED_SHORT
        5125: Uint32Array,  // UNSIGNED_INT
        5126: Float32Array  // FLOAT
    };

    const TypedArrayClass = componentTypes[accessor.componentType];
    if (!TypedArrayClass) {
        throw new Error(`Unsupported component type: ${accessor.componentType}`);
    }

    const bytesPerElement = TypedArrayClass.BYTES_PER_ELEMENT;
    const totalElements = count * componentCount;

    // Handle stride if specified
    const byteStride = bufferView.byteStride;
    if (byteStride && byteStride !== componentCount * bytesPerElement) {
        // Non-tightly packed data, need to extract manually
        const result = new TypedArrayClass(totalElements);
        for (let i = 0; i < count; i++) {
            const srcOffset = byteOffset + i * byteStride;
            const srcView = new TypedArrayClass(bufferData.buffer, bufferData.byteOffset + srcOffset, componentCount);
            result.set(srcView, i * componentCount);
        }
        return result;
    } else {
        // Tightly packed data
        return new TypedArrayClass(bufferData.buffer, bufferData.byteOffset + byteOffset, totalElements);
    }
}

// =============================================================================
// Animation Extraction
// =============================================================================

/**
 * Extract animations from GLTF file
 * @param {Object} gltf - The GLTF JSON object
 * @param {Uint8Array|null} binaryChunk - The binary chunk (for GLB files)
 * @returns {Array} Array of animation objects
 */
function extractGLTFAnimations(gltf, binaryChunk) {
    if (!gltf.animations || gltf.animations.length === 0) {
        return [];
    }

    const animations = [];

    for (let animIdx = 0; animIdx < gltf.animations.length; animIdx++) {
        const anim = gltf.animations[animIdx];
        const animName = anim.name || `Animation ${animIdx}`;

        // Parse all channels for this animation
        const channels = [];
        let duration = 0;

        for (const channel of anim.channels) {
            const sampler = anim.samplers[channel.sampler];

            // Get input (time) and output (values) data
            const inputData = getAccessorData(gltf, sampler.input, binaryChunk);
            const outputData = getAccessorData(gltf, sampler.output, binaryChunk);

            // Track max duration
            const maxTime = inputData[inputData.length - 1];
            if (maxTime > duration) {
                duration = maxTime;
            }

            // Get output type info for proper parsing
            const outputAccessor = gltf.accessors[sampler.output];

            channels.push({
                targetNode: channel.target.node,
                targetPath: channel.target.path, // 'translation', 'rotation', 'scale', 'weights'
                interpolation: sampler.interpolation || 'LINEAR', // 'LINEAR', 'STEP', 'CUBICSPLINE'
                times: Array.from(inputData),
                values: Array.from(outputData),
                outputType: outputAccessor.type // 'VEC3', 'VEC4', 'SCALAR'
            });
        }

        animations.push({
            name: animName,
            duration: duration,
            channels: channels
        });

        console.log(`Parsed animation "${animName}": ${duration.toFixed(2)}s, ${channels.length} channels`);
    }

    return animations;
}

/**
 * Sample animation at a specific time
 * @param {Object} animation - The animation object
 * @param {number} time - The time to sample
 * @param {boolean} loop - Whether to wrap time by the animation duration
 * @returns {Map} Map of node index to transform {translation, rotation, scale}
 */
function sampleAnimation(animation, time, loop = true) {
    const sampleTime = loop && animation.duration > 0 ? time % animation.duration : time;

    const nodeTransforms = new Map();

    for (const channel of animation.channels) {
        const nodeIdx = channel.targetNode;

        // Initialize transform if not exists
        if (!nodeTransforms.has(nodeIdx)) {
            nodeTransforms.set(nodeIdx, {
                translation: null,
                rotation: null,
                scale: null
            });
        }

        const transform = nodeTransforms.get(nodeIdx);

        // Find the keyframe indices to interpolate between
        const times = channel.times;
        let i0 = 0;
        let i1 = 0;

        for (let i = 0; i < times.length - 1; i++) {
            if (times[i] <= sampleTime && times[i + 1] > sampleTime) {
                i0 = i;
                i1 = i + 1;
                break;
            }
        }

        // Handle edge case at the end
        if (sampleTime >= times[times.length - 1]) {
            i0 = times.length - 1;
            i1 = times.length - 1;
        }

        // Calculate interpolation factor
        let t = 0;
        if (i0 !== i1) {
            t = (sampleTime - times[i0]) / (times[i1] - times[i0]);
        }

        // Get values based on output type
        const values = channel.values;
        let result;

        switch (channel.targetPath) {
            case 'translation':
            case 'scale': {
                // VEC3
                const v0 = [values[i0 * 3], values[i0 * 3 + 1], values[i0 * 3 + 2]];
                const v1 = [values[i1 * 3], values[i1 * 3 + 1], values[i1 * 3 + 2]];

                if (channel.interpolation === 'STEP') {
                    result = v0;
                } else {
                    // LINEAR interpolation
                    result = [
                        v0[0] + (v1[0] - v0[0]) * t,
                        v0[1] + (v1[1] - v0[1]) * t,
                        v0[2] + (v1[2] - v0[2]) * t
                    ];
                }
                break;
            }
            case 'rotation': {
                // VEC4 (quaternion)
                const q0 = [values[i0 * 4], values[i0 * 4 + 1], values[i0 * 4 + 2], values[i0 * 4 + 3]];
                const q1 = [values[i1 * 4], values[i1 * 4 + 1], values[i1 * 4 + 2], values[i1 * 4 + 3]];

                if (channel.interpolation === 'STEP') {
                    result = q0;
                } else {
                    // SLERP for quaternions
                    result = slerpQuaternion(q0, q1, t);
                }
                break;
            }
            case 'weights': {
                // Morph target weights (array of scalars)
                // For now, just linear interpolation
                const numWeights = values.length / times.length;
                result = [];
                for (let w = 0; w < numWeights; w++) {
                    const w0 = values[i0 * numWeights + w];
                    const w1 = values[i1 * numWeights + w];
                    result.push(w0 + (w1 - w0) * t);
                }
                break;
            }
        }

        transform[channel.targetPath] = result;
    }

    return nodeTransforms;
}

/**
 * Spherical linear interpolation for quaternions
 */
function slerpQuaternion(q0, q1, t) {
    // Compute cosine of angle between quaternions
    let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];

    // If negative dot, negate one quaternion to take shorter path
    const q1Copy = [...q1];
    if (dot < 0) {
        q1Copy[0] = -q1Copy[0];
        q1Copy[1] = -q1Copy[1];
        q1Copy[2] = -q1Copy[2];
        q1Copy[3] = -q1Copy[3];
        dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995) {
        const result = [
            q0[0] + (q1Copy[0] - q0[0]) * t,
            q0[1] + (q1Copy[1] - q0[1]) * t,
            q0[2] + (q1Copy[2] - q0[2]) * t,
            q0[3] + (q1Copy[3] - q0[3]) * t
        ];
        // Normalize
        const len = Math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2 + result[3] ** 2);
        return [result[0] / len, result[1] / len, result[2] / len, result[3] / len];
    }

    // Standard SLERP
    const theta0 = Math.acos(dot);
    const theta = theta0 * t;
    const sinTheta = Math.sin(theta);
    const sinTheta0 = Math.sin(theta0);

    const s0 = Math.cos(theta) - dot * sinTheta / sinTheta0;
    const s1 = sinTheta / sinTheta0;

    return [
        q0[0] * s0 + q1Copy[0] * s1,
        q0[1] * s0 + q1Copy[1] * s1,
        q0[2] * s0 + q1Copy[2] * s1,
        q0[3] * s0 + q1Copy[3] * s1
    ];
}

/**
 * Apply node transforms to compute world matrices
 * @param {Array} nodes - GLTF nodes array
 * @param {Map} animatedTransforms - Map of animated node transforms
 * @returns {Array} Array of 4x4 world matrices for each node
 */
function computeNodeWorldMatrices(nodes, animatedTransforms = new Map()) {
    const worldMatrices = [];

    // First pass: compute local matrices
    const localMatrices = nodes.map((node, idx) => {
        let translation = node.translation || [0, 0, 0];
        let rotation = node.rotation || [0, 0, 0, 1];
        let scale = node.scale || [1, 1, 1];

        // Override with animated values
        if (animatedTransforms.has(idx)) {
            const animated = animatedTransforms.get(idx);
            if (animated.translation) translation = animated.translation;
            if (animated.rotation) rotation = animated.rotation;
            if (animated.scale) scale = animated.scale;
        }

        // If node has a matrix, use it directly (but animations override)
        if (node.matrix && !animatedTransforms.has(idx)) {
            return [...node.matrix];
        }

        return composeTRSMatrix(translation, rotation, scale);
    });

    // Second pass: compute world matrices by walking hierarchy
    function computeWorld(nodeIdx, parentMatrix) {
        const localMatrix = localMatrices[nodeIdx];
        const worldMatrix = multiplyMatrices(parentMatrix, localMatrix);
        worldMatrices[nodeIdx] = worldMatrix;

        const node = nodes[nodeIdx];
        if (node.children) {
            for (const childIdx of node.children) {
                computeWorld(childIdx, worldMatrix);
            }
        }
    }

    // Find root nodes (nodes without parents)
    const hasParent = new Set();
    for (const node of nodes) {
        if (node.children) {
            for (const child of node.children) {
                hasParent.add(child);
            }
        }
    }

    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    for (let i = 0; i < nodes.length; i++) {
        if (!hasParent.has(i)) {
            computeWorld(i, identity);
        }
    }

    return worldMatrices;
}

/**
 * Compose a 4x4 transformation matrix from translation, rotation (quaternion), and scale
 */
function composeTRSMatrix(t, r, s) {
    // Quaternion to rotation matrix
    const x = r[0], y = r[1], z = r[2], w = r[3];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;

    return [
        (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
        (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
        (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
        t[0], t[1], t[2], 1
    ];
}

/**
 * Multiply two 4x4 matrices (column-major order)
 */
function multiplyMatrices(a, b) {
    const result = new Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            result[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    return result;
}

/**
 * Extract skin data (inverse bind matrices) from GLTF
 * @param {Object} gltf - The GLTF JSON object
 * @param {Uint8Array|null} binaryChunk - The binary chunk
 * @returns {Object|null} Skin data with inverseBindMatrices and joint node indices
 */
function extractGLTFSkin(gltf, binaryChunk) {
    if (gltf._resolvedSkin) {
        return gltf._resolvedSkin;
    }
    if (!gltf.skins || gltf.skins.length === 0) {
        return null;
    }

    // Use the first skin (most models only have one)
    const skin = gltf.skins[0];

    // Get inverse bind matrices
    let inverseBindMatrices = [];
    if (skin.inverseBindMatrices !== undefined) {
        const ibmData = getAccessorData(gltf, skin.inverseBindMatrices, binaryChunk);
        // Convert to array of 16-element arrays (4x4 matrices)
        for (let i = 0; i < skin.joints.length; i++) {
            const matrix = [];
            for (let j = 0; j < 16; j++) {
                matrix.push(ibmData[i * 16 + j]);
            }
            inverseBindMatrices.push(matrix);
        }
    } else {
        // Default to identity matrices
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        for (let i = 0; i < skin.joints.length; i++) {
            inverseBindMatrices.push([...identity]);
        }
    }

    console.log(`Parsed skin: ${skin.joints.length} joints`);

    return {
        joints: skin.joints,  // Array of node indices
        skeleton: skin.skeleton,  // Optional root node
        inverseBindMatrices: inverseBindMatrices,
        jointCount: skin.joints.length
    };
}

/**
 * Compute joint matrices for GPU skinning
 * This computes: jointMatrix[i] = globalTransform[jointNode[i]] * inverseBindMatrix[i]
 * The shader will handle coordinate normalization after skinning
 * @param {Array} nodes - GLTF nodes
 * @param {Object} skin - Skin data
 * @param {Map} animatedTransforms - Animated node transforms
 * @returns {Float32Array} Flat array of joint matrices (16 floats per joint)
 */
function computeJointMatrices(nodes, skin, animatedTransforms) {
    // Compute world matrices for all nodes
    const worldMatrices = computeNodeWorldMatrices(nodes, animatedTransforms);

    // Simple approach: just compute worldMatrix * inverseBindMatrix
    // The shader receives ORIGINAL vertices and will normalize after skinning
    const jointMatrices = new Float32Array(skin.jointCount * 16);

    for (let i = 0; i < skin.jointCount; i++) {
        const jointNodeIdx = skin.joints[i];
        const worldMatrix = worldMatrices[jointNodeIdx] || [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const ibm = skin.inverseBindMatrices[i];

        // jointMatrix = worldMatrix * inverseBindMatrix
        const jointMatrix = multiplyMatrices(worldMatrix, ibm);

        // Copy to output
        for (let j = 0; j < 16; j++) {
            jointMatrices[i * 16 + j] = jointMatrix[j];
        }
    }

    return jointMatrices;
}
