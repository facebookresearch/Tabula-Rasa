/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Tabula Rasa - WebGPU implementation
 *
 * Mirrors the Python renderer:
 * - Count-Unique Estimator with histogram-based variance reduction
 * - 2D Hammersley sampling for spatial jitter
 * - Variance correction: 1/sqrt(sum(weights²))
 */

// The workgroup-size variant compiled during init. The sample-count slider defaults to
// 256, so this is the only variant the first frame needs.
const DEFAULT_WORKGROUP_SIZE = 256;

// Field types understood by the shaders: 0 = White, 1 = Blended Adaptive, 2 = Adaptive.
const VALID_FIELD_TYPES = [0, 1, 2];
const DEFAULT_FIELD_TYPE = 1;
// Keep the built-in cube off exact field-cell planes without changing UI scale semantics.
const DEFAULT_MESH_GEOMETRY_SCALE = 1.0001;

// The mesh shown before the user loads their own. Emitted as Wavefront OBJ text so
// it travels the same parseOBJ -> loadMesh path as a dropped file, rather than being
// a second, special-cased geometry source.
const DEFAULT_MESH_OBJ = `# Unit cube
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
vn  0.0  0.0 -1.0
vn  0.0  0.0  1.0
vn -1.0  0.0  0.0
vn  1.0  0.0  0.0
vn  0.0 -1.0  0.0
vn  0.0  1.0  0.0
f 1//1 3//1 2//1
f 1//1 4//1 3//1
f 5//2 6//2 7//2
f 5//2 7//2 8//2
f 1//3 5//3 8//3
f 1//3 8//3 4//3
f 2//4 3//4 7//4
f 2//4 7//4 6//4
f 1//5 2//5 6//5
f 1//5 6//5 5//5
f 4//6 8//6 7//6
f 4//6 7//6 3//6
`;

// =============================================================================
// Mesh Generation Utilities
// =============================================================================


/**
 * Generate a cube mesh (for background skybox)
 * Cube is centered at origin with given size
 * Normals point INWARD (for viewing from inside)
 * @param {number} size - Half-size of the cube (extends from -size to +size)
 * @returns {Object} - { positions, normals, indices }
 */
function generateCubeMesh(size = 10.0) {
    // 6 faces, 4 vertices each = 24 vertices
    // Each face has normals pointing inward (we're inside the cube)
    const positions = [];
    const normals = [];
    const indices = [];

    // Face definitions: [normal direction (inward), up direction, right direction]
    const faces = [
        // Front face (looking at +Z from inside, normal points -Z)
        { center: [0, 0, size], normal: [0, 0, -1], up: [0, 1, 0], right: [1, 0, 0] },
        // Back face (looking at -Z from inside, normal points +Z)
        { center: [0, 0, -size], normal: [0, 0, 1], up: [0, 1, 0], right: [-1, 0, 0] },
        // Right face (looking at +X from inside, normal points -X)
        { center: [size, 0, 0], normal: [-1, 0, 0], up: [0, 1, 0], right: [0, 0, -1] },
        // Left face (looking at -X from inside, normal points +X)
        { center: [-size, 0, 0], normal: [1, 0, 0], up: [0, 1, 0], right: [0, 0, 1] },
        // Top face (looking at +Y from inside, normal points -Y)
        { center: [0, size, 0], normal: [0, -1, 0], up: [0, 0, 1], right: [1, 0, 0] },
        // Bottom face (looking at -Y from inside, normal points +Y)
        { center: [0, -size, 0], normal: [0, 1, 0], up: [0, 0, -1], right: [1, 0, 0] },
    ];

    for (let f = 0; f < faces.length; f++) {
        const face = faces[f];
        const baseIndex = f * 4;

        // Generate 4 corners of the face
        // Corners: (-1,-1), (1,-1), (1,1), (-1,1) in face-local coordinates
        const corners = [
            [-1, -1], [1, -1], [1, 1], [-1, 1]
        ];

        for (let c = 0; c < 4; c++) {
            const [u, v] = corners[c];
            // Position = center + u*right*size + v*up*size
            const px = face.center[0] + u * face.right[0] * size + v * face.up[0] * size;
            const py = face.center[1] + u * face.right[1] * size + v * face.up[1] * size;
            const pz = face.center[2] + u * face.right[2] * size + v * face.up[2] * size;

            positions.push(px, py, pz);
            normals.push(face.normal[0], face.normal[1], face.normal[2]);
        }

        // Two triangles per face - clockwise winding when viewed from inside
        indices.push(baseIndex, baseIndex + 1, baseIndex + 2);
        indices.push(baseIndex, baseIndex + 2, baseIndex + 3);
    }

    return {
        positions: new Float32Array(positions),
        normals: new Float32Array(normals),
        indices: new Uint32Array(indices),
        vertexCount: 24,
        indexCount: 36,
    };
}

// =============================================================================
// Histogram Visualization
// =============================================================================

class HistogramVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.outputMode = 0;  // 0 = Final Noise, 1+ = other modes
        this.binCount = 64;   // Number of bars drawn in the debug histogram
    }

    setOutputMode(mode) {
        this.outputMode = mode;
    }

    render(sourceCanvas) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        if (!sourceCanvas) return;

        // A WebGPU canvas cannot also acquire a 2D context, so copy it through a
        // temporary 2D canvas before reading its pixels.
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = sourceCanvas.width;
        tempCanvas.height = sourceCanvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(sourceCanvas, 0, 0);

        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        this.renderHistogramFromData(imageData.data);
    }

    renderHistogramFromData(pixels) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Reserve space for x-axis labels
        const margin = { bottom: 20, left: 5, right: 5 };
        const plotHeight = height - margin.bottom;

        // Only apply tonemapping conversion for noise output (mode 0)
        // Other modes output values directly in [0, 1] range
        const isNoiseMode = this.outputMode === 0;

        // Histogram range depends on output mode
        const minVal = isNoiseMode ? -3 : 0;
        const maxVal = isNoiseMode ? 3 : 1;
        const range = maxVal - minVal;

        const binCount = this.binCount;
        const bins = new Array(binCount).fill(0);

        // Count pixel values into bins (using all RGB channels)
        for (let i = 0; i < pixels.length; i += 4) {
            // Process R, G, B channels (skip alpha)
            for (let c = 0; c < 3; c++) {
                const pixelValue = pixels[i + c] / 255;

                // For noise mode, convert from tonemapped [0,1] back to raw N(0,1)
                // For other modes, use the value directly
                const value = isNoiseMode
                    ? (pixelValue - 0.5) / 0.166666  // Convert back to N(0,1)
                    : pixelValue;

                // Map value to bin index
                const normalizedPos = (value - minVal) / range;
                const bin = Math.min(Math.max(0, Math.floor(normalizedPos * binCount)), binCount - 1);
                bins[bin]++;
            }
        }

        // Normalize
        const maxCount = Math.max(...bins);
        if (maxCount === 0) return;

        const normalizedBins = bins.map(b => b / maxCount);

        // Draw histogram bars
        const barWidth = width / binCount;
        ctx.fillStyle = '#1266a0';  // 6.1:1 on white

        for (let i = 0; i < binCount; i++) {
            const barHeight = normalizedBins[i] * plotHeight * 0.9;
            const x = i * barWidth;
            const y = plotHeight - barHeight;
            ctx.fillRect(x, y, barWidth - 1, barHeight);
        }

        // Draw x-axis line
        ctx.strokeStyle = '#767676';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, plotHeight);
        ctx.lineTo(width, plotHeight);
        ctx.stroke();

        // Draw reference Gaussian curve (standard normal N(0,1)) - ABOVE bars in orange
        ctx.strokeStyle = '#c2410c';  // 5.2:1 on white
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (let i = 0; i < binCount; i++) {
            const x = (i + 0.5) / binCount * width;  // Center of each bin
            const rawVal = minVal + (i + 0.5) / binCount * range;  // Center value of bin
            const gaussian = Math.exp(-rawVal * rawVal / 2);  // Standard normal PDF (unnormalized)
            const y = plotHeight - gaussian * plotHeight * 0.9;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw x-axis labels and tick marks (raw N(0,1) values)
        ctx.fillStyle = '#595959';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';

        // Label positions: -3, -2, -1, 0, 1, 2, 3
        const tickValues = [-3, -2, -1, 0, 1, 2, 3];
        for (const val of tickValues) {
            const x = ((val - minVal) / range) * width;

            // Draw tick mark
            ctx.beginPath();
            ctx.moveTo(x, plotHeight);
            ctx.lineTo(x, plotHeight + 4);
            ctx.stroke();

            // Draw label
            ctx.fillText(val.toString(), x, plotHeight + 14);
        }

        // Draw center line at 0 (the mean of N(0,1))
        const zeroX = ((0 - minVal) / range) * width;
        ctx.strokeStyle = '#767676';
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(zeroX, 0);
        ctx.lineTo(zeroX, plotHeight);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// =============================================================================
// WebGPU Renderer Class
// =============================================================================

class WebGPUNoiseRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.device = null;
        this.context = null;

        // Render resolution may be non-square; the 3D noise field is isotropic.
        this.resolutionX = 512;
        this.resolutionY = 512;
        this.fieldResolution = 512;
        this.sampleCount = 256;  // Match HTML default (max 256 due to workgroup size)
        this.time = 0;
        this.fieldSeed = 0.0;  // Seed for noise field (adjustable via UI)

        // Output mode: 0 = Final Noise; 3-9 select a G-buffer channel
        this.outputMode = 0;

        // Field type: 0 = White, 1 = Blended Adaptive, 2 = Adaptive
        this.fieldType = DEFAULT_FIELD_TYPE;

        // RNG type: 0 = Threefry (JAX-compatible), 1 = PCG (fast)
        this.rngType = 1;

        // Estimator mode: 0 = Biased (N samples), 1 = Unbiased (2N samples)
        this.estimatorMode = 0;

        // Histogram bin count for sampling (1-256, default 256)
        this.histogramBinCount = 256;

        // Custom mesh parameters
        this.customMesh = null;  // { vertices: Float32Array, indices: Uint32Array, normals: Float32Array }
        this.meshCameraYaw = 0;  // degrees
        this.meshCameraPitch = 0;  // degrees
        this.meshCameraYawRotationSpeed = 5;  // degrees per second
        this.meshFov = 60.0;  // vertical field of view in degrees
        this.meshScale = 1.0;
        this.centerBackgroundCube = true;

        // Export state - when exporting, use frame-based control instead of time-based
        this.isExporting = false;
        this.exportFrame = 0;      // Current frame being exported (0-indexed)
        this.exportFps = 30;       // Export FPS for computing animation time
    }

    setOutputMode(mode) { this.outputMode = mode; }

    // Reject values outside the supported range instead of passing them to the shader,
    // where an unknown field type silently falls through to the White field. Settings
    // saved before the Soft Adaptive field was removed can still hold a stale 3.
    setFieldType(type) {
        this.fieldType = VALID_FIELD_TYPES.includes(type) ? type : DEFAULT_FIELD_TYPE;
    }

    // Load custom mesh from file data (supports OBJ text or GLTF/GLB ArrayBuffer)
    async loadMesh(data, filename = 'mesh.obj', { geometryScale = 1.0 } = {}) {
        let mesh;
        const lowerFilename = filename.toLowerCase();

        if (lowerFilename.endsWith('.gltf') || lowerFilename.endsWith('.glb')) {
            // GLTF/GLB format - data should be ArrayBuffer
            mesh = await parseGLTF(data, filename);
        } else {
            // OBJ format - data should be text string
            mesh = parseOBJ(data);
            // OBJ files don't have animations
            mesh.animations = [];
        }

        if (geometryScale !== 1.0) {
            for (let i = 0; i < mesh.vertices.length; i++) {
                mesh.vertices[i] *= geometryScale;
            }
        }

        this.customMesh = mesh;

        // Animation state
        this.meshAnimations = mesh.animations || [];
        this.currentAnimationIndex = -1;  // -1 means no animation (static pose)
        this.animationSpeed = 1.0;

        console.log(`Loaded mesh: ${mesh.triangleCount} triangles, ${mesh.vertices.length / 3} vertices`);
        if (this.meshAnimations.length > 0) {
            console.log(`  Animations: ${this.meshAnimations.map(a => a.name).join(', ')}`);
        }

        // Create GPU buffers for the mesh if device is ready
        if (this.device) {
            await this.createMeshBuffers();
        }

        return mesh;
    }

    // Set the current animation by index (-1 for static pose)
    setAnimation(index) {
        if (index < -1 || index >= this.meshAnimations.length) {
            console.warn(`Invalid animation index: ${index}`);
            return;
        }
        this.currentAnimationIndex = index;

        if (index >= 0) {
            const anim = this.meshAnimations[index];
            console.log(`Playing animation: "${anim.name}" (${anim.duration.toFixed(2)}s)`);
        } else {
            console.log('Animation disabled (static pose)');
        }
    }

    // Set animation playback speed
    setAnimationSpeed(speed) {
        this.animationSpeed = speed;
    }

    // Update animation state from absolute time (consistent with export)
    // This replaces the deltaTime-based approach for consistency
    updateAnimationFromTime(time) {
        if (this.currentAnimationIndex < 0 || !this.meshAnimations || !this.meshAnimations.length) {
            return false;  // No animation active
        }

        const animation = this.meshAnimations[this.currentAnimationIndex];

        // Compute animation time the same way as export:
        // scaledTime = effectiveTime * animationSpeed
        // animTime = scaledTime % duration
        const scaledTime = time * this.animationSpeed;
        const animTime = scaledTime % animation.duration;

        // Sample the animation and update mesh vertices
        this.applyAnimation(animation, animTime);

        return true;  // Animation is active
    }

    // Apply animation transforms to mesh vertices
    applyAnimation(animation, time) {
        if (!this.customMesh || !this.customMesh.gltf || !this.meshSkin) {
            return;  // No GLTF data or skin to animate
        }

        const gltf = this.customMesh.gltf;
        const nodes = gltf.nodes || [];

        // Sample animation to get node transforms
        const nodeTransforms = sampleAnimation(animation, time);

        // Compute joint matrices for GPU skinning
        const jointMatrices = computeJointMatrices(
            nodes,
            this.meshSkin,
            nodeTransforms
        );

        // Upload joint matrices to GPU
        if (this.jointMatricesBuffer && this.device) {
            this.device.queue.writeBuffer(this.jointMatricesBuffer, 0, jointMatrices);
        }
    }

    async createMeshBuffers() {
        if (!this.customMesh || !this.device) return;

        // =============================================================================
        // Create vertex/index buffers for mesh rasterization pipeline
        // Uses interleaved vertex format: position(3) + normal(3)
        // For skinned meshes: position(3) + normal(3) + joints(4 u8) + weights(4)
        // =============================================================================

        // The customMesh has:
        // - vertices: Float32Array of positions (3 floats per vertex, 9 floats per triangle)
        // - normals: Float32Array of normals (3 floats per vertex, 9 floats per triangle)
        // - indices: Uint32Array of indices
        // - joints: Uint8Array of joint indices (4 per vertex) - for skinned meshes
        // - weights: Float32Array of joint weights (4 per vertex) - for skinned meshes

        const triangleCount = this.customMesh.triangleCount;
        const vertexCount = triangleCount * 3;
        const hasSkinning = this.customMesh.hasSkinning && this.customMesh.joints && this.customMesh.weights;

        // Track if this mesh has skinning for rendering decisions
        this.meshHasSkinning = hasSkinning;

        if (hasSkinning) {
            // Create skinned vertex buffer with ORIGINAL (non-normalized) vertices
            // The shader will apply skinning in original space, then normalize the result
            // This is cleaner because inverse bind matrices expect original GLTF coordinates
            //
            // Layout: position(3) + normal(3) + joints(4 u8) + weights(4) = 44 bytes per vertex
            const bytesPerVertex = 44;
            const skinedMeshVertexData = new ArrayBuffer(vertexCount * bytesPerVertex);
            const floatView = new Float32Array(skinedMeshVertexData);
            const uint8View = new Uint8Array(skinedMeshVertexData);

            // We need to expand original indexed vertices to triangle list format
            const origVerts = this.customMesh.originalVertices;
            const origNormals = this.customMesh.originalNormals;
            const origIndices = this.customMesh.originalIndices;

            for (let t = 0; t < triangleCount; t++) {
                const i0 = origIndices[t * 3];
                const i1 = origIndices[t * 3 + 1];
                const i2 = origIndices[t * 3 + 2];

                const vertexIndices = [i0, i1, i2];

                for (let v = 0; v < 3; v++) {
                    const origIdx = vertexIndices[v];
                    const outIdx = t * 3 + v;
                    const byteOffset = outIdx * bytesPerVertex;
                    const floatOffset = byteOffset / 4;

                    // Position - use ORIGINAL coordinates (not normalized)
                    floatView[floatOffset + 0] = origVerts[origIdx * 3 + 0];
                    floatView[floatOffset + 1] = origVerts[origIdx * 3 + 1];
                    floatView[floatOffset + 2] = origVerts[origIdx * 3 + 2];

                    // Normal - use original normals
                    if (origNormals && origNormals.length > 0) {
                        floatView[floatOffset + 3] = origNormals[origIdx * 3 + 0];
                        floatView[floatOffset + 4] = origNormals[origIdx * 3 + 1];
                        floatView[floatOffset + 5] = origNormals[origIdx * 3 + 2];
                    } else {
                        floatView[floatOffset + 3] = 0;
                        floatView[floatOffset + 4] = 1;
                        floatView[floatOffset + 5] = 0;
                    }

                    // Joints (4 u8 at offset 24)
                    const jointBase = outIdx * 4;
                    uint8View[byteOffset + 24] = this.customMesh.joints[jointBase + 0];
                    uint8View[byteOffset + 25] = this.customMesh.joints[jointBase + 1];
                    uint8View[byteOffset + 26] = this.customMesh.joints[jointBase + 2];
                    uint8View[byteOffset + 27] = this.customMesh.joints[jointBase + 3];

                    // Weights (4 floats at offset 28)
                    const weightOffset = (byteOffset + 28) / 4;
                    floatView[weightOffset + 0] = this.customMesh.weights[jointBase + 0];
                    floatView[weightOffset + 1] = this.customMesh.weights[jointBase + 1];
                    floatView[weightOffset + 2] = this.customMesh.weights[jointBase + 2];
                    floatView[weightOffset + 3] = this.customMesh.weights[jointBase + 3];
                }
            }

            // Clean up old buffers
            if (this.skinnedMeshRasterVertexBuffer) {
                this.skinnedMeshRasterVertexBuffer.destroy();
            }

            this.skinnedMeshRasterVertexBuffer = this.device.createBuffer({
                label: 'Skinned Mesh Rasterization Vertex Buffer',
                size: skinedMeshVertexData.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(this.skinnedMeshRasterVertexBuffer, 0, skinedMeshVertexData);

            console.log(`Created skinned mesh vertex buffer: ${vertexCount} vertices with ORIGINAL coordinates`);

            // Also extract skin data for animation
            if (this.customMesh.gltf && this.customMesh.gltf.skins) {
                this.meshSkin = extractGLTFSkin(this.customMesh.gltf, this.customMesh.binaryChunk);
                console.log(`Extracted skin with ${this.meshSkin.jointCount} joints`);
            }
        }

        // Create interleaved vertex buffer (position + normal) for non-skinned meshes or as fallback
        const customMeshVertexData = new Float32Array(vertexCount * 6);
        for (let i = 0; i < vertexCount; i++) {
            const posBase = i * 3;
            const outBase = i * 6;

            // Position
            customMeshVertexData[outBase + 0] = this.customMesh.vertices[posBase + 0];
            customMeshVertexData[outBase + 1] = this.customMesh.vertices[posBase + 1];
            customMeshVertexData[outBase + 2] = this.customMesh.vertices[posBase + 2];

            // Normal
            customMeshVertexData[outBase + 3] = this.customMesh.normals[posBase + 0];
            customMeshVertexData[outBase + 4] = this.customMesh.normals[posBase + 1];
            customMeshVertexData[outBase + 5] = this.customMesh.normals[posBase + 2];
        }

        // Clean up old custom mesh rasterization buffers if they exist
        if (this.customMeshRasterVertexBuffer) {
            this.customMeshRasterVertexBuffer.destroy();
        }
        if (this.customMeshRasterIndexBuffer) {
            this.customMeshRasterIndexBuffer.destroy();
        }

        this.customMeshRasterVertexBuffer = this.device.createBuffer({
            label: 'Custom Mesh Rasterization Vertex Buffer',
            size: customMeshVertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.customMeshRasterVertexBuffer, 0, customMeshVertexData);

        this.customMeshRasterIndexBuffer = this.device.createBuffer({
            label: 'Custom Mesh Rasterization Index Buffer',
            size: this.customMesh.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.customMeshRasterIndexBuffer, 0, this.customMesh.indices);

        this.customMeshRasterIndexCount = this.customMesh.indices.length;

        console.log(`Created custom mesh rasterization buffers: ${vertexCount} vertices, ${this.customMeshRasterIndexCount} indices`);

        console.log('Mesh GPU buffers created');
    }

    async loadShader(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load shader: ${url}`);
        }
        return await response.text();
    }

    async loadShaderWithIncludes(url) {
        console.log('Fetching shader with preprocessor:', url);
        if (!this.preprocessor) {
            this.preprocessor = new WGSLPreprocessorWithGuards('');
        }
        this.preprocessor.resetGuards();
        return await this.preprocessor.process(url);
    }

    async init() {
        if (!navigator.gpu) {
            throw new Error('WebGPU is not supported in this browser');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('Failed to get WebGPU adapter');
        }

        // Log adapter info and limits
        console.log('=== WebGPU Adapter Limits ===');
        // Try different methods to get adapter info (API varies by browser)
        if (adapter.info) {
            console.log('Adapter:', adapter.info.vendor, adapter.info.architecture, adapter.info.device, adapter.info.description);
        }
        console.log('Compute limits:');
        console.log('  maxComputeInvocationsPerWorkgroup:', adapter.limits.maxComputeInvocationsPerWorkgroup);
        console.log('  maxComputeWorkgroupSizeX:', adapter.limits.maxComputeWorkgroupSizeX);
        console.log('  maxComputeWorkgroupSizeY:', adapter.limits.maxComputeWorkgroupSizeY);
        console.log('  maxComputeWorkgroupSizeZ:', adapter.limits.maxComputeWorkgroupSizeZ);
        console.log('  maxComputeWorkgroupsPerDimension:', adapter.limits.maxComputeWorkgroupsPerDimension);
        console.log('  maxComputeWorkgroupStorageSize:', adapter.limits.maxComputeWorkgroupStorageSize, 'bytes');
        console.log('  maxStorageBufferBindingSize:', adapter.limits.maxStorageBufferBindingSize, 'bytes');
        console.log('=============================');

        // Request timestamp query feature if available
        const requiredFeatures = [];
        if (adapter.features.has('timestamp-query')) {
            requiredFeatures.push('timestamp-query');
        }

        // Request higher limits for adaptive workgroup sizes
        // We need larger workgroup sizes (up to 1024) and more shared memory
        const requiredLimits = {
            maxComputeWorkgroupSizeX: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeX),
            maxComputeWorkgroupSizeY: Math.min(1024, adapter.limits.maxComputeWorkgroupSizeY),
            maxComputeWorkgroupSizeZ: Math.min(64, adapter.limits.maxComputeWorkgroupSizeZ),
            maxComputeInvocationsPerWorkgroup: Math.min(1024, adapter.limits.maxComputeInvocationsPerWorkgroup),
            maxComputeWorkgroupStorageSize: Math.min(32768, adapter.limits.maxComputeWorkgroupStorageSize),
        };
        console.log('Requesting limits:', requiredLimits);

        this.device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits,
        });

        this.hasTimestampQuery = this.device.features.has('timestamp-query');

        // Setup timestamp query if supported
        if (this.hasTimestampQuery) {
            this.timestampQuerySet = this.device.createQuerySet({
                type: 'timestamp',
                count: 2,
            });
            this.timestampBuffer = this.device.createBuffer({
                size: 16, // 2 x 8 bytes for 2 timestamps
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            // Double-buffer for async readback
            this.timestampReadBuffers = [
                this.device.createBuffer({
                    size: 16,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
                this.device.createBuffer({
                    size: 16,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                }),
            ];
            this.timestampBufferIndex = 0;
            this.timestampBufferMapped = [false, false];
            this.lastGpuTime = 0;
        }

        this.device.addEventListener('uncapturederror', (event) => {
            console.error('WebGPU uncaptured error:', event.error);
        });

        this.context = this.canvas.getContext('webgpu');
        if (!this.context) {
            throw new Error('Failed to get WebGPU canvas context');
        }

        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
            alphaMode: 'premultiplied',
        });

        // Load the shader that needs preprocessing (it #includes common.wgsl)
        const meshRasterizerSource = await this.loadShaderWithIncludes('shaders/rasterizer_mesh.wgsl');

        // Load other shaders in parallel (they don't use the preprocessor)
        const [computeSource, renderSource, tonemapSource] = await Promise.all([
            this.loadShader('shaders/noise.wgsl'),
            this.loadShader('shaders/render.wgsl'),
            this.loadShader('shaders/tonemap.wgsl')
        ]);

        this.meshRasterizerModule = this.device.createShaderModule({
            label: 'Mesh Rasterizer Shader (Vertex/Fragment)',
            code: meshRasterizerSource,
        });

        this.computeModule = this.device.createShaderModule({
            label: 'Compute Shader',
            code: computeSource,
        });

        this.renderModule = this.device.createShaderModule({
            label: 'Render Shader',
            code: renderSource,
        });

        this.tonemapModule = this.device.createShaderModule({
            label: 'Tonemap Shader',
            code: tonemapSource,
        });

        await this.createResources();
        console.log('WebGPU initialized successfully');
    }

    /**
     * Create the GPU buffers for the background cube used by the mesh rasterizer.
     */
    async createMeshRasterizationBuffers() {
        // Generate background cube mesh (large cube for skybox)
        this.cubeMesh = generateCubeMesh(10.0);  // Large cube extending 10 units in each direction
        console.log(`Generated cube mesh: ${this.cubeMesh.vertexCount} vertices, ${this.cubeMesh.indexCount} indices`);

        // Create interleaved position and normal data for the cube
        const cubeVertexData = new Float32Array(this.cubeMesh.vertexCount * 6);
        for (let i = 0; i < this.cubeMesh.vertexCount; i++) {
            const baseIn = i * 3;
            const baseOut = i * 6;
            cubeVertexData[baseOut + 0] = this.cubeMesh.positions[baseIn + 0];
            cubeVertexData[baseOut + 1] = this.cubeMesh.positions[baseIn + 1];
            cubeVertexData[baseOut + 2] = this.cubeMesh.positions[baseIn + 2];
            cubeVertexData[baseOut + 3] = this.cubeMesh.normals[baseIn + 0];
            cubeVertexData[baseOut + 4] = this.cubeMesh.normals[baseIn + 1];
            cubeVertexData[baseOut + 5] = this.cubeMesh.normals[baseIn + 2];
        }

        this.cubeVertexBuffer = this.device.createBuffer({
            label: 'Cube Vertex Buffer',
            size: cubeVertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.cubeVertexBuffer, 0, cubeVertexData);

        this.cubeIndexBuffer = this.device.createBuffer({
            label: 'Cube Index Buffer',
            size: this.cubeMesh.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.cubeIndexBuffer, 0, this.cubeMesh.indices);
    }

    async createResources() {
        this.paramsBuffer = this.device.createBuffer({
            label: 'Params Buffer',
            size: 64,  // 16 floats * 4 bytes = 64
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Camera buffer: 16 floats for camera data
        // camPos(3), camDir(3), camRight(3), camUp(3), fov, near, far, pad
        this.cameraBuffer = this.device.createBuffer({
            label: 'Camera Buffer',
            size: 64,  // 16 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        await this.createMeshRasterizationBuffers();

        this.createSizeDependentTextures();

        // =============================================================================
        // Mesh Rasterization Pipeline - True GPU rasterization with actual geometry
        // Uses vertex/fragment shaders to render spheres and quads instead of raymarching
        // =============================================================================

        // Create joint matrices buffer for skeletal animation (128 joints * 16 floats * 4 bytes = 8192 bytes)
        this.jointMatricesBuffer = this.device.createBuffer({
            label: 'Joint Matrices Buffer',
            size: 128 * 16 * 4,  // 128 joints * mat4x4 (16 floats) * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Initialize with identity matrices
        const identityMatrices = new Float32Array(128 * 16);
        for (let i = 0; i < 128; i++) {
            const offset = i * 16;
            identityMatrices[offset + 0] = 1;  // mat[0][0]
            identityMatrices[offset + 5] = 1;  // mat[1][1]
            identityMatrices[offset + 10] = 1; // mat[2][2]
            identityMatrices[offset + 15] = 1; // mat[3][3]
        }
        this.device.queue.writeBuffer(this.jointMatricesBuffer, 0, identityMatrices);

        this.meshRasterizerBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Mesh Rasterizer Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },  // Params
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },  // Camera
                { binding: 6, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },  // JointMatrices (for skeletal animation)
            ],
        });

        // Vertex buffer layout: interleaved position(3) + normal(3) = 6 floats per vertex
        const meshVertexBufferLayout = {
            arrayStride: 6 * 4,
            attributes: [
                { shaderLocation: 0, offset: 0,     format: 'float32x3' },  // position
                { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },  // normal
            ],
        };

        // Skinned vertex layout: position(3) + normal(3) + joints(4 u8) + weights(4) = 44 bytes
        const skinnedVertexBufferLayout = {
            arrayStride: 44,
            attributes: [
                { shaderLocation: 0, offset: 0,      format: 'float32x3' },  // position
                { shaderLocation: 1, offset: 12,     format: 'float32x3' },  // normal
                { shaderLocation: 3, offset: 24,     format: 'uint8x4' },    // joints (4 u8 as vec4<u32>)
                { shaderLocation: 4, offset: 28,     format: 'float32x4' },  // weights
            ],
        };

        this.meshRasterizerPipeline = this.device.createRenderPipeline({
            label: 'Mesh Rasterizer Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.meshRasterizerBindGroupLayout],
            }),
            vertex: {
                module: this.meshRasterizerModule,
                entryPoint: 'vertexMain',
                buffers: [meshVertexBufferLayout],
            },
            fragment: {
                module: this.meshRasterizerModule,
                entryPoint: 'fragmentMain',
                targets: [
                    { format: 'rgba32float' },  // Position texture
                    { format: 'rgba32float' },  // Footprint texture
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',  // No culling - floor quad needs to be visible from both sides
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });

        this.meshRasterizerBindGroup = this.device.createBindGroup({
            label: 'Mesh Rasterizer Bind Group',
            layout: this.meshRasterizerBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 2, resource: { buffer: this.cameraBuffer } },
                { binding: 6, resource: { buffer: this.jointMatricesBuffer } },
            ],
        });

        // Skinned mesh pipeline - uses same layout but skinned vertex shader and different vertex buffer
        this.skinnedMeshPipeline = this.device.createRenderPipeline({
            label: 'Skinned Mesh Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.meshRasterizerBindGroupLayout],
            }),
            vertex: {
                module: this.meshRasterizerModule,
                entryPoint: 'vertexMainSkinned',
                buffers: [skinnedVertexBufferLayout],
            },
            fragment: {
                module: this.meshRasterizerModule,
                entryPoint: 'fragmentMain',
                targets: [
                    { format: 'rgba32float' },  // Position texture
                    { format: 'rgba32float' },  // Footprint texture
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });

        // Background cube pipeline - uses same layout but different vertex shader entry point
        // Uses same settings as main mesh pipeline (cullMode: none, depthCompare: less)
        this.cubePipeline = this.device.createRenderPipeline({
            label: 'Background Cube Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.meshRasterizerBindGroupLayout],
            }),
            vertex: {
                module: this.meshRasterizerModule,
                entryPoint: 'vertexMainCube',
                buffers: [meshVertexBufferLayout],
            },
            fragment: {
                module: this.meshRasterizerModule,
                entryPoint: 'fragmentMain',
                targets: [
                    { format: 'rgba32float' },  // Position texture
                    { format: 'rgba32float' },  // Footprint texture
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',  // Same as main mesh - no culling
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',  // Same as main mesh
                format: 'depth24plus',
            },
        });

        console.log('Mesh rasterization pipeline created for true GPU rasterization');

        // Compute Pipeline - outputs to noiseTextures (rgba32float x 4 = 16 channels)
        // Create multiple pipeline variants with different workgroup sizes for adaptive performance
        this.computeBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Compute Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },  // noiseTexture0 (channels 0-3)
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // camera for world-space position
                { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },  // noiseTexture1 (channels 4-7)
                { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },  // noiseTexture2 (channels 8-11)
                { binding: 9, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },  // noiseTexture3 (channels 12-15)
            ],
        });

        this.computePipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.computeBindGroupLayout],
        });

        // Workgroup sizes: 32, 64, 128, 256. Each is a separate backend compile of
        // noise.wgsl, because WORKGROUP_SIZE feeds @workgroup_size(), and that shader
        // costs seconds to compile. Building all four here delays the first frame by
        // roughly the sum, so only the default is built now and warmComputePipelines()
        // fills in the rest once a frame is on screen.
        this.workgroupSizes = [32, 64, 128, 256];
        this.computePipelines = {};
        this.pendingComputePipelines = {};
        this.failedComputePipelines = new Set();

        this.computePipelines[DEFAULT_WORKGROUP_SIZE] = this.device.createComputePipeline({
            label: `Compute Pipeline (workgroup=${DEFAULT_WORKGROUP_SIZE})`,
            layout: this.computePipelineLayout,
            compute: {
                module: this.computeModule,
                entryPoint: 'main',
                constants: {
                    WORKGROUP_SIZE: DEFAULT_WORKGROUP_SIZE,
                },
            },
        });

        console.log(`Created compute pipeline for workgroup size ${DEFAULT_WORKGROUP_SIZE}; other sizes compile on demand`);

        // Tonemap Pipeline - converts noiseTexture (float) to outputTexture (rgba8unorm)
        this.tonemapBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Tonemap Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // Params buffer for outputMode
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } },
            ],
        });

        this.tonemapPipeline = this.device.createComputePipeline({
            label: 'Tonemap Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.tonemapBindGroupLayout],
            }),
            compute: {
                module: this.tonemapModule,
                entryPoint: 'main',
            },
        });

        // Render Pipeline
        this.renderBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Render Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
            ],
        });

        this.sampler = this.device.createSampler({
            magFilter: 'nearest',
            minFilter: 'nearest',
        });

        this.renderPipeline = this.device.createRenderPipeline({
            label: 'Render Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.renderBindGroupLayout],
            }),
            vertex: {
                module: this.renderModule,
                entryPoint: 'vertexMain',
            },
            fragment: {
                module: this.renderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: this.canvasFormat }],
            },
        });

        this.createSizeDependentBindGroups();
    }

    // Every texture whose size follows the render resolution. Called once during
    // init and again on each resolution change, so the two paths cannot drift.
    createSizeDependentTextures() {
        for (const texture of [
            this.positionTexture, this.footprintTexture, this.outputTexture,
            this.depthTexture, this.noiseTexture, this.noiseTexture1,
            this.noiseTexture2, this.noiseTexture3,
        ]) {
            texture?.destroy();
        }

        const size = [this.resolutionX, this.resolutionY];
        // COPY_SRC is what makes the safetensors export readback possible.
        const attachment = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
        const storage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;

        // G-buffer: written by the raster pass, read by the noise pass.
        this.positionTexture = this.device.createTexture({
            label: 'Position Texture',
            size,
            format: 'rgba32float',
            usage: attachment,
        });
        this.footprintTexture = this.device.createTexture({
            label: 'Footprint Texture',
            size,
            format: 'rgba32float',
            usage: attachment,
        });

        // Raw N(0,1) noise, 16 channels across four rgba32float targets.
        [this.noiseTexture, this.noiseTexture1, this.noiseTexture2, this.noiseTexture3] =
            [0, 1, 2, 3].map((i) => this.device.createTexture({
                label: `Noise Texture ${i} (channels ${i * 4}-${i * 4 + 3})`,
                size,
                format: 'rgba32float',
                usage: storage,
            }));

        // Tonemapped result, blitted to the canvas.
        this.outputTexture = this.device.createTexture({
            label: 'Output Texture',
            size,
            format: 'rgba8unorm',
            usage: storage,
        });

        this.depthTexture = this.device.createTexture({
            label: 'Depth Texture',
            size,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    // The bind groups that reference those textures, so they have to be rebuilt
    // whenever the textures are. Requires the layouts and sampler to exist.
    createSizeDependentBindGroups() {
        this.computeBindGroup = this.device.createBindGroup({
            label: 'Compute Bind Group',
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: this.noiseTexture.createView() },       // channels 0-3
                { binding: 2, resource: this.positionTexture.createView() },
                { binding: 3, resource: this.footprintTexture.createView() },
                { binding: 5, resource: { buffer: this.cameraBuffer } },        // camera for world-space position
                { binding: 7, resource: this.noiseTexture1.createView() },      // channels 4-7
                { binding: 8, resource: this.noiseTexture2.createView() },      // channels 8-11
                { binding: 9, resource: this.noiseTexture3.createView() },      // channels 12-15
            ],
        });

        this.tonemapBindGroup = this.device.createBindGroup({
            label: 'Tonemap Bind Group',
            layout: this.tonemapBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: this.noiseTexture.createView() },
                { binding: 2, resource: this.outputTexture.createView() },
            ],
        });

        this.renderBindGroup = this.device.createBindGroup({
            label: 'Render Bind Group',
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: this.outputTexture.createView() },
                { binding: 1, resource: this.sampler },
            ],
        });
    }

    setResolution(width, height = null) {
        // Allow single value for backward compatibility (square resolution)
        const newWidth = width;
        const newHeight = height !== null ? height : width;

        if (newWidth !== this.resolutionX || newHeight !== this.resolutionY) {
            this.resolutionX = newWidth;
            this.resolutionY = newHeight;
            this.canvas.width = newWidth;
            this.canvas.height = newHeight;

            this.createSizeDependentTextures();
            this.createSizeDependentBindGroups();
        }
    }

    setSampleCount(sampleCount) {
        this.sampleCount = Math.min(sampleCount, 256);
    }

    setFieldResolution(resolution) {
        this.fieldResolution = resolution;
    }

    // Set histogram bin count (1-256)
    setHistogramBinCount(binCount) {
        this.histogramBinCount = Math.max(1, Math.min(256, binCount));
    }

    // Select the optimal workgroup size based on sample count
    // Matches sample count to nearest power-of-2 workgroup size for best efficiency
    // Capped at 256 for stability
    getOptimalWorkgroupSize(sampleCount) {
        // Available sizes: 32, 64, 128, 256 (capped)
        if (sampleCount <= 32) return 32;
        if (sampleCount <= 64) return 64;
        if (sampleCount <= 128) return 128;
        return 256;
    }

    // Get the compute pipeline for the current sample count. Variants other than the
    // default are compiled on first use, so fall back to the default until the
    // requested one is ready. The shader clamps maxSamples by WORKGROUP_SIZE, so the
    // default produces identical output, just with some threads idle.
    getComputePipelineForSampleCount() {
        const workgroupSize = this.getOptimalWorkgroupSize(this.sampleCount);
        const pipeline = this.computePipelines[workgroupSize];

        if (!pipeline) {
            this.ensureComputePipeline(workgroupSize);
            return this.computePipelines[DEFAULT_WORKGROUP_SIZE];
        }

        // Log when workgroup size changes
        if (this._lastWorkgroupSize !== workgroupSize) {
            console.log(`Workgroup size: ${workgroupSize} (sample count: ${this.sampleCount})`);
            this._lastWorkgroupSize = workgroupSize;
        }

        return pipeline;
    }

    // Compile one workgroup-size variant without blocking rendering. Repeated calls
    // share the in-flight promise rather than queueing a duplicate compile.
    ensureComputePipeline(size) {
        if (this.computePipelines[size]) {
            return Promise.resolve(this.computePipelines[size]);
        }
        if (this.failedComputePipelines.has(size)) {
            return Promise.resolve(null);
        }
        if (this.pendingComputePipelines[size]) {
            return this.pendingComputePipelines[size];
        }

        const pending = this.device.createComputePipelineAsync({
            label: `Compute Pipeline (workgroup=${size})`,
            layout: this.computePipelineLayout,
            compute: {
                module: this.computeModule,
                entryPoint: 'main',
                constants: {
                    WORKGROUP_SIZE: size,
                },
            },
        }).then((pipeline) => {
            this.computePipelines[size] = pipeline;
            delete this.pendingComputePipelines[size];
            return pipeline;
        }).catch((error) => {
            delete this.pendingComputePipelines[size];
            this.failedComputePipelines.add(size);
            console.error(`Failed to compile compute pipeline (workgroup=${size}):`, error);
            return null;
        });

        this.pendingComputePipelines[size] = pending;
        return pending;
    }

    // Compile the workgroup-size variants that init skipped. Call once a frame is on
    // screen, so these compiles never sit in front of the first frame.
    warmComputePipelines() {
        return Promise.all(this.workgroupSizes.map((size) => this.ensureComputePipeline(size)));
    }

    setTime(time) {
        this.time = time;
    }

    getVRAMEstimate() {
        const bytesPerPixel = { rgba32float: 16, rgba8unorm: 4 };
        const positionTex = this.resolutionX * this.resolutionY * bytesPerPixel.rgba32float;
        const footprintTex = this.resolutionX * this.resolutionY * bytesPerPixel.rgba32float;
        const outputTex = this.resolutionX * this.resolutionY * bytesPerPixel.rgba8unorm;
        const totalBytes = positionTex + footprintTex + outputTex + 64 + 128;
        return (totalBytes / (1024 * 1024)).toFixed(2);
    }

    render(skipScreenOutput = false) {
        // Compute effective time for this render
        // During export, use frame-based time instead of this.time
        let effectiveTime;
        if (this.isExporting) {
            effectiveTime = this.exportFrame / this.exportFps;
        } else {
            effectiveTime = this.time;
        }

        // Camera angles use the same effective time for interactive rendering and export.
        const cameraYaw = (this.meshCameraYaw + effectiveTime * this.meshCameraYawRotationSpeed) * Math.PI / 180.0;
        const cameraPitch = -this.meshCameraPitch * Math.PI / 180.0;

        // Get mesh normalization params (for skinned animation to match static mesh coordinate space)
        const normalization = this.customMesh?.normalization || { centerX: 0, centerY: 0, centerZ: 0, scale: 1 };

        // Must stay in the same order as the Params struct in the shaders.
        const paramsData = new Float32Array([
            this.resolutionX,
            this.resolutionY,
            this.fieldResolution,
            this.sampleCount,
            this.fieldSeed,  // Constant field seed - noise field doesn't change per frame
            this.outputMode,
            this.fieldType,
            this.rngType,
            this.estimatorMode,
            this.histogramBinCount,
            this.meshScale,
            normalization.centerX,
            normalization.centerY,
            normalization.centerZ,
            normalization.scale,
            this.centerBackgroundCube ? 1.0 : 0.0,
        ]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        // Orbit the camera around the mesh, which itself stays put.
        const meshCameraDistance = 3.0;
        const cosPitch = Math.cos(cameraPitch);
        const camPos = [
            meshCameraDistance * Math.sin(cameraYaw) * cosPitch,
            meshCameraDistance * Math.sin(cameraPitch),
            meshCameraDistance * Math.cos(cameraYaw) * cosPitch
        ];
        const sinYaw = Math.sin(cameraYaw);
        const cosYaw = Math.cos(cameraYaw);
        const sinPitch = Math.sin(cameraPitch);
        const camDir = [
            -sinYaw * cosPitch,
            -sinPitch,
            -cosYaw * cosPitch
        ];
        // Analytic basis remains valid at +/-90 degrees pitch.
        const camRight = [-cosYaw, 0.0, sinYaw];
        const camUp = [
            sinYaw * sinPitch,
            -cosPitch,
            cosYaw * sinPitch
        ];

        const fov = this.meshFov * Math.PI / 180.0;

        const cameraData = new Float32Array([
            camPos[0], camPos[1], camPos[2], 0.0,
            camDir[0], camDir[1], camDir[2], 0.0,
            camRight[0], camRight[1], camRight[2], 0.0,
            camUp[0], camUp[1], camUp[2], fov
        ]);
        this.device.queue.writeBuffer(this.cameraBuffer, 0, cameraData);

        const commandEncoder = this.device.createCommandEncoder();

        // Pass 1: Rasterizer
        // Hardware mesh rasterization of the loaded mesh plus the background cube.
        {
            // Custom Mesh Rasterization: User-loaded OBJ mesh + background cube
            const passDescriptor = {
                colorAttachments: [
                    {
                        view: this.positionTexture.createView(),
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                    {
                        view: this.footprintTexture.createView(),
                        clearValue: { r: 0, g: -1, b: 0, a: 0 },  // objectId = -1 for background
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
                depthStencilAttachment: {
                    view: this.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            };
            if (this.hasTimestampQuery) {
                passDescriptor.timestampWrites = {
                    querySet: this.timestampQuerySet,
                    beginningOfPassWriteIndex: 0,
                };
            }
            const pass = commandEncoder.beginRenderPass(passDescriptor);

            // Draw background cube first (renders at far depth, behind everything)
            pass.setPipeline(this.cubePipeline);
            pass.setBindGroup(0, this.meshRasterizerBindGroup);
            pass.setVertexBuffer(0, this.cubeVertexBuffer);
            pass.setIndexBuffer(this.cubeIndexBuffer, 'uint32');
            pass.drawIndexed(this.cubeMesh.indexCount, 1);

            // Draw custom mesh (single instance)
            // Use skinned pipeline if mesh has skinning data and animation is active
            if (this.meshHasSkinning && this.skinnedMeshPipeline && this.skinnedMeshRasterVertexBuffer && this.currentAnimationIndex >= 0) {
                pass.setPipeline(this.skinnedMeshPipeline);
                pass.setBindGroup(0, this.meshRasterizerBindGroup);
                pass.setVertexBuffer(0, this.skinnedMeshRasterVertexBuffer);
                pass.setIndexBuffer(this.customMeshRasterIndexBuffer, 'uint32');
                pass.drawIndexed(this.customMeshRasterIndexCount, 1);
            } else {
                pass.setPipeline(this.meshRasterizerPipeline);
                pass.setBindGroup(0, this.meshRasterizerBindGroup);
                pass.setVertexBuffer(0, this.customMeshRasterVertexBuffer);
                pass.setIndexBuffer(this.customMeshRasterIndexBuffer, 'uint32');
                pass.drawIndexed(this.customMeshRasterIndexCount, 1);
            }

            pass.end();
        }

        // Pass 2: Noise/Histogram (outputs to noiseTexture)
        // Uses adaptive workgroup size based on sample count
        {
            const pass = commandEncoder.beginComputePass();
            const pipeline = this.getComputePipelineForSampleCount();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, this.computeBindGroup);
            // One workgroup per pixel, threads per workgroup matches sample count
            pass.dispatchWorkgroups(this.resolutionX, this.resolutionY, 1);
            pass.end();
        }

        // Pass 3: Tonemap (converts noiseTexture float -> outputTexture rgba8unorm)
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.tonemapPipeline);
            pass.setBindGroup(0, this.tonemapBindGroup);
            pass.dispatchWorkgroups(
                Math.ceil(this.resolutionX / 16),
                Math.ceil(this.resolutionY / 16)
            );
            pass.end();
        }

        // Pass 4: Render to screen (skip during export to prevent flashing)
        if (!skipScreenOutput) {
            const passDescriptor = {
                colorAttachments: [{
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            };
            if (this.hasTimestampQuery) {
                passDescriptor.timestampWrites = {
                    querySet: this.timestampQuerySet,
                    endOfPassWriteIndex: 1,
                };
            }
            const pass = commandEncoder.beginRenderPass(passDescriptor);
            pass.setPipeline(this.renderPipeline);
            pass.setBindGroup(0, this.renderBindGroup);
            pass.draw(3);
            pass.end();
        }

        // Resolve timestamps if supported - only if target buffer is not mapped
        if (this.hasTimestampQuery && !this.timestampBufferMapped[this.timestampBufferIndex]) {
            const readBuffer = this.timestampReadBuffers[this.timestampBufferIndex];
            commandEncoder.resolveQuerySet(this.timestampQuerySet, 0, 2, this.timestampBuffer, 0);
            commandEncoder.copyBufferToBuffer(this.timestampBuffer, 0, readBuffer, 0, 16);
        }

        this.device.queue.submit([commandEncoder.finish()]);

        // Read back timestamps asynchronously using double-buffering
        if (this.hasTimestampQuery && !this.timestampBufferMapped[this.timestampBufferIndex]) {
            const readIndex = this.timestampBufferIndex;
            const readBuffer = this.timestampReadBuffers[readIndex];

            this.timestampBufferMapped[readIndex] = true;
            readBuffer.mapAsync(GPUMapMode.READ).then(() => {
                const data = new BigInt64Array(readBuffer.getMappedRange());
                const startNs = data[0];
                const endNs = data[1];
                this.lastGpuTime = Number(endNs - startNs) / 1_000_000; // Convert ns to ms
                readBuffer.unmap();
                this.timestampBufferMapped[readIndex] = false;
            }).catch(() => {
                this.timestampBufferMapped[readIndex] = false;
            });

            // Swap buffers
            this.timestampBufferIndex = 1 - this.timestampBufferIndex;
        }
    }

    getLastGpuTime() {
        return this.lastGpuTime || 0;
    }

    // =============================================================================
    // Safetensors Export
    // =============================================================================

    async readTextureToArray(texture, format = 'rgba32float') {
        const bytesPerPixel = format === 'rgba32float' ? 16 : 4;  // 4 floats * 4 bytes or 4 bytes
        const bufferSize = this.resolutionX * this.resolutionY * bytesPerPixel;

        // Create a buffer to copy the texture to
        const readBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            { texture: texture },
            { buffer: readBuffer, bytesPerRow: this.resolutionX * bytesPerPixel, rowsPerImage: this.resolutionY },
            { width: this.resolutionX, height: this.resolutionY, depthOrArrayLayers: 1 }
        );
        this.device.queue.submit([commandEncoder.finish()]);

        // Wait for the copy to complete and map the buffer
        await readBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBuffer.getMappedRange();
        const data = new Float32Array(arrayBuffer.slice(0));
        readBuffer.unmap();
        readBuffer.destroy();

        return data;
    }

    async exportFrameData() {
        const H = this.resolutionY;
        const W = this.resolutionX;
        const originalOutputMode = this.outputMode;

        // Helper to render a specific mode and read raw values from noiseTexture
        // showOnScreen parameter controls whether to display the frame (for debugging export)
        const renderModeAndRead = async (mode, showOnScreen = false) => {
            this.outputMode = mode;
            this.render(!showOnScreen);  // skipScreenOutput = !showOnScreen
            await this.device.queue.onSubmittedWorkDone();
            return await this.readTextureToArray(this.noiseTexture);
        };

        // Mode 0: Raw noise N(0,1) - read all 4 textures for 16 channels
        // Show this frame on screen so user can see export progress
        const noiseData0 = await renderModeAndRead(0, true);
        // After render, also read the other 3 noise textures (they're already populated)
        const noiseData1 = await this.readTextureToArray(this.noiseTexture1);
        const noiseData2 = await this.readTextureToArray(this.noiseTexture2);
        const noiseData3 = await this.readTextureToArray(this.noiseTexture3);

        // Mode 4: Raw depth (world units)
        const depthData = await renderModeAndRead(4);

        // Mode 5: Raw object ID
        const objectIdData = await renderModeAndRead(5);

        // Mode 6: Raw world-space position
        const worldPosData = await renderModeAndRead(6);

        // Mode 7: Raw world-space normals [-1, 1]
        const normalRawData = await renderModeAndRead(7);

        // Restore original output mode
        this.outputMode = originalOutputMode;

        // Create value tensor: [H, W, 16]
        // Read raw N(0,1) values from all 4 noise textures (4 channels each = 16 total)
        const valueData = new Float32Array(H * W * 16);
        for (let y = 0; y < H; y++) {
            const srcY = H - 1 - y;  // Flip Y
            for (let x = 0; x < W; x++) {
                const srcIdx = (srcY * W + x) * 4;
                const dstIdx = (y * W + x) * 16;
                // Channels 0-3 from noiseTexture0
                valueData[dstIdx + 0] = noiseData0[srcIdx + 0];
                valueData[dstIdx + 1] = noiseData0[srcIdx + 1];
                valueData[dstIdx + 2] = noiseData0[srcIdx + 2];
                valueData[dstIdx + 3] = noiseData0[srcIdx + 3];
                // Channels 4-7 from noiseTexture1
                valueData[dstIdx + 4] = noiseData1[srcIdx + 0];
                valueData[dstIdx + 5] = noiseData1[srcIdx + 1];
                valueData[dstIdx + 6] = noiseData1[srcIdx + 2];
                valueData[dstIdx + 7] = noiseData1[srcIdx + 3];
                // Channels 8-11 from noiseTexture2
                valueData[dstIdx + 8] = noiseData2[srcIdx + 0];
                valueData[dstIdx + 9] = noiseData2[srcIdx + 1];
                valueData[dstIdx + 10] = noiseData2[srcIdx + 2];
                valueData[dstIdx + 11] = noiseData2[srcIdx + 3];
                // Channels 12-15 from noiseTexture3
                valueData[dstIdx + 12] = noiseData3[srcIdx + 0];
                valueData[dstIdx + 13] = noiseData3[srcIdx + 1];
                valueData[dstIdx + 14] = noiseData3[srcIdx + 2];
                valueData[dstIdx + 15] = noiseData3[srcIdx + 3];
            }
        }

        // Extract depth data: [H, W] - raw depth in world units (flipped Y)
        const depthOut = new Float32Array(H * W);
        for (let y = 0; y < H; y++) {
            const srcY = H - 1 - y;  // Flip Y
            for (let x = 0; x < W; x++) {
                const srcIdx = (srcY * W + x) * 4;
                const dstIdx = y * W + x;
                depthOut[dstIdx] = depthData[srcIdx + 0];
            }
        }

        // Extract id data: [H, W] - raw object ID (flipped Y)
        // Remap to positive integers:
        //   -1 (miss/background) -> 0
        //   -2 (floor) -> 1
        //   0, 1, 2, ... (balls) -> 2, 3, 4, ...
        const idData = new Float32Array(H * W);
        for (let y = 0; y < H; y++) {
            const srcY = H - 1 - y;  // Flip Y
            for (let x = 0; x < W; x++) {
                const srcIdx = (srcY * W + x) * 4;
                const dstIdx = y * W + x;
                const rawId = objectIdData[srcIdx + 0];
                if (rawId < -1.5) {
                    // Floor (rawId == -2)
                    idData[dstIdx] = 1;
                } else if (rawId < -0.5) {
                    // Background/miss (rawId == -1)
                    idData[dstIdx] = 0;
                } else {
                    // Ball (rawId == 0, 1, 2, ...) -> 2, 3, 4, ...
                    idData[dstIdx] = Math.round(rawId) + 2;
                }
            }
        }

        // Extract world-space position: [H, W, 3] - raw from shader (flipped Y)
        const posData = new Float32Array(H * W * 3);
        for (let y = 0; y < H; y++) {
            const srcY = H - 1 - y;  // Flip Y
            for (let x = 0; x < W; x++) {
                const srcIdx = (srcY * W + x) * 4;
                const dstIdx = (y * W + x) * 3;
                posData[dstIdx + 0] = worldPosData[srcIdx + 0];
                posData[dstIdx + 1] = worldPosData[srcIdx + 1];
                posData[dstIdx + 2] = worldPosData[srcIdx + 2];
            }
        }

        // Extract world-space normals: [H, W, 3] - raw from shader [-1, 1] (flipped Y)
        const normalData = new Float32Array(H * W * 3);
        for (let y = 0; y < H; y++) {
            const srcY = H - 1 - y;  // Flip Y
            for (let x = 0; x < W; x++) {
                const srcIdx = (srcY * W + x) * 4;
                const dstIdx = (y * W + x) * 3;
                normalData[dstIdx + 0] = normalRawData[srcIdx + 0];
                normalData[dstIdx + 1] = normalRawData[srcIdx + 1];
                normalData[dstIdx + 2] = normalRawData[srcIdx + 2];
            }
        }

        return {
            value: valueData,      // [H, W, 16] - raw N(0,1) noise
            depth: depthOut,       // [H, W] - raw depth in world units
            id: idData,            // [H, W] - object ID (remapped to positive)
            position: posData,     // [H, W, 3] - object-space position (matches Python)
            normal: normalData,    // [H, W, 3] - world-space normals [-1, 1] (matches Python)
        };
    }

    async exportToSafetensors(numFrames = 100, progressCallback = null, fps = 30) {
        return this.exportToSafetensorsInternal(numFrames, progressCallback, false, fps);
    }

    async exportNoiseToSafetensors(numFrames = 100, progressCallback = null, fps = 30, dirHandle = null) {
        return this.exportToSafetensorsInternal(numFrames, progressCallback, true, fps, dirHandle);
    }

    async exportToSafetensorsInternal(numFrames = 100, progressCallback = null, noiseOnly = false, fps = 30, dirHandle = null) {
        const H = this.resolutionY;
        const W = this.resolutionX;

        // Calculate sizes per frame
        const valueFrameSize = H * W * 16 * 4;    // [H, W, 16] float32
        const depthFrameSize = H * W * 4;         // [H, W] float32
        const idFrameSize = H * W * 4;            // [H, W] float32
        const posFrameSize = H * W * 3 * 4;       // [H, W, 3] float32
        const normalFrameSize = H * W * 3 * 4;    // [H, W, 3] float32

        // Total sizes for all frames (concatenated tensors)
        const valueTotalSize = valueFrameSize * numFrames;    // [numFrames, H, W, 16]
        const depthTotalSize = depthFrameSize * numFrames;    // [numFrames, H, W]
        const idTotalSize = idFrameSize * numFrames;          // [numFrames, H, W]
        const posTotalSize = posFrameSize * numFrames;        // [numFrames, H, W, 3]
        const normalTotalSize = normalFrameSize * numFrames;  // [numFrames, H, W, 3]

        const totalDataSize = noiseOnly ? valueTotalSize : (valueTotalSize + depthTotalSize + idTotalSize + posTotalSize + normalTotalSize);

        const exportType = noiseOnly ? 'noise only' : 'all attributes';
        console.log(`Export (${exportType}): ${numFrames} frames at ${H}x${W}, ~${(totalDataSize / (1024 * 1024)).toFixed(1)} MB data`);

        // Store original output mode and switch to noise mode for export
        const originalOutputMode = this.outputMode;
        this.outputMode = 0;  // Noise mode for proper raw values

        try {
            // Build header with concatenated tensors [numFrames, H, W, C]
            const header = { '__metadata__': {} };
            let offset = 0;

            header['value'] = { dtype: 'F32', shape: [numFrames, H, W, 16], data_offsets: [offset, offset + valueTotalSize] };
            offset += valueTotalSize;

            if (!noiseOnly) {
                header['depth'] = { dtype: 'F32', shape: [numFrames, H, W], data_offsets: [offset, offset + depthTotalSize] };
                offset += depthTotalSize;
                header['id'] = { dtype: 'F32', shape: [numFrames, H, W], data_offsets: [offset, offset + idTotalSize] };
                offset += idTotalSize;
                header['position'] = { dtype: 'F32', shape: [numFrames, H, W, 3], data_offsets: [offset, offset + posTotalSize] };
                offset += posTotalSize;
                header['normal'] = { dtype: 'F32', shape: [numFrames, H, W, 3], data_offsets: [offset, offset + normalTotalSize] };
                offset += normalTotalSize;
            }

            const headerJson = JSON.stringify(header);
            const headerBytes = new TextEncoder().encode(headerJson);
            const headerPadding = (8 - (headerBytes.length % 8)) % 8;
            const paddedHeaderLength = headerBytes.length + headerPadding;

            // Try File System Access API for large files (streaming to disk)
            // Skip if dirHandle is provided (batch mode - we'll write to directory instead)
            if (!dirHandle && window.showSaveFilePicker && totalDataSize > 500 * 1024 * 1024) {
                console.log('Using File System Access API for streaming write...');
                return await this.exportToSafetensorsStreaming(
                    numFrames, H, W,
                    valueFrameSize, depthFrameSize, idFrameSize, posFrameSize, normalFrameSize,
                    headerBytes, headerPadding, paddedHeaderLength,
                    progressCallback, noiseOnly, fps
                );
            }

            // For smaller files, use in-memory approach
            const totalSize = 8 + paddedHeaderLength + totalDataSize;
            console.log(`Allocating ${(totalSize / (1024 * 1024)).toFixed(1)} MB buffer...`);

            let buffer;
            try {
                buffer = new ArrayBuffer(totalSize);
            } catch (e) {
                // If allocation fails, try streaming API
                if (window.showSaveFilePicker) {
                    console.log('Memory allocation failed, falling back to streaming...');
                    return await this.exportToSafetensorsStreaming(
                        numFrames, H, W,
                        valueFrameSize, depthFrameSize, idFrameSize, posFrameSize, normalFrameSize,
                        headerBytes, headerPadding, paddedHeaderLength,
                        progressCallback, noiseOnly, fps
                    );
                }
                throw new Error(`Failed to allocate ${(totalSize / (1024 * 1024)).toFixed(1)} MB. Your browser may not support large allocations.`);
            }

            const view = new DataView(buffer);
            const u8 = new Uint8Array(buffer);

            // Write header size
            view.setBigUint64(0, BigInt(paddedHeaderLength), true);

            // Write header JSON
            u8.set(headerBytes, 8);
            for (let i = 0; i < headerPadding; i++) {
                u8[8 + headerBytes.length + i] = 0x20;
            }

            // Collect all frame data first
            // Enable export mode for frame-based rendering
            this.isExporting = true;
            this.exportFps = fps;

            // Store original state to restore after export
            const originalTime = this.time;

            // Check if skeletal animation is available
            const hasSkeletalAnim = this.currentAnimationIndex >= 0 &&
                                    this.meshAnimations &&
                                    this.meshAnimations.length > 0 &&
                                    this.meshSkin;

            if (hasSkeletalAnim) {
                const animation = this.meshAnimations[this.currentAnimationIndex];
                console.log(`Export: Skeletal animation "${animation.name}" enabled, duration=${animation.duration.toFixed(3)}s, joints=${this.meshSkin.jointCount}, fps=${fps}`);
            } else {
                console.log(`Export: No skeletal animation (animIdx=${this.currentAnimationIndex}, anims=${this.meshAnimations?.length || 0}, skin=${!!this.meshSkin})`);
            }

            const allFrameData = [];
            for (let frame = 0; frame < numFrames; frame++) {
                // Set the current export frame - render() will use this directly
                this.exportFrame = frame;

                // Calculate the effective time for this frame
                const effectiveTime = frame / fps;

                // Also set this.time for consistency (though render() uses exportFrame/exportFps during export)
                this.time = effectiveTime;

                // Update the Time slider UI to show current export frame
                const timeSlider = document.getElementById('timeSlider');
                const timeValue = document.getElementById('timeValue');
                if (timeSlider) timeSlider.value = effectiveTime;
                if (timeValue) timeValue.textContent = effectiveTime.toFixed(2);

                // Update skeletal animation using the same effectiveTime as mesh rotation
                // This matches how the viewer works: both use the same time source
                if (hasSkeletalAnim) {
                    this.updateAnimationFromTime(effectiveTime);

                    // Debug: log first few frames
                    if (frame < 3) {
                        const animation = this.meshAnimations[this.currentAnimationIndex];
                        const scaledTime = effectiveTime * this.animationSpeed;
                        const animTime = scaledTime % animation.duration;
                        console.log(`  Frame ${frame}: effectiveTime=${effectiveTime.toFixed(3)}, scaledTime=${scaledTime.toFixed(3)}, animTime=${animTime.toFixed(3)}, animSpeed=${this.animationSpeed}`);
                    }
                }

                const frameData = await this.exportFrameData();
                allFrameData.push(frameData);

                if (progressCallback) {
                    progressCallback(frame + 1, numFrames);
                }
            }

            // Disable export mode and restore original state
            this.isExporting = false;
            this.setTime(originalTime);

            // Write concatenated tensors in order: value, depth, id, position, normal
            let dataOffset = 8 + paddedHeaderLength;

            // Write all value frames
            for (let frame = 0; frame < numFrames; frame++) {
                u8.set(new Uint8Array(allFrameData[frame].value.buffer), dataOffset);
                dataOffset += valueFrameSize;
            }

            if (!noiseOnly) {
                // Write all depth frames
                for (let frame = 0; frame < numFrames; frame++) {
                    u8.set(new Uint8Array(allFrameData[frame].depth.buffer), dataOffset);
                    dataOffset += depthFrameSize;
                }
                // Write all id frames
                for (let frame = 0; frame < numFrames; frame++) {
                    u8.set(new Uint8Array(allFrameData[frame].id.buffer), dataOffset);
                    dataOffset += idFrameSize;
                }
                // Write all position frames
                for (let frame = 0; frame < numFrames; frame++) {
                    u8.set(new Uint8Array(allFrameData[frame].position.buffer), dataOffset);
                    dataOffset += posFrameSize;
                }
                // Write all normal frames
                for (let frame = 0; frame < numFrames; frame++) {
                    u8.set(new Uint8Array(allFrameData[frame].normal.buffer), dataOffset);
                    dataOffset += normalFrameSize;
                }
            }

            return buffer;
        } finally {
            this.outputMode = originalOutputMode;
        }
    }

    // Streaming export using File System Access API - handles unlimited file sizes
    async exportToSafetensorsStreaming(
        numFrames, H, W,
        valueFrameSize, depthFrameSize, idFrameSize, posFrameSize, normalFrameSize,
        headerBytes, headerPadding, paddedHeaderLength,
        progressCallback, noiseOnly = false, fps = 30
    ) {
        // Show save file picker
        const fileHandle = await window.showSaveFilePicker({
            suggestedName: `tabula_rasa_${W}x${H}_${numFrames}frames.safetensors`,
            types: [{
                description: 'Safetensors file',
                accept: { 'application/octet-stream': ['.safetensors'] },
            }],
        });

        const writable = await fileHandle.createWritable();

        try {
            // Write header size (8 bytes)
            const headerSizeBuffer = new ArrayBuffer(8);
            new DataView(headerSizeBuffer).setBigUint64(0, BigInt(paddedHeaderLength), true);
            await writable.write(headerSizeBuffer);

            // Write header JSON + padding
            await writable.write(headerBytes);
            if (headerPadding > 0) {
                await writable.write(new Uint8Array(headerPadding).fill(0x20));
            }

            // Collect all frame data first (needed for concatenated tensor format)
            // Enable export mode for frame-based rendering
            this.isExporting = true;
            this.exportFps = fps;

            // Store original state to restore after export
            const originalTime = this.time;

            // Check if skeletal animation is available
            const hasSkeletalAnim = this.currentAnimationIndex >= 0 &&
                                    this.meshAnimations &&
                                    this.meshAnimations.length > 0 &&
                                    this.meshSkin;

            if (hasSkeletalAnim) {
                const animation = this.meshAnimations[this.currentAnimationIndex];
                console.log(`Export (streaming): Skeletal animation "${animation.name}" enabled, duration=${animation.duration.toFixed(3)}s, joints=${this.meshSkin.jointCount}, fps=${fps}`);
            } else {
                console.log(`Export (streaming): No skeletal animation (animIdx=${this.currentAnimationIndex}, anims=${this.meshAnimations?.length || 0}, skin=${!!this.meshSkin})`);
            }

            const allFrameData = [];
            for (let frame = 0; frame < numFrames; frame++) {
                // Set the current export frame - render() will use this directly
                this.exportFrame = frame;

                // Calculate the effective time for this frame
                const effectiveTime = frame / fps;

                // Also set this.time for consistency (though render() uses exportFrame/exportFps during export)
                this.time = effectiveTime;

                // Update the Time slider UI to show current export frame
                const timeSlider = document.getElementById('timeSlider');
                const timeValue = document.getElementById('timeValue');
                if (timeSlider) timeSlider.value = effectiveTime;
                if (timeValue) timeValue.textContent = effectiveTime.toFixed(2);

                // Update skeletal animation using the same effectiveTime as mesh rotation
                // This matches how the viewer works: both use the same time source
                if (hasSkeletalAnim) {
                    this.updateAnimationFromTime(effectiveTime);

                    // Debug: log first few frames
                    if (frame < 3) {
                        const animation = this.meshAnimations[this.currentAnimationIndex];
                        const scaledTime = effectiveTime * this.animationSpeed;
                        const animTime = scaledTime % animation.duration;
                        console.log(`  Frame ${frame}: effectiveTime=${effectiveTime.toFixed(3)}, scaledTime=${scaledTime.toFixed(3)}, animTime=${animTime.toFixed(3)}, animSpeed=${this.animationSpeed}`);
                    }
                }

                const frameData = await this.exportFrameData();
                allFrameData.push(frameData);

                if (progressCallback) {
                    progressCallback(frame + 1, numFrames);
                }
            }

            // Disable export mode and restore original state
            this.isExporting = false;
            this.setTime(originalTime);

            // Write concatenated tensors in order: value, depth, id, position, normal
            // All frames of each tensor type are written contiguously

            // Write all value frames
            for (let frame = 0; frame < numFrames; frame++) {
                await writable.write(new Uint8Array(allFrameData[frame].value.buffer));
            }

            if (!noiseOnly) {
                // Write all depth frames
                for (let frame = 0; frame < numFrames; frame++) {
                    await writable.write(new Uint8Array(allFrameData[frame].depth.buffer));
                }
                // Write all id frames
                for (let frame = 0; frame < numFrames; frame++) {
                    await writable.write(new Uint8Array(allFrameData[frame].id.buffer));
                }
                // Write all position frames
                for (let frame = 0; frame < numFrames; frame++) {
                    await writable.write(new Uint8Array(allFrameData[frame].position.buffer));
                }
                // Write all normal frames
                for (let frame = 0; frame < numFrames; frame++) {
                    await writable.write(new Uint8Array(allFrameData[frame].normal.buffer));
                }
            }

            await writable.close();
            return 'streamed';  // Special return value indicating file was streamed
        } catch (e) {
            await writable.abort();
            throw e;
        }
    }
}

// =============================================================================
// Settings Persistence
// =============================================================================

const SETTINGS_KEY = 'tabula_rasa_settings';
// Bumped when the meaning of a stored value changes, so stale blobs can be migrated.
const SETTINGS_VERSION = 3;

function saveSettings(renderer) {
    const settings = {
        version: SETTINGS_VERSION,
        resolutionX: renderer.resolutionX,
        resolutionY: renderer.resolutionY,
        fieldResolution: renderer.fieldResolution,
        sampleCount: renderer.sampleCount,
        time: renderer.time,
        outputMode: renderer.outputMode,
        fieldType: renderer.fieldType,
        rngType: renderer.rngType,
        centerBackgroundCube: renderer.centerBackgroundCube,
        showHistogram: document.getElementById('showHistogram')?.checked ?? true,
    };
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (e) {
        console.warn('Failed to save settings:', e);
    }
}

function loadSettings() {
    try {
        const stored = localStorage.getItem(SETTINGS_KEY);
        if (stored) {
            const settings = JSON.parse(stored);
            const storedVersion = settings.version ?? 0;
            if (storedVersion < 2) {
                // Field type values were renumbered when the Soft Adaptive field was
                // removed. Older values can still be in range but select another field.
                delete settings.fieldType;
            }
            if (settings.fieldResolution === undefined && settings.fieldResolutionX !== undefined) {
                settings.fieldResolution = settings.fieldResolutionX;
            }
            delete settings.fieldResolutionX;
            delete settings.fieldResolutionY;
            return settings;
        }
    } catch (e) {
        console.warn('Failed to load settings:', e);
    }
    return null;
}

function applyRendererSettings(renderer, settings) {
    // Map of settings to their setter methods
    const setterMap = {
        sampleCount: 'setSampleCount',
        time: 'setTime',
        outputMode: 'setOutputMode',
        fieldType: 'setFieldType',
    };

    // Apply settings that have setter methods
    for (const [key, setter] of Object.entries(setterMap)) {
        if (settings[key] !== undefined) {
            renderer[setter](settings[key]);
        }
    }

    // Apply direct property assignments
    const directProps = ['rngType', 'centerBackgroundCube'];
    for (const prop of directProps) {
        if (settings[prop] !== undefined) {
            renderer[prop] = settings[prop];
        }
    }

    // Apply resolution settings (require both dimensions)
    if (settings.resolutionX && settings.resolutionY) {
        renderer.setResolution(settings.resolutionX, settings.resolutionY);
    }
    if (settings.fieldResolution) {
        renderer.setFieldResolution(settings.fieldResolution);
    }
}

function applySettings(renderer, settings, updateUI) {
    if (!settings) return;

    applyRendererSettings(renderer, settings);
    updateUI(settings);
}

// =============================================================================
// Main Application
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    const noiseCanvas = document.getElementById('noiseCanvas');
    const histogramCanvas = document.getElementById('histogramCanvas');
    const errorContainer = document.getElementById('errorContainer');

    if (!navigator.gpu) {
        errorContainer.innerHTML = `
            <div class="error">
                <h3>WebGPU Not Supported</h3>
                <p>Your browser does not support WebGPU.</p>
            </div>
        `;
        return;
    }

    let renderer;
    let histogramVis;

    try {
        renderer = new WebGPUNoiseRenderer(noiseCanvas);
        await renderer.init();
        await renderer.loadMesh(
            DEFAULT_MESH_OBJ,
            'default-cube.obj',
            { geometryScale: DEFAULT_MESH_GEOMETRY_SCALE },
        );
        histogramVis = new HistogramVisualizer(histogramCanvas);

        // Expose renderer globally for inline event handlers
        window.mcRenderer = renderer;
    } catch (error) {
        console.error('WebGPU initialization failed:', error);
        errorContainer.innerHTML = `
            <div class="error">
                <h3>WebGPU Initialization Failed</h3>
                <p>${error.message}</p>
            </div>
        `;
        return;
    }

    // Get UI elements
    const timeSlider = document.getElementById('timeSlider');
    const timeValue = document.getElementById('timeValue');
    const resWidthSlider = document.getElementById('resWidthSlider');
    const resWidthValue = document.getElementById('resWidthValue');
    const resHeightSlider = document.getElementById('resHeightSlider');
    const resHeightValue = document.getElementById('resHeightValue');
    const sampleSlider = document.getElementById('sampleSlider');
    const sampleValue = document.getElementById('sampleValue');
    const fieldResolutionSlider = document.getElementById('fieldResolutionSlider');
    const fieldResolutionValue = document.getElementById('fieldResolutionValue');
    const statsElement = document.getElementById('stats');

    function createEditableInput(currentValue, options, accessibleName) {
        const { min, max, step, isFloat } = options;
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'value-input';
        input.value = isFloat ? parseFloat(currentValue) : parseInt(currentValue);
        input.min = min;
        input.max = max;
        input.step = step;
        input.setAttribute('aria-label', accessibleName.replace(/^Edit\s+/i, ''));
        return input;
    }

    function parseAndClampValue(inputValue, fallbackValue, options) {
        const { min, max, isFloat } = options;
        const parsedValue = isFloat ? parseFloat(inputValue) : parseInt(inputValue);
        const fallback = isFloat ? parseFloat(fallbackValue) : parseInt(fallbackValue);
        const value = Number.isNaN(parsedValue) ? fallback : parsedValue;
        return Math.max(min, Math.min(max, value));
    }

    function formatEditableValue(value, decimalPlaces) {
        return decimalPlaces > 0 ? value.toFixed(decimalPlaces) : String(Math.trunc(value));
    }

    function makeEditable(valueSpan, slider, onChange, options = {}) {
        const opts = {
            min: 1,
            max: 4096,
            step: 1,
            isFloat: false,
            decimalPlaces: 0,
            onStart: null,
            ...options,
        };
        const accessibleName = (valueSpan.getAttribute('aria-label') || 'Edit value')
            .replace(/,\s*current value.*$/i, '');
        const updateAccessibleName = () => {
            valueSpan.setAttribute(
                'aria-label',
                `${accessibleName}, current value ${valueSpan.textContent.trim()}`,
            );
        };

        valueSpan.style.cursor = 'pointer';
        valueSpan.title = 'Click to edit';
        updateAccessibleName();
        new MutationObserver(updateAccessibleName).observe(valueSpan, {
            childList: true,
            characterData: true,
            subtree: true,
        });

        valueSpan.addEventListener('click', (e) => {
            e.stopPropagation();
            opts.onStart?.();

            const input = createEditableInput(valueSpan.textContent, opts, accessibleName);
            const parent = valueSpan.parentNode;
            parent.replaceChild(input, valueSpan);
            input.focus();
            input.select();

            let settled = false;
            const finishEdit = ({ commit, restoreFocus }) => {
                if (settled) return;
                settled = true;

                if (commit) {
                    const value = parseAndClampValue(input.value, valueSpan.textContent, opts);
                    valueSpan.textContent = formatEditableValue(value, opts.decimalPlaces);
                    slider.value = value;
                    updateAccessibleName();
                    onChange(value);
                    scheduleSaveSettings();
                }

                if (input.parentNode === parent) {
                    parent.replaceChild(valueSpan, input);
                }
                if (restoreFocus) {
                    queueMicrotask(() => valueSpan.focus({ preventScroll: true }));
                }
            };

            input.addEventListener('blur', () => finishEdit({ commit: true, restoreFocus: false }));
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    finishEdit({ commit: true, restoreFocus: true });
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    finishEdit({ commit: false, restoreFocus: true });
                }
            });
        });
    }

    makeEditable(timeValue, timeSlider, (value) => {
        renderer.setTime(value);
        renderer.updateAnimationFromTime(value);
        update();
    }, {
        min: 0,
        max: 5,
        step: 0.01,
        isFloat: true,
        decimalPlaces: 2,
        onStart: () => {
            if (animating) setAnimating(false);
        },
    });

    makeEditable(resWidthValue, resWidthSlider, (value) => {
        renderer.setResolution(value, renderer.resolutionY);
        update();
    }, { min: 64, max: 1024, step: 1 });

    makeEditable(resHeightValue, resHeightSlider, (value) => {
        renderer.setResolution(renderer.resolutionX, value);
        update();
    }, { min: 64, max: 1024, step: 1 });

    makeEditable(fieldResolutionValue, fieldResolutionSlider, (value) => {
        renderer.setFieldResolution(value);
        update();
    }, { min: 16, max: 2048, step: 1 });

    makeEditable(sampleValue, sampleSlider, (value) => {
        renderer.setSampleCount(value);
        update();
    }, { min: 1, max: 256, step: 1 });

    // Output mode and field type
    const outputModeSelect = document.getElementById('outputModeSelect');
    const fieldTypeSelect = document.getElementById('fieldTypeSelect');

    // Histogram toggle
    const showHistogramCheckbox = document.getElementById('showHistogram');
    const binCountSlider = document.getElementById('binCountSlider');
    const binCountValue = document.getElementById('binCountValue');

    makeEditable(binCountValue, binCountSlider, (value) => {
        renderer.setHistogramBinCount(value);
        update();
    }, { min: 1, max: 256, step: 1 });

    showHistogramCheckbox.addEventListener('change', (e) => {
        const histogramEnabled = e.target.checked;
        console.log('Histogram toggle:', histogramEnabled);
        // Hide the entire histogram wrapper, not just dim it
        histogramCanvas.parentElement.style.display = histogramEnabled ? 'block' : 'none';
        update();
    });

    if (binCountSlider) {
        binCountSlider.addEventListener('input', (e) => {
            const bins = parseInt(e.target.value);
            binCountValue.textContent = bins;
            renderer.setHistogramBinCount(bins);
            update();
        });
    }

    // Performance tracking
    let frameCount = 0;
    let lastFpsUpdate = performance.now();
    let fps = 0;

    function update() {
        renderer.render();

        // Only render histogram if checkbox is checked (GPU→CPU readback is slow)
        const histogramEnabled = showHistogramCheckbox.checked;
        histogramCanvas.parentElement.style.display = histogramEnabled ? 'block' : 'none';
        if (histogramEnabled) {
            histogramVis.setOutputMode(renderer.outputMode);
            histogramVis.render(noiseCanvas);
        }

        frameCount++;
        const now = performance.now();
        if (now - lastFpsUpdate >= 1000) {
            fps = frameCount;
            frameCount = 0;
            lastFpsUpdate = now;
            const vram = renderer.getVRAMEstimate();
            const avgText = renderer.hasTimestampQuery ? `${renderer.getLastGpuTime().toFixed(1)}ms` : '(timestamps not supported)';
            statsElement.textContent = `FPS: ${fps} | Avg: ${avgText} | VRAM: ${vram}MB`;
        }
    }

    // Event handlers
    timeSlider.addEventListener('input', (e) => {
        // Stop auto-play when manually dragging the slider
        if (animating) {
            setAnimating(false);
        }

        const time = parseFloat(e.target.value);
        timeValue.textContent = time.toFixed(2);
        renderer.setTime(time);
        // Update skeletal animation to match the current time (consistent with export)
        renderer.updateAnimationFromTime(time);
        update();
    });

    resWidthSlider.addEventListener('input', (e) => {
        const width = parseInt(e.target.value);
        resWidthValue.textContent = width;
        renderer.setResolution(width, renderer.resolutionY);
        update();
    });

    resHeightSlider.addEventListener('input', (e) => {
        const height = parseInt(e.target.value);
        resHeightValue.textContent = height;
        renderer.setResolution(renderer.resolutionX, height);
        update();
    });

    sampleSlider.addEventListener('input', (e) => {
        const samples = parseInt(e.target.value);
        sampleValue.textContent = samples;
        renderer.setSampleCount(samples);
        update();
    });

    fieldResolutionSlider.addEventListener('input', (e) => {
        const fieldResolution = parseInt(e.target.value);
        fieldResolutionValue.textContent = fieldResolution;
        renderer.setFieldResolution(fieldResolution);
        update();
    });

    outputModeSelect.addEventListener('change', (e) => {
        const mode = parseInt(e.target.value);
        renderer.setOutputMode(mode);
        update();
    });

    fieldTypeSelect.addEventListener('change', (e) => {
        const type = parseInt(e.target.value);
        console.log('Field type changed to:', type);
        renderer.setFieldType(type);
        console.log('Renderer fieldType is now:', renderer.fieldType);
        update();
    });

    // RNG type selector
    const rngTypeSelect = document.getElementById('rngTypeSelect');
    rngTypeSelect.addEventListener('change', (e) => {
        const type = parseInt(e.target.value);
        renderer.rngType = type;
        update();
    });

    // Seed input
    const seedInput = document.getElementById('seedInput');
    seedInput.addEventListener('input', (e) => {
        const seed = parseFloat(e.target.value) || 0;
        renderer.fieldSeed = seed;
        update();
    });

    // Estimator mode selector (Biased vs Unbiased)
    const estimatorModeSelect = document.getElementById('estimatorModeSelect');
    estimatorModeSelect.addEventListener('change', (e) => {
        const mode = parseInt(e.target.value);
        renderer.estimatorMode = mode;
        update();
    });

    // Custom mesh controls
    const objFileInput = document.getElementById('objFileInput');
    const meshDropZone = document.getElementById('meshDropZone');
    const meshStatusGroup = document.getElementById('meshStatusGroup');
    const meshStatus = document.getElementById('meshStatus');
    const meshYawSlider = document.getElementById('meshYawSlider');
    const meshYawValue = document.getElementById('meshYawValue');
    const meshPitchSlider = document.getElementById('meshPitchSlider');
    const meshPitchValue = document.getElementById('meshPitchValue');
    const meshYawRotationSpeedSlider = document.getElementById('meshYawRotationSpeedSlider');
    const meshYawRotationSpeedValue = document.getElementById('meshYawRotationSpeedValue');
    const meshScaleSlider = document.getElementById('meshScaleSlider');
    const meshScaleValue = document.getElementById('meshScaleValue');
    const centerBackgroundCubeCheckbox = document.getElementById('centerBackgroundCube');

    makeEditable(meshYawValue, meshYawSlider, (value) => {
        renderer.meshCameraYaw = value;
        update();
    }, { min: -180, max: 180, step: 1 });

    makeEditable(meshPitchValue, meshPitchSlider, (value) => {
        renderer.meshCameraPitch = value;
        update();
    }, { min: -90, max: 90, step: 1 });

    makeEditable(meshYawRotationSpeedValue, meshYawRotationSpeedSlider, (value) => {
        renderer.meshCameraYawRotationSpeed = value;
        update();
    }, { min: -180, max: 180, step: 1 });

    makeEditable(meshScaleValue, meshScaleSlider, (value) => {
        renderer.meshScale = value;
        update();
    }, { min: 0.1, max: 10, step: 0.1, isFloat: true, decimalPlaces: 1 });

    // Supported mesh file extensions
    const SUPPORTED_MESH_EXTENSIONS = ['.obj', '.gltf', '.glb'];

    // Check if file is a supported mesh format
    function isSupportedMeshFile(filename) {
        const lower = filename.toLowerCase();
        return SUPPORTED_MESH_EXTENSIONS.some(ext => lower.endsWith(ext));
    }

    // Helper function to load mesh file (OBJ, GLTF, or GLB)
    async function loadMeshFile(file) {
        try {
            const filename = file.name;
            const lowerName = filename.toLowerCase();
            let mesh;

            if (lowerName.endsWith('.gltf') || lowerName.endsWith('.glb')) {
                // GLTF/GLB format - read as ArrayBuffer
                const arrayBuffer = await file.arrayBuffer();
                mesh = await renderer.loadMesh(arrayBuffer, filename);
            } else {
                // OBJ format - read as text
                const text = await file.text();
                mesh = await renderer.loadMesh(text, filename);
            }

            if (meshStatusGroup) meshStatusGroup.style.display = 'block';
            if (meshStatus) meshStatus.textContent = `${filename} (${mesh.triangleCount} tris)`;
            if (meshDropZone) {
                meshDropZone.classList.remove('is-error', 'is-dragover');
                meshDropZone.classList.add('is-loaded');
                meshDropZone.innerHTML = `<span class="drop-zone-icon" aria-hidden="true">&#10003;</span><span class="drop-zone-label">Loaded ${filename}</span><span class="drop-zone-hint">${mesh.triangleCount} triangles - choose another file to replace it</span>`;
            }

            // Populate animation dropdown
            const animationSelect = document.getElementById('animationSelect');
            const animationSelectGroup = document.getElementById('animationSelectGroup');
            const animationSpeedGroup = document.getElementById('animationSpeedGroup');

            if (animationSelect && mesh.animations && mesh.animations.length > 0) {
                // Clear existing options and add "None" option
                animationSelect.innerHTML = '<option value="-1">None (static pose)</option>';

                // Add animation options
                mesh.animations.forEach((anim, index) => {
                    const option = document.createElement('option');
                    option.value = index.toString();
                    option.textContent = `${anim.name} (${anim.duration.toFixed(2)}s)`;
                    animationSelect.appendChild(option);
                });

                // Show animation controls
                if (animationSelectGroup) animationSelectGroup.style.display = 'block';
                if (animationSpeedGroup) animationSpeedGroup.style.display = 'block';

                // Auto-select the first animation if available
                animationSelect.value = '0';
                renderer.setAnimation(0);

                // Auto-start animation playback
                if (!animating) {
                    setAnimating(true);
                }

                console.log(`Found ${mesh.animations.length} animations`);
            } else {
                // Hide animation controls if no animations
                if (animationSelectGroup) animationSelectGroup.style.display = 'none';
                if (animationSpeedGroup) animationSpeedGroup.style.display = 'none';
            }

            // Auto-switch to Custom Mesh mode when a mesh is loaded
            update();
            console.log(`Loaded mesh: ${filename} with ${mesh.triangleCount} triangles`);
        } catch (err) {
            console.error('Failed to load mesh:', err);
            alert('Failed to load mesh file: ' + err.message);
            if (meshDropZone) {
                meshDropZone.classList.remove('is-loaded', 'is-dragover');
                meshDropZone.classList.add('is-error');
            }
        }
    }

    // Click to browse (for drop zone in Custom Mesh panel)
    if (meshDropZone) {
        meshDropZone.addEventListener('click', () => {
            if (objFileInput) objFileInput.click();
        });

        // Drag and drop handlers for the drop zone
        meshDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            meshDropZone.classList.add('is-dragover');
        });

        meshDropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            meshDropZone.classList.remove('is-dragover');
        });

        meshDropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            meshDropZone.classList.remove('is-dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (isSupportedMeshFile(file.name)) {
                    await loadMeshFile(file);
                } else {
                    alert('Please drop a supported mesh file (.obj, .gltf, .glb)');
                }
            }
        });
    }

    // GLOBAL drag and drop - allows dropping OBJ files anywhere on the page
    const globalDropOverlay = document.getElementById('globalDropOverlay');
    let dragCounter = 0;  // Track nested drag events

    document.body.addEventListener('dragenter', (e) => {
        // Only show overlay for file drags (not slider drags)
        if (!e.dataTransfer || !e.dataTransfer.types.includes('Files')) {
            return;
        }
        e.preventDefault();
        dragCounter++;
        if (globalDropOverlay) {
            globalDropOverlay.style.display = 'block';
        }
    });

    document.body.addEventListener('dragleave', (e) => {
        // Only handle file drags
        if (!e.dataTransfer || !e.dataTransfer.types.includes('Files')) {
            return;
        }
        e.preventDefault();
        dragCounter--;
        if (dragCounter === 0 && globalDropOverlay) {
            globalDropOverlay.style.display = 'none';
        }
    });

    document.body.addEventListener('dragover', (e) => {
        // Only handle file drags
        if (e.dataTransfer && e.dataTransfer.types.includes('Files')) {
            e.preventDefault();
        }
    });

    document.body.addEventListener('drop', async (e) => {
        // Only handle file drops
        if (!e.dataTransfer || !e.dataTransfer.types.includes('Files')) {
            return;
        }
        e.preventDefault();
        dragCounter = 0;
        if (globalDropOverlay) {
            globalDropOverlay.style.display = 'none';
        }
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (isSupportedMeshFile(file.name)) {
                console.log('Global drop: Loading mesh file...');
                await loadMeshFile(file);
            }
        }
    });

    if (objFileInput) {
        objFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            await loadMeshFile(file);
        });
    }

    // Animation controls
    const animationSelect = document.getElementById('animationSelect');
    const animationSpeedSlider = document.getElementById('animationSpeedSlider');
    const animationSpeedValue = document.getElementById('animationSpeedValue');

    makeEditable(animationSpeedValue, animationSpeedSlider, (value) => {
        renderer.setAnimationSpeed(value);
    }, { min: 0, max: 3, step: 0.1, isFloat: true, decimalPlaces: 1 });

    if (animationSelect) {
        animationSelect.addEventListener('change', (e) => {
            const index = parseInt(e.target.value);
            renderer.setAnimation(index);
            update();
        });
    }

    if (animationSpeedSlider) {
        animationSpeedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            if (animationSpeedValue) animationSpeedValue.textContent = speed.toFixed(1);
            renderer.setAnimationSpeed(speed);
        });
    }

    if (meshYawSlider) {
        meshYawSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (meshYawValue) meshYawValue.textContent = value;
            renderer.meshCameraYaw = value;
            update();
        });
    }

    if (meshPitchSlider) {
        meshPitchSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (meshPitchValue) meshPitchValue.textContent = value;
            renderer.meshCameraPitch = value;
            update();
        });
    }

    if (meshYawRotationSpeedSlider) {
        meshYawRotationSpeedSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (meshYawRotationSpeedValue) meshYawRotationSpeedValue.textContent = value;
            renderer.meshCameraYawRotationSpeed = value;
            update();
        });
    }


    if (meshScaleSlider) {
        meshScaleSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (meshScaleValue) meshScaleValue.textContent = value.toFixed(1);
            renderer.meshScale = value;
            update();
        });
    }

    if (centerBackgroundCubeCheckbox) {
        centerBackgroundCubeCheckbox.addEventListener('change', (e) => {
            renderer.centerBackgroundCube = e.target.checked;
            update();
        });
    }

    // Field of view slider for custom mesh mode
    const meshFovSlider = document.getElementById('meshFovSlider');
    const meshFovValue = document.getElementById('meshFovValue');

    makeEditable(meshFovValue, meshFovSlider, (value) => {
        renderer.meshFov = value;
        update();
    }, { min: 10, max: 120, step: 1 });

    if (meshFovSlider) {
        meshFovSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (meshFovValue) meshFovValue.textContent = value.toFixed(0);
            renderer.meshFov = value;
            update();
        });
    }

    // The mesh controls are always active, so seed the renderer from the markup's
    // slider defaults up front rather than waiting for the first input event.
    if (meshYawSlider) renderer.meshCameraYaw = parseFloat(meshYawSlider.value);
    if (meshPitchSlider) renderer.meshCameraPitch = parseFloat(meshPitchSlider.value);
    if (meshYawRotationSpeedSlider) renderer.meshCameraYawRotationSpeed = parseFloat(meshYawRotationSpeedSlider.value);
    if (meshScaleSlider) renderer.meshScale = parseFloat(meshScaleSlider.value);
    if (centerBackgroundCubeCheckbox) renderer.centerBackgroundCube = centerBackgroundCubeCheckbox.checked;
    if (meshFovSlider) renderer.meshFov = parseFloat(meshFovSlider.value);

    // Animation. It autoplays so the effect is visible without input, except
    // when the user has asked for reduced motion (WCAG 2.3.3). Either the button
    // or SPACE stops it, which is what SC 2.2.2 requires of moving content.
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let animating = false;
    let animationId = null;
    let lastAnimationTime = performance.now();
    const playPauseButton = document.getElementById('playPauseButton');

    function animate() {
        if (!animating) return;

        const now = performance.now();
        const deltaTime = (now - lastAnimationTime) / 1000;  // Convert to seconds
        lastAnimationTime = now;

        // Skip automatic time updates during export - export controls time manually
        if (!renderer.isExporting) {
            // Increment the time slider
            const time = parseFloat(timeSlider.value);
            const newTime = (time + deltaTime) % 5;  // Use actual deltaTime for smooth playback
            timeSlider.value = newTime;
            timeValue.textContent = newTime.toFixed(2);
            renderer.setTime(newTime);

            // Update skeletal animation using the same time value as mesh rotation
            // This is consistent with how export works: both use effectiveTime
            renderer.updateAnimationFromTime(newTime);

            update();
        }

        animationId = requestAnimationFrame(animate);
    }

    function setAnimating(shouldAnimate) {
        animating = shouldAnimate;
        if (animating) {
            lastAnimationTime = performance.now();  // Reset time when starting
            animate();
        } else if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        if (playPauseButton) {
            playPauseButton.classList.toggle('is-playing', animating);
            playPauseButton.setAttribute('aria-label', animating ? 'Pause animation' : 'Play animation');
        }
    }

    if (playPauseButton) {
        playPauseButton.addEventListener('click', () => setAnimating(!animating));
    }

    const interactiveRoles = new Set([
        'button', 'checkbox', 'combobox', 'grid', 'gridcell', 'link', 'listbox',
        'menu', 'menubar', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
        'option', 'radio', 'radiogroup', 'scrollbar', 'searchbox', 'slider',
        'spinbutton', 'switch', 'tab', 'tablist', 'textbox', 'tree', 'treegrid',
        'treeitem',
    ]);

    function eventTargetsInteractiveElement(event) {
        const path = typeof event.composedPath === 'function'
            ? event.composedPath()
            : [event.target];
        return path.some((target) => {
            if (!(target instanceof Element)) return false;
            if (target.matches(
                'button, input, select, textarea, summary, a[href], area[href], '
                + 'audio[controls], video[controls], [contenteditable]:not([contenteditable="false"])',
            ) || target.isContentEditable) {
                return true;
            }
            const roles = (target.getAttribute('role') || '').trim().split(/\s+/);
            return roles.some((role) => interactiveRoles.has(role));
        });
    }

    document.addEventListener('keydown', (e) => {
        const hasModifier = e.altKey || e.ctrlKey || e.metaKey || e.shiftKey;
        if (
            e.code !== 'Space'
            || e.defaultPrevented
            || e.repeat
            || hasModifier
            || eventTargetsInteractiveElement(e)
        ) {
            return;
        }
        e.preventDefault();
        setAnimating(!animating);
    });

    // Initial render
    update();

    setAnimating(!prefersReducedMotion);

    // Now that the first frame has been submitted, compile the remaining workgroup-size
    // variants in the background, so changing the sample count later does not stall on a
    // multi-second shader compile. Waiting on the first frame keeps these compiles from
    // competing with the one the first frame needs.
    renderer.device.queue.onSubmittedWorkDone()
        .then(() => renderer.warmComputePipelines())
        .catch((error) => console.warn('Skipped warming compute pipelines:', error));

    // Load and apply saved settings
    const savedSettings = loadSettings();
    if (savedSettings) {
        console.log('Restoring saved settings:', savedSettings);
        applySettings(renderer, savedSettings, (settings) => {
            // Update UI to match loaded settings
            if (settings.resolutionX) {
                resWidthSlider.value = settings.resolutionX;
                resWidthValue.textContent = settings.resolutionX;
            }
            if (settings.resolutionY) {
                resHeightSlider.value = settings.resolutionY;
                resHeightValue.textContent = settings.resolutionY;
            }
            if (settings.fieldResolution) {
                fieldResolutionSlider.value = settings.fieldResolution;
                fieldResolutionValue.textContent = settings.fieldResolution;
            }
            if (settings.sampleCount) {
                sampleSlider.value = settings.sampleCount;
                sampleValue.textContent = settings.sampleCount;
            }
            if (settings.time !== undefined) {
                timeSlider.value = settings.time;
                timeValue.textContent = settings.time.toFixed(2);
            }
            if (settings.outputMode !== undefined) {
                outputModeSelect.value = settings.outputMode;
            }
            if (settings.fieldType !== undefined) {
                fieldTypeSelect.value = renderer.fieldType;
            }
            if (settings.rngType !== undefined) {
                rngTypeSelect.value = settings.rngType;
            }
            if (settings.centerBackgroundCube !== undefined) {
                centerBackgroundCubeCheckbox.checked = settings.centerBackgroundCube;
            }
            if (settings.showHistogram !== undefined) {
                showHistogramCheckbox.checked = settings.showHistogram;
                histogramCanvas.parentElement.style.display = settings.showHistogram ? 'block' : 'none';
            }
        });
        update();
    }

    // Save settings whenever they change (debounced)
    let saveTimeout = null;
    function scheduleSaveSettings() {
        if (saveTimeout) clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => saveSettings(renderer), 500);
    }

    // Add save triggers to all controls
    [timeSlider, resWidthSlider, resHeightSlider, sampleSlider,
     fieldResolutionSlider].forEach(slider => {
        if (slider) slider.addEventListener('input', scheduleSaveSettings);
    });

    [outputModeSelect, fieldTypeSelect, rngTypeSelect].forEach(select => {
        if (select) select.addEventListener('change', scheduleSaveSettings);
    });

    if (showHistogramCheckbox) showHistogramCheckbox.addEventListener('change', scheduleSaveSettings);
    if (centerBackgroundCubeCheckbox) centerBackgroundCubeCheckbox.addEventListener('change', scheduleSaveSettings);

    // Export button handler
    const exportButton = document.getElementById('exportButton');
    const exportNoiseButton = document.getElementById('exportNoiseButton');
    const exportLengthSelect = document.getElementById('exportLengthSelect');
    const exportFpsSelect = document.getElementById('exportFpsSelect');

    // Export Noise button (value only) - supports batch seed export
    const batchSeedsInput = document.getElementById('batchSeedsInput');
    if (exportNoiseButton) {
        exportNoiseButton.addEventListener('click', async () => {
            const exportLength = parseFloat(exportLengthSelect.value);  // 1-5 seconds
            const exportFps = parseFloat(exportFpsSelect.value);  // 10-60 FPS
            const numFrames = Math.round(exportLength * exportFps);
            const batchSeeds = Math.max(1, parseInt(batchSeedsInput.value) || 1);
            const startSeed = renderer.fieldSeed;
            const originalText = exportNoiseButton.textContent;
            const baseFilename = `tabula_rasa_${renderer.resolutionX}x${renderer.resolutionY}_${numFrames}frames_${exportFps}fps_noise`;

            exportNoiseButton.disabled = true;

            try {
                // For batch export, get directory handle first (required for large files)
                let dirHandle = null;
                if (batchSeeds > 1 && window.showDirectoryPicker) {
                    try {
                        dirHandle = await window.showDirectoryPicker({
                            mode: 'readwrite',
                            startIn: 'downloads'
                        });
                    } catch (e) {
                        // User cancelled - abort batch export
                        console.log('Directory picker cancelled, aborting batch export');
                        exportNoiseButton.textContent = 'Cancelled';
                        setTimeout(() => {
                            exportNoiseButton.textContent = originalText;
                            exportNoiseButton.disabled = false;
                        }, 1500);
                        return;
                    }
                }

                for (let seedIdx = 0; seedIdx < batchSeeds; seedIdx++) {
                    const currentSeed = startSeed + seedIdx;
                    renderer.fieldSeed = currentSeed;

                    // Update seed input to show current seed
                    if (seedInput) seedInput.value = currentSeed;

                    const buffer = await renderer.exportNoiseToSafetensors(numFrames, (current, total) => {
                        const framePercent = Math.round((current / total) * 100);
                        const overallPercent = Math.round(((seedIdx + current / total) / batchSeeds) * 100);
                        exportNoiseButton.textContent = batchSeeds > 1
                            ? `Seed ${seedIdx + 1}/${batchSeeds} (${framePercent}%) - Overall ${overallPercent}%`
                            : `Exporting... ${framePercent}%`;
                    }, exportFps, dirHandle);

                    // Download the file (skip if streamed directly to disk or written to directory)
                    if (buffer !== 'streamed') {
                        const blob = new Blob([buffer], { type: 'application/octet-stream' });

                        if (dirHandle) {
                            // Write to directory handle
                            const seedSuffix = batchSeeds > 1 ? `.${currentSeed}` : '';
                            const filename = `${baseFilename}${seedSuffix}.safetensors`;
                            const fileHandle = await dirHandle.getFileHandle(filename, { create: true });
                            const writable = await fileHandle.createWritable();
                            await writable.write(blob);
                            await writable.close();
                        } else {
                            // Fall back to download link
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            const seedSuffix = batchSeeds > 1 ? `.${currentSeed}` : '';
                            a.download = `${baseFilename}${seedSuffix}.safetensors`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                        }
                    }
                }

                exportNoiseButton.textContent = 'Export Complete!';
                setTimeout(() => {
                    exportNoiseButton.textContent = originalText;
                    exportNoiseButton.disabled = false;
                }, 2000);
            } catch (error) {
                console.error('Export failed:', error);
                exportNoiseButton.textContent = 'Export Failed!';
                setTimeout(() => {
                    exportNoiseButton.textContent = originalText;
                    exportNoiseButton.disabled = false;
                }, 2000);
            }
        });
    }

    // Export All button (all attributes)
    if (exportButton) {
        exportButton.addEventListener('click', async () => {
            const exportLength = parseInt(exportLengthSelect.value);  // 1-5 seconds
            const exportFps = parseInt(exportFpsSelect.value);  // 10-60 FPS
            const numFrames = exportLength * exportFps;
            const originalText = exportButton.textContent;

            exportButton.disabled = true;
            exportButton.textContent = 'Exporting... 0%';

            try {
                const buffer = await renderer.exportToSafetensors(numFrames, (current, total) => {
                    const percent = Math.round((current / total) * 100);
                    exportButton.textContent = `Exporting... ${percent}%`;
                }, exportFps);

                // Download the file (skip if streamed directly to disk)
                if (buffer !== 'streamed') {
                    const blob = new Blob([buffer], { type: 'application/octet-stream' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `tabula_rasa_${renderer.resolutionX}x${renderer.resolutionY}_${numFrames}frames_${exportFps}fps.safetensors`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }

                exportButton.textContent = 'Export Complete!';
                setTimeout(() => {
                    exportButton.textContent = originalText;
                    exportButton.disabled = false;
                }, 2000);
            } catch (error) {
                console.error('Export failed:', error);
                exportButton.textContent = 'Export Failed!';
                setTimeout(() => {
                    exportButton.textContent = originalText;
                exportButton.disabled = false;
                }, 2000);
            }
        });
    }

    console.log('Tabula Rasa ready - Full Count-Unique Estimator');
});
