/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

'use strict';

(function exposeSafetensorReader(global) {
    const MAX_HEADER_BYTES = 100 * 1024 * 1024;
    const MAX_FRAME_PIXELS = 20 * 1024 * 1024;
    const DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024;

    function float16ToNumber(bits) {
        const sign = bits & 0x8000 ? -1 : 1;
        const exponent = (bits >>> 10) & 0x1f;
        const fraction = bits & 0x03ff;
        if (exponent === 0) {
            return fraction === 0
                ? (sign < 0 ? -0 : 0)
                : sign * 2 ** -14 * (fraction / 1024);
        }
        if (exponent === 0x1f) {
            return fraction === 0 ? sign * Infinity : NaN;
        }
        return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
    }

    const bfloatScratch = new DataView(new ArrayBuffer(4));
    function bfloat16ToNumber(bits) {
        bfloatScratch.setUint32(0, bits << 16, false);
        return bfloatScratch.getFloat32(0, false);
    }

    const DTYPE_INFO = Object.freeze({
        BOOL: { bytes: 1, read: (view, offset) => view.getUint8(offset) === 0 ? 0 : 1 },
        I8: { bytes: 1, read: (view, offset) => view.getInt8(offset) },
        U8: { bytes: 1, read: (view, offset) => view.getUint8(offset) },
        I16: { bytes: 2, read: (view, offset) => view.getInt16(offset, true) },
        U16: { bytes: 2, read: (view, offset) => view.getUint16(offset, true) },
        I32: { bytes: 4, read: (view, offset) => view.getInt32(offset, true) },
        U32: { bytes: 4, read: (view, offset) => view.getUint32(offset, true) },
        I64: { bytes: 8, read: (view, offset) => Number(view.getBigInt64(offset, true)) },
        U64: { bytes: 8, read: (view, offset) => Number(view.getBigUint64(offset, true)) },
        F16: { bytes: 2, read: (view, offset) => float16ToNumber(view.getUint16(offset, true)) },
        BF16: { bytes: 2, read: (view, offset) => bfloat16ToNumber(view.getUint16(offset, true)) },
        F32: { bytes: 4, read: (view, offset) => view.getFloat32(offset, true) },
        F64: { bytes: 8, read: (view, offset) => view.getFloat64(offset, true) },
    });

    function throwIfAborted(signal) {
        if (signal?.aborted) {
            throw new DOMException('The safetensors read was cancelled.', 'AbortError');
        }
    }

    function isPlainObject(value) {
        return value !== null && typeof value === 'object' && !Array.isArray(value);
    }

    function safeProduct(values, context) {
        let result = 1;
        for (const value of values) {
            if (!Number.isSafeInteger(value) || value < 0) {
                throw new Error(`${context} contains a dimension that is not a nonnegative safe integer.`);
            }
            if (value === 0) return 0;
            if (result > Number.MAX_SAFE_INTEGER / value) {
                throw new Error(`${context} has an unsafe element count.`);
            }
            result *= value;
        }
        return result;
    }

    function safeMultiply(left, right, context) {
        if (left !== 0 && right > Number.MAX_SAFE_INTEGER / left) {
            throw new Error(`${context} exceeds JavaScript's safe byte range.`);
        }
        return left * right;
    }

    function deriveLayout(shape) {
        if (shape.length === 4) {
            const [frames, height, width, channels] = shape;
            return {
                rank: 4,
                frames,
                height,
                width,
                channels,
                visualizable: frames > 0 && height > 0 && width > 0 && channels > 0,
            };
        }
        if (shape.length === 3) {
            const [frames, height, width] = shape;
            return {
                rank: 3,
                frames,
                height,
                width,
                channels: 1,
                visualizable: frames > 0 && height > 0 && width > 0,
            };
        }
        if (shape.length === 2) {
            const [height, width] = shape;
            return {
                rank: 2,
                frames: 1,
                height,
                width,
                channels: 1,
                visualizable: height > 0 && width > 0,
            };
        }
        return {
            rank: shape.length,
            frames: 0,
            height: 0,
            width: 0,
            channels: 0,
            visualizable: false,
        };
    }

    async function readBlobRange(blob, start, end, signal) {
        throwIfAborted(signal);
        const buffer = await blob.slice(start, end).arrayBuffer();
        throwIfAborted(signal);
        if (buffer.byteLength !== end - start) {
            throw new Error('The safetensors file ended during a requested read.');
        }
        return buffer;
    }

    class SafetensorReader {
        constructor(blob, header, payloadOffset, tensors, metadata) {
            this.blob = blob;
            this.header = header;
            this.payloadOffset = payloadOffset;
            this.metadata = metadata;
            this.tensors = tensors;
        }

        static async open(blob, options = {}) {
            const { signal } = options;
            if (!blob || typeof blob.slice !== 'function' || !Number.isSafeInteger(blob.size)) {
                throw new TypeError('SafetensorReader.open expects a File or Blob.');
            }
            if (blob.size < 8) {
                throw new Error('The safetensors file is shorter than its 8-byte header prefix.');
            }

            const prefix = await readBlobRange(blob, 0, 8, signal);
            const headerLengthBig = new DataView(prefix).getBigUint64(0, true);
            if (headerLengthBig === 0n) {
                throw new Error('The safetensors header is empty.');
            }
            if (headerLengthBig > BigInt(MAX_HEADER_BYTES)) {
                throw new Error(`The safetensors header exceeds the ${MAX_HEADER_BYTES}-byte safety limit.`);
            }
            if (headerLengthBig > BigInt(Number.MAX_SAFE_INTEGER)) {
                throw new Error('The safetensors header length is not safely representable.');
            }

            const headerLength = Number(headerLengthBig);
            const payloadOffset = 8 + headerLength;
            if (!Number.isSafeInteger(payloadOffset) || payloadOffset > blob.size) {
                throw new Error('The safetensors header extends beyond the end of the file.');
            }

            const headerBuffer = await readBlobRange(blob, 8, payloadOffset, signal);
            let headerText;
            try {
                headerText = new TextDecoder('utf-8', { fatal: true }).decode(headerBuffer);
            } catch (error) {
                throw new Error(`The safetensors header is not valid UTF-8: ${error.message}`);
            }

            let header;
            try {
                header = JSON.parse(headerText);
            } catch (error) {
                throw new Error(`The safetensors header is not valid JSON: ${error.message}`);
            }
            if (!isPlainObject(header)) {
                throw new Error('The safetensors header must be a JSON object.');
            }

            const tensors = new Map();
            const ranges = [];
            for (const [name, rawInfo] of Object.entries(header)) {
                if (name === '__metadata__') continue;
                if (!isPlainObject(rawInfo)) {
                    throw new Error(`Tensor "${name}" must have an object descriptor.`);
                }

                const dtype = rawInfo.dtype;
                const dtypeInfo = DTYPE_INFO[dtype];
                if (!dtypeInfo) {
                    throw new Error(`Tensor "${name}" uses unsupported dtype "${dtype}".`);
                }
                if (!Array.isArray(rawInfo.shape)) {
                    throw new Error(`Tensor "${name}" is missing a valid shape array.`);
                }
                const shape = [...rawInfo.shape];
                const elementCount = safeProduct(shape, `Tensor "${name}" shape`);

                const offsets = rawInfo.data_offsets;
                if (
                    !Array.isArray(offsets)
                    || offsets.length !== 2
                    || !offsets.every(Number.isSafeInteger)
                    || offsets.some((offset) => offset < 0)
                    || offsets[0] > offsets[1]
                ) {
                    throw new Error(`Tensor "${name}" has invalid data_offsets.`);
                }

                const [relativeStart, relativeEnd] = offsets;
                const expectedBytes = safeMultiply(elementCount, dtypeInfo.bytes, `Tensor "${name}" payload`);
                if (relativeEnd - relativeStart !== expectedBytes) {
                    throw new Error(`Tensor "${name}" payload length does not match its shape and dtype.`);
                }

                const absoluteStart = payloadOffset + relativeStart;
                const absoluteEnd = payloadOffset + relativeEnd;
                if (
                    !Number.isSafeInteger(absoluteStart)
                    || !Number.isSafeInteger(absoluteEnd)
                    || absoluteStart < payloadOffset
                    || absoluteEnd > blob.size
                ) {
                    throw new Error(`Tensor "${name}" data range extends beyond the file.`);
                }

                const descriptor = Object.freeze({
                    name,
                    dtype,
                    shape: Object.freeze(shape),
                    dataOffsets: Object.freeze([relativeStart, relativeEnd]),
                    absoluteStart,
                    absoluteEnd,
                    elementCount,
                    elementBytes: dtypeInfo.bytes,
                    byteLength: expectedBytes,
                    layout: Object.freeze(deriveLayout(shape)),
                });
                tensors.set(name, descriptor);
                if (expectedBytes > 0) {
                    ranges.push({ name, start: relativeStart, end: relativeEnd });
                }
            }

            ranges.sort((left, right) => left.start - right.start || left.end - right.end);
            for (let index = 1; index < ranges.length; index++) {
                if (ranges[index].start < ranges[index - 1].end) {
                    throw new Error(`Tensor payloads "${ranges[index - 1].name}" and "${ranges[index].name}" overlap.`);
                }
            }

            const rawMetadata = header.__metadata__;
            const metadata = isPlainObject(rawMetadata) ? Object.freeze({ ...rawMetadata }) : Object.freeze({});
            return new SafetensorReader(blob, header, payloadOffset, tensors, metadata);
        }

        getTensorNames() {
            return [...this.tensors.keys()];
        }

        getTensorInfo(name) {
            return this.tensors.get(name) || null;
        }

        async readFrame(name, frameIndex = 0, channel = 0, options = {}) {
            const result = await this.readFrameChannels(name, frameIndex, [channel], options);
            return {
                data: result.data,
                width: result.width,
                height: result.height,
                frameIndex: result.frameIndex,
                channel,
                descriptor: result.descriptor,
            };
        }

        async readFrameChannels(name, frameIndex = 0, channels = [0], options = {}) {
            const descriptor = this.tensors.get(name);
            if (!descriptor) {
                throw new Error(`Tensor "${name}" does not exist.`);
            }
            const layout = descriptor.layout;
            if (!layout.visualizable) {
                throw new Error(`Tensor "${name}" with rank ${layout.rank} cannot be visualized as an image sequence.`);
            }
            if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= layout.frames) {
                throw new RangeError(`Frame ${frameIndex} is outside tensor "${name}".`);
            }
            if (
                !Array.isArray(channels)
                || channels.length === 0
                || channels.some((channel) => !Number.isInteger(channel) || channel < 0 || channel >= layout.channels)
                || new Set(channels).size !== channels.length
            ) {
                throw new RangeError(`Selected channels are invalid for tensor "${name}".`);
            }

            const pixelCount = safeMultiply(layout.height, layout.width, `Tensor "${name}" frame`);
            if (pixelCount > MAX_FRAME_PIXELS) {
                throw new Error(`Tensor "${name}" frame exceeds the ${MAX_FRAME_PIXELS}-pixel display limit.`);
            }

            const { signal, chunkBytes = DEFAULT_CHUNK_BYTES } = options;
            if (!Number.isSafeInteger(chunkBytes) || chunkBytes < descriptor.elementBytes) {
                throw new RangeError('chunkBytes must be a positive safe integer large enough for one element.');
            }

            const selectedChannelCount = channels.length;
            const outputCount = safeMultiply(pixelCount, selectedChannelCount, `Tensor "${name}" selected frame`);
            const values = new Float64Array(outputCount);
            const pixelBytes = safeMultiply(layout.channels, descriptor.elementBytes, `Tensor "${name}" pixel stride`);
            if (chunkBytes < pixelBytes) {
                throw new RangeError('chunkBytes must be large enough for one complete tensor pixel.');
            }
            const frameBytes = safeMultiply(pixelCount, pixelBytes, `Tensor "${name}" frame size`);
            const frameStart = descriptor.absoluteStart + frameIndex * frameBytes;
            const pixelsPerChunk = Math.max(1, Math.floor(chunkBytes / pixelBytes));
            const dtypeInfo = DTYPE_INFO[descriptor.dtype];

            for (let pixelStart = 0; pixelStart < pixelCount; pixelStart += pixelsPerChunk) {
                throwIfAborted(signal);
                const pixelsInChunk = Math.min(pixelsPerChunk, pixelCount - pixelStart);
                const byteStart = frameStart + pixelStart * pixelBytes;
                const byteEnd = byteStart + pixelsInChunk * pixelBytes;
                const buffer = await readBlobRange(this.blob, byteStart, byteEnd, signal);
                const view = new DataView(buffer);
                for (let localPixel = 0; localPixel < pixelsInChunk; localPixel++) {
                    const sourcePixelOffset = localPixel * pixelBytes;
                    const destinationPixel = (pixelStart + localPixel) * selectedChannelCount;
                    for (let selectedIndex = 0; selectedIndex < selectedChannelCount; selectedIndex++) {
                        const valueOffset = sourcePixelOffset + channels[selectedIndex] * descriptor.elementBytes;
                        values[destinationPixel + selectedIndex] = dtypeInfo.read(view, valueOffset);
                    }
                }
            }

            return {
                data: values,
                width: layout.width,
                height: layout.height,
                frameIndex,
                channels: [...channels],
                descriptor,
            };
        }
    }

    SafetensorReader.MAX_HEADER_BYTES = MAX_HEADER_BYTES;
    SafetensorReader.MAX_FRAME_PIXELS = MAX_FRAME_PIXELS;
    SafetensorReader.DTYPE_INFO = DTYPE_INFO;
    SafetensorReader.float16ToNumber = float16ToNumber;
    SafetensorReader.bfloat16ToNumber = bfloat16ToNumber;

    global.SafetensorReader = SafetensorReader;
})(window);
