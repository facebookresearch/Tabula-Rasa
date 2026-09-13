/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

'use strict';

(function exposeSafetensorViewer(global) {
    function calculateStats(data) {
        let count = 0;
        let nonfinite = 0;
        let min = Infinity;
        let max = -Infinity;
        let mean = 0;
        let sumSquaredDifference = 0;

        for (const value of data) {
            if (!Number.isFinite(value)) {
                nonfinite++;
                continue;
            }
            count++;
            min = Math.min(min, value);
            max = Math.max(max, value);
            const delta = value - mean;
            mean += delta / count;
            sumSquaredDifference += delta * (value - mean);
        }

        if (count === 0) {
            return {
                count: 0,
                nonfinite,
                min: NaN,
                max: NaN,
                mean: NaN,
                std: NaN,
            };
        }
        return {
            count,
            nonfinite,
            min,
            max,
            mean,
            std: Math.sqrt(sumSquaredDifference / count),
        };
    }

    function calculateRange(stats) {
        if (stats.count === 0) return [-1, 1];
        if (stats.min === stats.max) {
            const padding = Math.abs(stats.min) * 0.05 || 1;
            return [stats.min - padding, stats.max + padding];
        }
        return [stats.min, stats.max];
    }

    function buildHistogram(data, min, max, binCount = 64) {
        const bins = new Uint32Array(binCount);
        let finiteCount = 0;
        const span = max - min;
        for (const value of data) {
            if (!Number.isFinite(value)) continue;
            finiteCount++;
            const normalized = span > 0 ? (value - min) / span : 0.5;
            const index = Math.max(0, Math.min(binCount - 1, Math.floor(normalized * binCount)));
            bins[index]++;
        }
        return { bins, finiteCount };
    }

    function createStatsAccumulator() {
        return {
            count: 0,
            nonfinite: 0,
            min: Infinity,
            max: -Infinity,
            mean: 0,
            sumSquaredDifference: 0,
        };
    }

    function addToStats(accumulator, data, signal) {
        for (let index = 0; index < data.length; index++) {
            if ((index & 0xffff) === 0 && signal?.aborted) {
                throw new DOMException('The histogram scan was cancelled.', 'AbortError');
            }
            const value = data[index];
            if (!Number.isFinite(value)) {
                accumulator.nonfinite++;
                continue;
            }
            accumulator.count++;
            accumulator.min = Math.min(accumulator.min, value);
            accumulator.max = Math.max(accumulator.max, value);
            const delta = value - accumulator.mean;
            accumulator.mean += delta / accumulator.count;
            accumulator.sumSquaredDifference += delta * (value - accumulator.mean);
        }
    }

    function finishStats(accumulator) {
        if (accumulator.count === 0) {
            return {
                count: 0,
                nonfinite: accumulator.nonfinite,
                min: NaN,
                max: NaN,
                mean: NaN,
                std: NaN,
            };
        }
        return {
            count: accumulator.count,
            nonfinite: accumulator.nonfinite,
            min: accumulator.min,
            max: accumulator.max,
            mean: accumulator.mean,
            std: Math.sqrt(accumulator.sumSquaredDifference / accumulator.count),
        };
    }

    function addToHistogram(bins, data, min, max, signal) {
        const span = max - min;
        for (let index = 0; index < data.length; index++) {
            if ((index & 0xffff) === 0 && signal?.aborted) {
                throw new DOMException('The histogram scan was cancelled.', 'AbortError');
            }
            const value = data[index];
            if (!Number.isFinite(value)) continue;
            const normalized = span > 0 ? (value - min) / span : 0.5;
            const bin = Math.max(0, Math.min(bins.length - 1, Math.floor(normalized * bins.length)));
            bins[bin]++;
        }
    }

    async function calculateTensorHistogram(reader, tensorName, channels, options = {}) {
        const { signal, binCount = 64, onProgress = () => {} } = options;
        const descriptor = reader.getTensorInfo(tensorName);
        if (!descriptor?.layout.visualizable) {
            throw new Error(`Tensor "${tensorName}" cannot be scanned as an image sequence.`);
        }

        const frames = descriptor.layout.frames;
        const accumulator = createStatsAccumulator();
        for (let frame = 0; frame < frames; frame++) {
            const { data } = await reader.readFrameChannels(tensorName, frame, channels, { signal });
            addToStats(accumulator, data, signal);
            onProgress({ pass: 'statistics', frame: frame + 1, frames });
            await delay(0);
        }

        const stats = finishStats(accumulator);
        const bins = new Float64Array(binCount);
        if (stats.count > 0) {
            for (let frame = 0; frame < frames; frame++) {
                const { data } = await reader.readFrameChannels(tensorName, frame, channels, { signal });
                addToHistogram(bins, data, stats.min, stats.max, signal);
                onProgress({ pass: 'bins', frame: frame + 1, frames });
                await delay(0);
            }
        }

        return { bins, stats, channels: [...channels], frames };
    }

    function valueToGray(value, min, max) {
        if (Number.isNaN(value)) return 127;
        if (value === Infinity) return 255;
        if (value === -Infinity) return 0;
        if (max <= min) return 127;
        const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
        return Math.round(normalized * 255);
    }

    function fitPreviewDimensions(width, height, maximumDimension = 1024, maximumPixels = 1024 * 1024) {
        const scale = Math.min(
            1,
            maximumDimension / width,
            maximumDimension / height,
            Math.sqrt(maximumPixels / (width * height)),
        );
        return {
            width: Math.max(1, Math.round(width * scale)),
            height: Math.max(1, Math.round(height * scale)),
        };
    }

    function fitDisplayDimensions(width, height, maximumWidth, maximumHeight) {
        const scale = Math.min(maximumWidth / width, maximumHeight / height);
        return {
            width: Math.max(1, Math.floor(width * scale)),
            height: Math.max(1, Math.floor(height * scale)),
        };
    }

    function mapCanvasPoint(offsetX, offsetY, displayWidth, displayHeight, sourceWidth, sourceHeight) {
        return {
            x: Math.max(0, Math.min(sourceWidth - 1, Math.floor(offsetX / displayWidth * sourceWidth))),
            y: Math.max(0, Math.min(sourceHeight - 1, Math.floor(offsetY / displayHeight * sourceHeight))),
        };
    }

    function formatValue(value, digits = 5) {
        if (Number.isNaN(value)) return 'NaN';
        if (value === Infinity) return '+Infinity';
        if (value === -Infinity) return '-Infinity';
        if (!Number.isFinite(value)) return String(value);
        if (value === 0) return Object.is(value, -0) ? '-0' : '0';
        const magnitude = Math.abs(value);
        if (magnitude >= 100000 || magnitude < 0.0001) return value.toExponential(3);
        return Number(value.toFixed(digits)).toString();
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        const units = ['KiB', 'MiB', 'GiB', 'TiB'];
        let value = bytes / 1024;
        let unit = units[0];
        for (let index = 1; index < units.length && value >= 1024; index++) {
            value /= 1024;
            unit = units[index];
        }
        return `${value.toFixed(value >= 10 ? 1 : 2)} ${unit}`;
    }

    function delay(milliseconds) {
        return new Promise((resolve) => setTimeout(resolve, milliseconds));
    }

    function resolveRepresentation(tensorName, descriptor, selectedChannel = 0) {
        if (tensorName === 'value') {
            return {
                kind: 'scalar',
                channels: [selectedChannel],
                cacheKey: `${tensorName}:channel:${selectedChannel}`,
            };
        }
        if ((tensorName === 'position' || tensorName === 'normal') && descriptor.layout.channels >= 3) {
            return {
                kind: 'rgb',
                channels: [0, 1, 2],
                cacheKey: `${tensorName}:rgb`,
            };
        }
        return {
            kind: 'scalar',
            channels: [0],
            cacheKey: `${tensorName}:channel:0`,
        };
    }

    function rgbComponentToByte(value) {
        return valueToGray(value, -1, 1);
    }

    class SafetensorViewerController {
        constructor(document) {
            this.document = document;
            this.reader = null;
            this.file = null;
            this.descriptor = null;
            this.currentTensor = '';
            this.currentChannel = 0;
            this.currentFrame = 0;
            this.currentData = null;
            this.currentStats = null;
            this.currentRange = [-1, 1];
            this.lockedStats = null;
            this.tensorStats = null;
            this.representation = null;
            this.currentHistogram = null;
            this.histogramCache = new Map();
            this.loadGeneration = 0;
            this.frameGeneration = 0;
            this.histogramGeneration = 0;
            this.activeLoadRequest = null;
            this.activeFrameRequest = null;
            this.activeHistogramRequest = null;
            this.isPlaying = false;
            this.prefersReducedMotion = global.matchMedia('(prefers-reduced-motion: reduce)');

            const ids = [
                'viewerApp', 'fileSummary', 'errorContainer', 'previewSummary',
                'emptyState', 'previewCanvas', 'scaleRow', 'scaleMin', 'scaleMax',
                'previousFrame', 'playPause', 'nextFrame', 'frameSlider', 'frameValue',
                'pixelValue', 'statMin', 'statMax', 'statMean', 'statStd', 'histogramCanvas',
                'histogramRegion', 'histogramStatus', 'fileDropZone', 'fileInput',
                'fileMetadata', 'fileName', 'fileSize', 'tensorCount', 'tensorSelect',
                'channelControlGroup', 'channelSelect', 'tensorMetadata', 'tensorDtype',
                'tensorShape', 'tensorLayout', 'tensorBytes', 'tensorMessage',
                'playbackFps', 'playbackSpeed',
            ];
            this.elements = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
            this.bindEvents();
            this.setVisualizationEnabled(false);
        }

        bindEvents() {
            const elements = this.elements;
            elements.fileDropZone.addEventListener('click', () => elements.fileInput.click());
            elements.fileInput.addEventListener('change', () => {
                if (elements.fileInput.files.length > 0) this.loadFile(elements.fileInput.files[0]);
            });
            elements.fileDropZone.addEventListener('dragover', (event) => {
                event.preventDefault();
                elements.fileDropZone.classList.add('dragover');
            });
            elements.fileDropZone.addEventListener('dragleave', () => {
                elements.fileDropZone.classList.remove('dragover');
            });
            elements.fileDropZone.addEventListener('drop', (event) => {
                event.preventDefault();
                elements.fileDropZone.classList.remove('dragover');
                if (event.dataTransfer.files.length > 0) this.loadFile(event.dataTransfer.files[0]);
            });

            elements.tensorSelect.addEventListener('change', () => this.selectTensor(elements.tensorSelect.value));
            elements.channelSelect.addEventListener('change', async () => {
                if (this.currentTensor !== 'value') return;
                this.stopPlayback();
                this.cancelFrameRequest();
                this.cancelHistogramRequest();
                this.currentChannel = Number(elements.channelSelect.value);
                this.representation = resolveRepresentation(this.currentTensor, this.descriptor, this.currentChannel);
                this.lockedStats = null;
                this.tensorStats = null;
                this.currentHistogram = null;
                this.clearHistogram('Preparing all-frame distribution...');
                await this.renderFrame();
                this.ensureHistogram();
            });
            elements.frameSlider.addEventListener('input', () => {
                this.stopPlayback();
                this.currentFrame = Number(elements.frameSlider.value);
                this.renderFrame();
            });
            elements.previousFrame.addEventListener('click', () => this.stepFrame(-1));
            elements.nextFrame.addEventListener('click', () => this.stepFrame(1));
            elements.playPause.addEventListener('click', () => this.togglePlayback());
            elements.previewCanvas.addEventListener('pointermove', (event) => this.inspectPixel(event));
            elements.previewCanvas.addEventListener('pointerleave', () => {
                elements.pixelValue.textContent = 'Move the pointer over the preview';
            });
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) this.stopPlayback();
            });
            this.prefersReducedMotion.addEventListener?.('change', (event) => {
                if (event.matches) this.stopPlayback();
            });
            global.addEventListener('resize', () => {
                if (this.currentData) {
                    this.renderCanvas();
                    this.drawHistogram();
                }
            });
        }

        showError(message) {
            this.elements.errorContainer.textContent = message;
            this.elements.errorContainer.className = 'error-message';
        }

        clearError() {
            this.elements.errorContainer.textContent = '';
            this.elements.errorContainer.className = '';
        }

        setBusy(isBusy) {
            this.elements.viewerApp.setAttribute('aria-busy', String(isBusy));
            this.elements.fileDropZone.disabled = isBusy;
        }

        resetLoadedFile(message) {
            this.reader = null;
            this.file = null;
            this.descriptor = null;
            this.currentTensor = '';
            this.currentChannel = 0;
            this.currentFrame = 0;
            this.currentData = null;
            this.currentStats = null;
            this.lockedStats = null;
            this.tensorStats = null;
            this.representation = null;
            this.currentHistogram = null;
            this.histogramCache.clear();
            this.elements.fileSummary.textContent = 'No file loaded';
            this.elements.fileMetadata.hidden = true;
            this.elements.tensorMetadata.hidden = true;
            this.elements.tensorSelect.disabled = true;
            this.elements.tensorSelect.replaceChildren(new Option('No tensors loaded'));
            this.elements.channelControlGroup.hidden = true;
            this.elements.channelSelect.disabled = true;
            this.elements.channelSelect.replaceChildren(new Option('Channel 0', '0'));
            this.elements.tensorMessage.textContent = 'Load a file to inspect its tensors.';
            this.setVisualizationEnabled(false);
            this.clearPreview(message);
        }

        cancelLoadRequest() {
            this.loadGeneration++;
            this.activeLoadRequest?.abort();
            this.activeLoadRequest = null;
        }

        cancelFrameRequest() {
            this.frameGeneration++;
            this.activeFrameRequest?.abort();
            this.activeFrameRequest = null;
        }

        cancelHistogramRequest() {
            this.histogramGeneration++;
            this.activeHistogramRequest?.abort();
            this.activeHistogramRequest = null;
        }

        cancelAllRequests() {
            this.cancelLoadRequest();
            this.cancelFrameRequest();
            this.cancelHistogramRequest();
        }

        async loadFile(file) {
            this.stopPlayback();
            this.cancelAllRequests();
            this.clearError();
            this.setBusy(true);
            this.resetLoadedFile('Reading safetensors metadata...');
            this.elements.fileSummary.textContent = `Reading ${file.name}...`;

            const controller = new AbortController();
            const generation = this.loadGeneration;
            this.activeLoadRequest = controller;
            try {
                const reader = await global.SafetensorReader.open(file, { signal: controller.signal });
                if (controller.signal.aborted || generation !== this.loadGeneration) return;
                const tensorNames = reader.getTensorNames();
                if (tensorNames.length === 0) {
                    throw new Error('The safetensors file does not contain any tensors.');
                }

                this.reader = reader;
                this.file = file;
                this.elements.fileName.textContent = file.name;
                this.elements.fileSize.textContent = formatBytes(file.size);
                this.elements.tensorCount.textContent = String(tensorNames.length);
                this.elements.fileSummary.textContent = `${file.name} - ${formatBytes(file.size)}`;
                this.elements.fileMetadata.hidden = false;

                this.elements.tensorSelect.replaceChildren();
                for (const name of tensorNames) {
                    const descriptor = reader.getTensorInfo(name);
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = `${name} [${descriptor.shape.join(' x ') || 'scalar'}]`;
                    this.elements.tensorSelect.appendChild(option);
                }
                this.elements.tensorSelect.disabled = false;
                const initialTensor = tensorNames.includes('value') ? 'value' : tensorNames[0];
                this.elements.tensorSelect.value = initialTensor;
                await this.selectTensor(initialTensor);
                if (controller.signal.aborted || generation !== this.loadGeneration) return;
            } catch (error) {
                if (error.name === 'AbortError') return;
                this.resetLoadedFile('Choose another safetensors file to try again.');
                this.showError(error.message || String(error));
            } finally {
                if (this.activeLoadRequest === controller) {
                    this.activeLoadRequest = null;
                    this.setBusy(false);
                }
            }
        }

        async selectTensor(name) {
            if (!this.reader) return;
            this.stopPlayback();
            this.cancelFrameRequest();
            this.cancelHistogramRequest();
            this.currentTensor = name;
            this.currentFrame = 0;
            this.currentChannel = 0;
            this.currentData = null;
            this.currentStats = null;
            this.lockedStats = null;
            this.tensorStats = null;
            this.currentHistogram = null;
            this.descriptor = this.reader.getTensorInfo(name);
            const { descriptor } = this;
            const layout = descriptor.layout;
            this.representation = resolveRepresentation(name, descriptor, 0);

            this.elements.tensorDtype.textContent = descriptor.dtype;
            this.elements.tensorShape.textContent = `[${descriptor.shape.join(', ')}]`;
            this.elements.tensorBytes.textContent = formatBytes(descriptor.byteLength);
            this.elements.tensorLayout.textContent = layout.visualizable
                ? `${layout.frames} frame${layout.frames === 1 ? '' : 's'}, ${layout.width} x ${layout.height}, ${layout.channels} channel${layout.channels === 1 ? '' : 's'}`
                : `Rank ${layout.rank}`;
            this.elements.tensorMetadata.hidden = false;

            const showChannelControl = name === 'value';
            this.elements.channelControlGroup.hidden = !showChannelControl;
            this.elements.channelSelect.replaceChildren();
            const selectableChannels = showChannelControl ? Math.max(1, layout.channels) : 1;
            for (let channel = 0; channel < selectableChannels; channel++) {
                const option = document.createElement('option');
                option.value = String(channel);
                option.textContent = `Channel ${channel}`;
                this.elements.channelSelect.appendChild(option);
            }

            if (!layout.visualizable) {
                this.setVisualizationEnabled(false);
                this.elements.tensorSelect.disabled = false;
                this.elements.tensorMessage.textContent = `Rank-${layout.rank} tensors can be inspected here, but only rank-2, rank-3, and rank-4 tensors can be visualized.`;
                this.clearPreview(`Tensor ${name} is not image-shaped.`);
                return;
            }

            const isRgb = this.representation.kind === 'rgb';
            this.elements.scaleRow.hidden = name !== 'value' && name !== 'depth';
            this.currentRange = name === 'value' ? [-3, 3] : [-1, 1];
            this.elements.tensorMessage.textContent = isRgb
                ? `${name} uses channels 0, 1, and 2 as RGB, matching the Noise page.`
                : 'Rank-3 tensors use [frames, height, width]; rank-4 tensors add a final channel dimension.';
            this.elements.channelSelect.disabled = !showChannelControl || layout.channels <= 1;
            this.elements.frameSlider.min = '0';
            this.elements.frameSlider.max = String(layout.frames - 1);
            this.elements.frameSlider.value = '0';
            this.clearHistogram('Preparing all-frame distribution...');
            this.setVisualizationEnabled(true);
            await this.renderFrame();
            this.ensureHistogram();
        }

        setVisualizationEnabled(enabled) {
            const layout = enabled ? this.descriptor?.layout : null;
            const multipleFrames = Boolean(layout && layout.frames > 1);
            this.elements.frameSlider.disabled = !multipleFrames;
            this.elements.previousFrame.disabled = !multipleFrames;
            this.elements.nextFrame.disabled = !multipleFrames;
            this.elements.playPause.disabled = !multipleFrames;
            this.elements.playbackFps.disabled = !multipleFrames;
            this.elements.playbackSpeed.disabled = !multipleFrames;
            if (!enabled || this.currentTensor !== 'value') this.elements.channelSelect.disabled = true;
        }

        clearPreview(message) {
            this.currentData = null;
            this.currentStats = null;
            this.elements.previewCanvas.classList.add('hidden');
            this.elements.emptyState.classList.remove('hidden');
            this.elements.emptyState.textContent = message;
            this.elements.scaleRow.hidden = true;
            this.elements.previewSummary.textContent = 'No tensor selected';
            this.elements.pixelValue.textContent = 'Move the pointer over the preview';
            for (const id of ['statMin', 'statMax', 'statMean', 'statStd']) {
                this.elements[id].textContent = '-';
            }
            this.clearHistogram('No all-frame distribution available.');
        }

        clearHistogram(message) {
            const canvas = this.elements.histogramCanvas;
            const context = canvas.getContext('2d');
            context.clearRect(0, 0, canvas.width, canvas.height);
            canvas.setAttribute('aria-label', 'No all-frame histogram available');
            this.elements.histogramStatus.textContent = message;
            this.elements.histogramRegion.setAttribute('aria-busy', 'false');
        }

        readRepresentationFrame(reader, tensorName, frameIndex, representation, signal) {
            return reader.readFrameChannels(
                tensorName,
                frameIndex,
                representation.channels,
                { signal },
            );
        }

        async renderFrame() {
            if (!this.reader || !this.descriptor?.layout.visualizable || !this.representation) return false;
            this.cancelFrameRequest();
            const generation = this.frameGeneration;
            const controller = new AbortController();
            this.activeFrameRequest = controller;
            const reader = this.reader;
            const tensorName = this.currentTensor;
            const representation = this.representation;
            const layout = this.descriptor.layout;
            this.currentFrame = Math.max(0, Math.min(layout.frames - 1, this.currentFrame));
            const frameIndex = this.currentFrame;
            this.elements.frameSlider.value = String(frameIndex);
            this.elements.frameValue.textContent = `${frameIndex + 1} / ${layout.frames}`;

            try {
                const current = await this.readRepresentationFrame(
                    reader,
                    tensorName,
                    frameIndex,
                    representation,
                    controller.signal,
                );
                const data = current.data;
                if (
                    controller.signal.aborted
                    || generation !== this.frameGeneration
                    || reader !== this.reader
                    || tensorName !== this.currentTensor
                    || representation.cacheKey !== this.representation?.cacheKey
                ) {
                    return false;
                }

                this.currentData = data;
                this.currentStats = calculateStats(data);
                if (!this.lockedStats) this.lockedStats = this.currentStats;
                this.updateStableRange();
                this.renderCurrentData();
                return true;
            } catch (error) {
                if (error.name === 'AbortError') return false;
                this.stopPlayback();
                this.showError(error.message || String(error));
                return false;
            } finally {
                if (this.activeFrameRequest === controller) this.activeFrameRequest = null;
            }
        }

        updateStableRange() {
            if (this.representation?.kind === 'rgb') {
                this.currentRange = [-1, 1];
                return;
            }
            if (this.currentTensor === 'value') {
                this.currentRange = [-3, 3];
                return;
            }
            const stableStats = this.tensorStats || this.lockedStats;
            if (stableStats) this.currentRange = calculateRange(stableStats);
        }

        renderCurrentData() {
            if (!this.currentData || !this.currentStats) return;
            this.elements.scaleMin.textContent = formatValue(this.currentRange[0]);
            this.elements.scaleMax.textContent = formatValue(this.currentRange[1]);
            this.elements.scaleRow.hidden = this.currentTensor !== 'value' && this.currentTensor !== 'depth';
            this.elements.statMin.textContent = formatValue(this.currentStats.min);
            this.elements.statMax.textContent = formatValue(this.currentStats.max);
            this.elements.statMean.textContent = formatValue(this.currentStats.mean);
            this.elements.statStd.textContent = formatValue(this.currentStats.std);
            this.renderCanvas();
        }

        renderCanvas() {
            const canvas = this.elements.previewCanvas;
            const layout = this.descriptor.layout;
            const fitted = fitPreviewDimensions(layout.width, layout.height);
            canvas.width = fitted.width;
            canvas.height = fitted.height;

            const stageWidth = Math.max(1, canvas.parentElement.clientWidth - 2);
            const maximumWidth = Math.min(560, stageWidth);
            const maximumHeight = global.innerWidth <= 900
                ? Math.min(global.innerHeight * 0.7, 520)
                : Math.max(220, Math.min(520, global.innerHeight - 486));
            const displayed = fitDisplayDimensions(
                layout.width,
                layout.height,
                maximumWidth,
                maximumHeight,
            );
            canvas.style.width = `${displayed.width}px`;
            canvas.style.height = `${displayed.height}px`;
            const context = canvas.getContext('2d');
            const image = context.createImageData(fitted.width, fitted.height);
            const [min, max] = this.currentRange;
            const isRgb = this.representation.kind === 'rgb';
            const componentCount = this.representation.channels.length;

            for (let y = 0; y < fitted.height; y++) {
                const sourceY = Math.min(layout.height - 1, Math.floor((y + 0.5) / fitted.height * layout.height));
                for (let x = 0; x < fitted.width; x++) {
                    const sourceX = Math.min(layout.width - 1, Math.floor((x + 0.5) / fitted.width * layout.width));
                    const sourcePixel = sourceY * layout.width + sourceX;
                    const source = sourcePixel * componentCount;
                    const destination = (y * fitted.width + x) * 4;
                    if (isRgb) {
                        image.data[destination] = rgbComponentToByte(this.currentData[source]);
                        image.data[destination + 1] = rgbComponentToByte(this.currentData[source + 1]);
                        image.data[destination + 2] = rgbComponentToByte(this.currentData[source + 2]);
                    } else {
                        const gray = valueToGray(this.currentData[source], min, max);
                        image.data[destination] = gray;
                        image.data[destination + 1] = gray;
                        image.data[destination + 2] = gray;
                    }
                    image.data[destination + 3] = 255;
                }
            }
            context.putImageData(image, 0, 0);
            canvas.classList.remove('hidden');
            this.elements.emptyState.classList.add('hidden');

            const layoutText = `${layout.width} by ${layout.height}`;
            const frameText = `frame ${this.currentFrame + 1} of ${layout.frames}`;
            const statsText = this.currentStats.count > 0
                ? `minimum ${formatValue(this.currentStats.min)}, maximum ${formatValue(this.currentStats.max)}, mean ${formatValue(this.currentStats.mean)}`
                : 'no finite values';
            const representationText = isRgb
                ? 'RGB channels 0, 1, and 2'
                : `channel ${this.representation.channels[0]}`;
            canvas.setAttribute(
                'aria-label',
                `${this.currentTensor}, ${frameText}, ${representationText}, ${layoutText}; ${statsText}.`,
            );
            this.elements.previewSummary.textContent = `${this.currentTensor} / ${isRgb ? 'RGB' : representationText}`;
        }

        async ensureHistogram() {
            if (!this.reader || !this.representation || !this.descriptor?.layout.visualizable) return;
            const cacheKey = this.representation.cacheKey;
            const cached = this.histogramCache.get(cacheKey);
            if (cached) {
                this.currentHistogram = cached;
                this.tensorStats = cached.stats;
                this.updateStableRange();
                this.renderCurrentData();
                this.drawHistogram();
                this.elements.histogramStatus.textContent = `All ${cached.frames} frames, ${cached.stats.count.toLocaleString()} raw values. Cached.`;
                return;
            }

            this.cancelHistogramRequest();
            const generation = this.histogramGeneration;
            const controller = new AbortController();
            this.activeHistogramRequest = controller;
            const reader = this.reader;
            const tensorName = this.currentTensor;
            const representation = this.representation;
            this.elements.histogramRegion.setAttribute('aria-busy', 'true');
            this.elements.histogramStatus.textContent = 'Scanning raw tensor values...';

            try {
                const result = await calculateTensorHistogram(
                    reader,
                    tensorName,
                    representation.channels,
                    {
                        signal: controller.signal,
                        onProgress: ({ pass, frame, frames }) => {
                            if (
                                controller.signal.aborted
                                || generation !== this.histogramGeneration
                                || reader !== this.reader
                                || representation.cacheKey !== this.representation?.cacheKey
                            ) {
                                return;
                            }
                            const label = pass === 'statistics' ? 'Finding tensor range' : 'Building histogram';
                            this.elements.histogramStatus.textContent = `${label}: frame ${frame} of ${frames}.`;
                        },
                    },
                );
                if (
                    controller.signal.aborted
                    || generation !== this.histogramGeneration
                    || reader !== this.reader
                    || tensorName !== this.currentTensor
                    || representation.cacheKey !== this.representation?.cacheKey
                ) {
                    return;
                }

                this.histogramCache.set(cacheKey, result);
                this.currentHistogram = result;
                this.tensorStats = result.stats;
                this.updateStableRange();
                this.renderCurrentData();
                this.drawHistogram();
                const nonfinite = result.stats.nonfinite > 0
                    ? `, ${result.stats.nonfinite.toLocaleString()} nonfinite values excluded`
                    : '';
                this.elements.histogramStatus.textContent = `All ${result.frames} frames, ${result.stats.count.toLocaleString()} raw values${nonfinite}.`;
            } catch (error) {
                if (error.name === 'AbortError') return;
                if (
                    generation === this.histogramGeneration
                    && reader === this.reader
                    && representation.cacheKey === this.representation?.cacheKey
                ) {
                    this.currentHistogram = null;
                    this.elements.histogramStatus.textContent = `All-frame distribution unavailable: ${error.message || String(error)}`;
                    this.elements.histogramCanvas.setAttribute('aria-label', 'All-frame histogram unavailable');
                }
            } finally {
                if (this.activeHistogramRequest === controller) {
                    this.activeHistogramRequest = null;
                    this.elements.histogramRegion.setAttribute('aria-busy', 'false');
                }
            }
        }

        drawHistogram() {
            const canvas = this.elements.histogramCanvas;
            const cssWidth = Math.max(1, Math.round(canvas.getBoundingClientRect().width || 560));
            const cssHeight = 48;
            const ratio = Math.max(1, Math.min(2, global.devicePixelRatio || 1));
            canvas.width = Math.round(cssWidth * ratio);
            canvas.height = Math.round(cssHeight * ratio);
            const context = canvas.getContext('2d');
            context.scale(ratio, ratio);
            context.fillStyle = '#ffffff';
            context.fillRect(0, 0, cssWidth, cssHeight);

            const histogram = this.currentHistogram;
            if (!histogram || histogram.stats.count === 0) {
                context.fillStyle = '#595959';
                context.font = '12px Segoe UI';
                context.fillText(histogram ? 'No finite values' : 'Distribution is not ready', 10, 22);
                canvas.setAttribute(
                    'aria-label',
                    histogram
                        ? 'All-frame histogram unavailable because the tensor has no finite values.'
                        : 'All-frame histogram is not ready.',
                );
                return;
            }

            const maximum = Math.max(1, ...histogram.bins);
            const barWidth = cssWidth / histogram.bins.length;
            context.fillStyle = '#0a5c6b';
            for (let index = 0; index < histogram.bins.length; index++) {
                const height = histogram.bins[index] / maximum * (cssHeight - 8);
                context.fillRect(index * barWidth, cssHeight - height, Math.max(1, barWidth - 1), height);
            }

            const showIdealGaussian = this.currentTensor === 'value';
            const range = histogram.stats.max - histogram.stats.min;
            if (showIdealGaussian && range > 0) {
                context.strokeStyle = '#c2410c';
                context.lineWidth = 2;
                context.setLineDash([4, 3]);
                context.beginPath();
                for (let index = 0; index < histogram.bins.length; index++) {
                    const position = (index + 0.5) / histogram.bins.length;
                    const x = position * cssWidth;
                    const rawValue = histogram.stats.min + position * range;
                    const gaussian = Math.exp(-rawValue * rawValue / 2);
                    const y = cssHeight - gaussian * (cssHeight - 8) * 0.9;
                    if (index === 0) context.moveTo(x, y);
                    else context.lineTo(x, y);
                }
                context.stroke();
                context.setLineDash([]);
            }

            const referenceText = showIdealGaussian ? ' An ideal standard normal reference curve is overlaid.' : '';
            canvas.setAttribute(
                'aria-label',
                `All-frame histogram of ${histogram.stats.count} raw finite values from ${formatValue(histogram.stats.min)} to ${formatValue(histogram.stats.max)}.${referenceText}`,
            );
        }

        inspectPixel(event) {
            if (!this.currentData || !this.descriptor) return;
            const canvas = this.elements.previewCanvas;
            const rect = canvas.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) return;
            const point = mapCanvasPoint(
                event.clientX - rect.left,
                event.clientY - rect.top,
                rect.width,
                rect.height,
                this.descriptor.layout.width,
                this.descriptor.layout.height,
            );
            const pixel = point.y * this.descriptor.layout.width + point.x;
            if (this.representation.kind === 'rgb') {
                const source = pixel * 3;
                const vector = [
                    formatValue(this.currentData[source], 8),
                    formatValue(this.currentData[source + 1], 8),
                    formatValue(this.currentData[source + 2], 8),
                ];
                this.elements.pixelValue.textContent = `(${point.x}, ${point.y}), RGB / XYZ: [${vector.join(', ')}]`;
            } else {
                const value = this.currentData[pixel];
                this.elements.pixelValue.textContent = `(${point.x}, ${point.y}), channel ${this.representation.channels[0]}: ${formatValue(value, 8)}`;
            }
        }

        async stepFrame(direction) {
            if (!this.descriptor?.layout.visualizable) return;
            this.stopPlayback();
            const frames = this.descriptor.layout.frames;
            this.currentFrame = (this.currentFrame + direction + frames) % frames;
            await this.renderFrame();
        }

        updatePlaybackButton() {
            this.elements.playPause.classList.toggle('is-playing', this.isPlaying);
            this.elements.playPause.setAttribute('aria-label', this.isPlaying ? 'Pause frame playback' : 'Play frames');
            this.elements.playPause.setAttribute('aria-pressed', String(this.isPlaying));
        }

        togglePlayback() {
            if (this.isPlaying) {
                this.stopPlayback();
            } else {
                this.startPlayback();
            }
        }

        async startPlayback() {
            if (this.isPlaying || !this.descriptor || this.descriptor.layout.frames <= 1) return;
            this.isPlaying = true;
            this.updatePlaybackButton();

            while (this.isPlaying) {
                const fps = Number(this.elements.playbackFps.value);
                const speed = Number(this.elements.playbackSpeed.value);
                await delay(1000 / Math.max(0.01, fps * speed));
                if (!this.isPlaying) break;
                this.currentFrame = (this.currentFrame + 1) % this.descriptor.layout.frames;
                const rendered = await this.renderFrame();
                if (!rendered && this.isPlaying) {
                    this.stopPlayback();
                    break;
                }
            }
        }

        stopPlayback() {
            if (!this.isPlaying) return;
            this.isPlaying = false;
            this.updatePlaybackButton();
        }
    }

    global.SafetensorViewer = Object.freeze({
        Controller: SafetensorViewerController,
        calculateStats,
        calculateRange,
        buildHistogram,
        calculateTensorHistogram,
        resolveRepresentation,
        rgbComponentToByte,
        valueToGray,
        fitPreviewDimensions,
        fitDisplayDimensions,
        mapCanvasPoint,
        formatValue,
        formatBytes,
    });

    document.addEventListener('DOMContentLoaded', () => {
        if (document.getElementById('viewerApp')) {
            global.safetensorViewer = new SafetensorViewerController(document);
        }
    });
})(window);
