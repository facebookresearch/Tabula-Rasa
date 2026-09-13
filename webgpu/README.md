# Tabula Rasa — WebGPU demo

A real-time WebGPU port of the Python renderer in [`../python`](../python). It
rasterizes a mesh into a G-buffer, then estimates correlated Gaussian noise over
that geometry with the count-unique estimator, entirely on the GPU.

No build step and no Python are needed — it is plain ES2020 and WGSL.

## Running

Serve this directory over HTTP (module and shader fetches will not work from
`file://`):

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/index.html>. The companion file viewer is at
<http://localhost:8000/safetensor-viewer.html>; both pages link to each other.

The renderer opens on a unit cube. Drag and drop an `.obj`, `.gltf`, or `.glb`
file onto the page to load your own mesh; glTF files with skinning and animation
are animated on the GPU.

## Browser requirements

WebGPU is required: Chrome or Edge 113+, or Safari 17+. On Linux, Chrome may
need `--enable-unsafe-webgpu`. Firefox needs a build with WebGPU enabled.

## Controls

| Control | Description |
|---|---|
| Seed | Seed for the noise field |
| Field Type | White, Adaptive, or Blended Adaptive |
| RNG Type | Threefry (matches JAX) or PCG (faster) |
| Estimator Mode | Biased (N samples) or unbiased (2N independent samples) |
| Output Mode | Final noise, or a G-buffer channel for inspection |
| Histogram Bins | Bin count for the estimator's weighted histogram (1–256) |
| Time | Animation time, which drives camera orbit and mesh animation |
| Width / Height | Render resolution |
| Sample Count | Monte Carlo samples per pixel (capped at 256) |
| Field Resolution | Noise field grid resolution |
| Mesh Yaw / Pitch / Yaw Speed / Scale / FOV | Camera orbit and framing |
| Center Background Between Field Cells | Keep the background cube away from numerically unstable field boundaries |
| Export | Write the noise, or every channel, as `.safetensors` |
| SPACE | Toggle time animation |

Numeric readouts are click-to-edit as well as draggable. Every control has a
label, is reachable by keyboard, and shows a visible focus ring.

## Inspecting exports

Open `safetensor-viewer.html`, or use the **Safetensors Viewer** button in the
Noise page header. The matching header button returns to the generator. The
viewer reads local files in the browser and does not upload them. It supports
numeric rank-2 `[H, W]`, rank-3 `[T, H, W]`, and rank-4 `[T, H, W, C]`
tensors. Other ranks remain visible in the metadata panel but are not rendered.

The `value` tensor is shown one channel at a time with a stable `[-3, 3]` range.
Channel selection is hidden for other tensors. Exported `position` and `normal`
tensors use their first three components as RGB, matching the Noise page's
`value * 0.5 + 0.5` visualization; other tensors use channel zero.

Playback updates the preview and current-frame statistics without changing its
exposure. The histogram is a separate, static distribution of raw values from
every frame of the active tensor representation. For `value`, an orange dashed
standard-normal curve provides the same ideal reference as the Noise page. The
viewer scans frames incrementally in bounded chunks and caches the compact
all-frame result, rather than loading the complete file or tensor into memory.

Supported dtypes are F16, BF16, F32, F64, signed and unsigned 8/16/32/64-bit
integers, and BOOL. Values from 64-bit integer tensors outside JavaScript's
exact integer range are approximate when displayed.

## Pipeline

Four passes per frame:

1. **Raster** (`shaders/rasterizer_mesh.wgsl`) — draw the background cube and the
   mesh into two `rgba32float` targets: object-space position with depth, and
   `(footprint, objectId, normal.x, normal.y)`. The pixel footprint comes from
   `dpdx`/`dpdy` of the object-space position, so it accounts for perspective and
   surface orientation without any analytic Jacobian.
2. **Noise** (`shaders/noise.wgsl`) — one workgroup per pixel, one thread per
   sample. Each thread jitters within the pixel, reads the G-buffer bilinearly to
   get a 3D position, hashes it to a field cell, and accumulates a 16-channel
   Gaussian value. A weighted histogram in workgroup memory, kept per LOD level to
   avoid collisions between levels, sketches the weight distribution; the result is
   divided by `sqrt(sum(w_i^2))` to restore unit variance.
3. **Tonemap** (`shaders/tonemap.wgsl`) — map raw values into a displayable range
   for the selected output mode. Raw values stay in the noise textures, so export
   is unaffected.
4. **Blit** (`shaders/render.wgsl`) — draw to the canvas.

Noise is generated in 16 channels across four `rgba32float` textures.

## Files

| File | Role |
|---|---|
| `index.html` | Page, UI controls, and styling |
| `main.js` | Renderer, pipelines, export, histogram widget, UI wiring |
| `safetensor-viewer.html` | Accessible viewer page and responsive layout |
| `safetensor-reader.js` | Validated incremental safetensors parser and frame reader |
| `safetensor-viewer.js` | Viewer state, playback, statistics, and canvas rendering |
| `gltf-loader.js` | glTF/GLB parsing, skinning, animation sampling |
| `obj-loader.js` | Wavefront OBJ parsing |
| `wgsl-preprocessor.js` | `#include` and `#pragma once` for WGSL |
| `shaders/common.wgsl` | Structs shared between the shaders |
| `shaders/rasterizer_mesh.wgsl` | Mesh G-buffer, static and skinned |
| `shaders/noise.wgsl` | The count-unique estimator |
| `shaders/tonemap.wgsl` | Output-mode tonemapping |
| `shaders/render.wgsl` | Canvas blit |

`Params` is declared identically in `common.wgsl`, `noise.wgsl`, and
`tonemap.wgsl`, and packed by hand in `main.js`. The four copies must stay in
sync; `tests/mesh-orientation-test.html` asserts that they do.

## Tests

Open these in a browser; the tab title becomes `PASS` or `FAIL`.

- `tests/gltf-loader-test.html` — glTF parsing, joint ordering, and node
  transforms, against synthetic GLB files built in memory.
- `tests/mesh-orientation-test.html` — camera and mesh conventions, the
  `Params` layout, and generator-page UI invariants, checked from source.
- `tests/safetensor-viewer-test.html` — safetensors validation, dtype and layout
  decoding, bounded reads, viewer helpers, accessibility, and navigation. Its
  fixtures are generated entirely in memory.

## Relationship to the Python renderer

Same algorithm, different tradeoffs. The Python renderer is the reference
implementation and supports every map, field, and estimator variant from the
paper; this demo keeps the mesh path only, and runs in real time. Threefry is
included so its output can be compared against JAX directly.

## License

CC-BY-NC 4.0, as found in the `LICENSE` file at the repository root.
