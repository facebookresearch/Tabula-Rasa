# Tabula Rasa

Monte Carlo estimation of Gaussian noise with unit variance and controlled
spatio-temporal correlation.

Ordinary procedural noise gives you a value per point. What it does not give you
is control over how that value correlates with the values at nearby points and at
nearby moments in time, while still guaranteeing that every sample is unit
variance Gaussian. Tabula Rasa estimates such a field with a Monte Carlo
estimator, prefilters it so the noise resolution matches the pixel footprint, and
corrects the variance that filtering would otherwise destroy.

The resulting noise is intended as input to diffusion-based video generation, and
is also useful for stylization, animated texture synthesis, and steganography.

Two implementations are included:

| | |
|---|---|
| [`python/`](python) | Reference implementation in JAX. Every sampler, filter, map, field, and estimator variant. Renders to video, EXR, and safetensors. |
| [`webgpu/`](webgpu) | Real-time browser demo. The mesh path only, running entirely on the GPU. No build step. |

## Quick start

### Python renderer

```bash
cd python
conda env create -f environment.yml
conda activate tabula-rasa
pip install -e .
python examples/run_experiments.py suite
```

The Conda environment installs CUDA 12 JAX by default for NVIDIA GPU rendering.
For a CPU-only installation, use `pip install -r requirements.txt` in a Python
3.12 environment with `ffmpeg` on `PATH`. Full device-selection details are in
[`python/SETUP.md`](python/SETUP.md).

### WebGPU demo

```bash
cd webgpu
python -m http.server 8000
```

Open <http://localhost:8000/index.html> in a WebGPU-capable browser. It starts on
a cube; drag in an `.obj`, `.gltf`, or `.glb` to use your own mesh. See
[`webgpu/README.md`](webgpu/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please also read our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Third-party code

Third-party material and its licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with license texts in
[`third_party_licenses/`](third_party_licenses).

## License

The majority of Tabula Rasa is licensed under [CC-BY-NC](LICENSE), however
portions of the project are available under separate license terms:

- Random123 (BSD-3-Clause license): [Third-party notice](THIRD_PARTY_NOTICES.md#1-threefry2x32--random123-and-jax)
- JAX (Apache-2.0 license): [Third-party notice](THIRD_PARTY_NOTICES.md#1-threefry2x32--random123-and-jax)
- PCG (Apache-2.0 license): [Third-party notice](THIRD_PARTY_NOTICES.md#2-pcg-hash-function--apache-20-m-e-oneill)
- pbrt-v2 (BSD-2-Clause license): [Third-party notice](THIRD_PARTY_NOTICES.md#3-larcher-pillichshammer--sobol-radical-inverses--bsd-2-clause-pbrt-v2)
