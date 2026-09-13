# Setup

The Python renderer lives in this directory. The WebGPU demo, in `../webgpu`,
needs no Python at all — see [../webgpu/README.md](../webgpu/README.md).

## Conda

```bash
conda env create -f environment.yml
conda activate tabula-rasa
pip install -e .
```

`conda env create` can be slow to solve. If you have `mamba` or `micromamba`,
`mamba env create -f environment.yml` resolves the same file much faster.

`environment.yml` installs Python and `ffmpeg` from conda-forge, loads the
shared Python packages from `requirements.txt`, and adds CUDA 12 support to JAX
for NVIDIA GPU rendering.

## pip only

Requires Python 3.12 and `ffmpeg` on `PATH`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`ffmpeg` is not a Python package — install it with your system package manager
(`apt install ffmpeg`, `brew install ffmpeg`) or via conda. It is needed only for
video output, which `mediapy` invokes as a subprocess.

## JAX device selection

The Conda route installs CUDA 12 JAX by default. Verify that the NVIDIA GPU is
visible:

```bash
python -c "import jax; print(jax.devices())"
```

To run an individual command on CPU without changing the environment:

```bash
JAX_PLATFORMS=cpu python examples/run_experiments.py suite
```

For a CPU-only installation, use the pip-only route above. If the Conda
environment is already installed, remove its CUDA plugin to make it CPU-only:

```bash
python -m pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt
```

The NVIDIA support libraries may remain installed but are unused after the
plugin is removed. Restore GPU support with:

```bash
python -m pip install -U "jax[cuda12]"
```

## Running

```bash
python examples/run_experiments.py --help
python examples/run_experiments.py suite
```

Input is read from `./in` and output written to `./out`, both relative to the
directory you run from.

## Notebook

`examples/experiments.ipynb` needs a kernel registered for the environment:

```bash
python -m ipykernel install --user --name tabula-rasa --display-name "Tabula Rasa"
```

## Verifying the install

```bash
python -c "import tabula_rasa; print(sorted(tabula_rasa.__all__))"
python tests/test_two_pass_independence.py
```
