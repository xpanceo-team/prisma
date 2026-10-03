# Installation

PRISMA requires Python 3.11 or later. The Linux x86_64 profiles below use
Python 3.11 or 3.12 and PyTorch 2.6.0. Use one profile in a fresh virtual
environment, from the repository root.

## Install with pip

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

### CPU

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
python -m pip install -e .
```

### NVIDIA GPU (CUDA 12.4)

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install -e .
```

The NVIDIA driver must support the CUDA runtime bundled with PyTorch.
A local CUDA toolkit is not required for these binary wheels.

## Install with uv

Create and activate an environment with Python 3.12:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### CPU

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
uv pip install -e .
```

### NVIDIA GPU (CUDA 12.4)

```bash
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --only-binary=:all: torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
uv pip install -e .
```

Use `uv run --no-sync prisma ...` with this environment if you use `uv run`.
The commands above install dependencies through `uv pip`; automatic project
synchronization by `uv run` can resolve a different PyTorch build.

## Verify the installation

```bash
python -c "import prisma, torch, torch_scatter, torch_sparse; print('PRISMA:', prisma.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
prisma --help
python -m pip check
```

For an environment created with `uv`, use `uv pip check` instead of
`python -m pip check`.

Use `python -m pip install .` or `uv pip install .` in the final installation
step for a non-editable installation.

## Other platforms and PyTorch versions

Select PyTorch from the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
and matching extension wheels from the [PyTorch Geometric wheel index](https://data.pyg.org/whl/).
The extension wheels must match the Python version, OS, architecture, PyTorch
version, and CPU/CUDA build. Keep `--only-binary=:all:` when installing the
extensions so unsupported combinations fail without attempting compilation.

The PyTorch 2.6.0 extension pages above do not provide Python 3.13 wheels for
`torch-scatter` and `torch-sparse`. Use Python 3.11 or 3.12 for these profiles.

## Optional dependencies

Install the dependencies required by the selected workflow.

ML screening and validation:

```bash
python -m pip install -e ".[validation]"
```

VASP workflow:

```bash
python -m pip install -e ".[dft]"
```

The PET model backbone uses the metatensor runtime:

```bash
python -m pip install -e ".[pet]"
```

VASP itself, its pseudopotentials, and a configured SLURM environment are
external requirements described in [DFT/VASP validation](dft.md).

## Troubleshooting

### A PyG extension is being built from source

A direct `pip install -e .` or `uv pip install -e .` in an empty environment
can fail with `ModuleNotFoundError: No module named 'torch'` while building
`torch-scatter`. PyTorch is unavailable inside its isolated build environment.

Use one of the profiles above to install `torch-scatter` and `torch-sparse`
from the wheel page matching the installed PyTorch and CUDA build. Confirm the build
with:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### CUDA is unavailable

Compare `nvidia-smi` with the installed PyTorch build:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The CUDA version reported by PyTorch identifies the runtime used by the wheel;
it does not need to be identical to the maximum CUDA version displayed by
`nvidia-smi`, but the NVIDIA driver must support it.
