# Installation

PRISMA supports Python 3.11 and later. Use a virtual environment managed by
Conda, `venv`, or another environment manager.

## PyTorch and PyTorch Geometric

Install PyTorch for the compute platform first. Select the command for the
machine from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

PRISMA uses the compiled `torch-scatter` and `torch-sparse` extensions. Install
binary wheels built for the installed PyTorch and CUDA versions from the
[PyTorch Geometric wheel index](https://data.pyg.org/whl/). For example, this
is the configuration validated with PRISMA on CUDA 12.4:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

The exact PyTorch version above describes the validated environment; it is not
a package requirement. For another PyTorch or CUDA release, select the
corresponding wheel page. Installing an unmatched extension commonly causes
an attempted source build or an import-time undefined-symbol error.

## Install PRISMA

From the repository root:

```bash
python -m pip install -e .
```

Use a non-editable installation when the source tree will not be modified:

```bash
python -m pip install .
```

Verify the environment:

```bash
python -c "import prisma, torch, torch_scatter, torch_sparse; print('PRISMA:', prisma.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pip check
```

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

Stop the installation and install `torch-scatter` and `torch-sparse` from the
wheel page matching the installed PyTorch and CUDA build. Confirm the build
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
