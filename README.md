# PRISMA

PRISMA is a modular closed-loop framework for crystal generation, machine-learning screening, and DFT validation of AI-designed materials.

It integrates diffusion-based generative modeling, fast ML screening, and automated first-principles validation into a reusable workflow for computational materials discovery.

The Python API is organized into two packages:

- `prisma`: diffusion-based crystal generation, training, sampling, ML screening, and validation utilities.
- `vaspoperator`: automated VASP/DFT validation backend for SLURM-managed HPC environments.

## Workflow

PRISMA follows an iterative generation-to-validation loop:

1. Train or initialize an unconditional diffusion model on broad crystal datasets to explore stable and metastable structure space.
2. Fine-tune the model on property-labeled datasets for conditional generation of targeted materials.
3. Generate candidate crystals and filter them with fast ML-based screening, including stability, novelty, uniqueness, and target-property predictors.
4. Validate selected candidates with first-principles calculations using VASP.
5. Feed DFT-confirmed structures and properties back into the training dataset for subsequent conditional fine-tuning cycles.

For optical-material discovery, PRISMA combines diffusion generation, MatterSim/MACE-based screening, optical-property regressors, and DFT validation of structural stability and dielectric response.

## Repository Structure

```text
prisma/
|-- config/                    # VASP, SLURM, workflow, and plotting YAML configs
|-- notebooks/                 # Generation and validation examples
|-- scripts/                   # Training, fine-tuning, and development scripts
|-- src/
|   |-- prisma/                 # Generation, training, configs, pipelines, and models
|   `-- vaspoperator/          # Automated DFT/VASP/SLURM validation backend
|-- tests/                     # Unit tests
|-- env.yml                    # Conda environment reference
`-- pyproject.toml             # Python package metadata and dependencies
```

## Installation

PRISMA requires Python 3.11 or later. Install PyTorch and the compiled PyTorch
Geometric extensions for the target platform before installing PRISMA. The
following configuration is validated for CUDA 12.4:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

Install PRISMA and its remaining dependencies from the repository root:

```bash
python -m pip install .
```

Optional validation dependencies, including MatterSim and MACE, can be
installed with:

```bash
python -m pip install ".[validation]"
```

PET models require their metatensor runtime:

```bash
python -m pip install ".[pet]"
```

## Optional DFT Install

The automated VASP backend has additional dependencies:

```bash
pip install -e ".[dft]"
```

On older HPC systems, `polars` may require the CPU-compatible build:

```bash
pip install polars-lts-cpu
```

The DFT backend requires the following external components:

- A licensed VASP installation available on the target HPC system.
- A working SLURM configuration and cluster-specific module/runtime setup.
- VASP pseudopotentials accessible to `pymatgen`.
- Site-specific edits to `config/server.yaml`, `config/vasp.yaml`, `config/steps.yaml`, and `config/sumo.yaml`.

For POTCAR generation, point `pymatgen` to your pseudopotential library:

```bash
export PMG_VASP_PSP_DIR="/path/to/potcars"
```

## Generation Usage

Generation examples are available in `notebooks/generation.ipynb`.

Load a MatterGen-compatible pipeline and generate structures:

```python
import torch

from prisma.pipelines.mattergen import MatterGenPipeline


pipeline = MatterGenPipeline.from_pretrained(
    "xpanceo-team/mattergen-formula-e_above_hull",
    use_safetensors=False,
)

device = "cuda"
batch_size = 2
random_seed = 42

condition = {
    "e_above_hull": torch.tensor([0.0430, 0.0430]),
    "formula": ["Na3 Mn1 Co1 Ni1 O6", "Na3 Mn1 Co1 Ni1 O6"],
}

generator = torch.Generator(device=device).manual_seed(random_seed)

structures = pipeline(
    batch_size=batch_size,
    condition=condition,
    guidance_scale=3.0,
    device=device,
    generator=generator,
)
```

## Training

### Preparing a dataset

Input tables contain a structure column alongside condition and metadata
columns. PRISMA accepts Parquet, CSV, JSON, JSONL, datasets saved with
`save_to_disk()`, and Hub datasets. Structure values may be pymatgen JSON, CIF,
or POSCAR text.

Prepare and save an Arrow-backed `DatasetDict`:

```bash
prisma data prepare data/raw.parquet \
    --output data/materials
```

Use explicit column names and structure format when needed:

```bash
prisma data prepare data/raw.csv \
    --structure-column crystal \
    --structure-format cif \
    --id-column source_id \
    --output data/materials
```

Existing splits are preserved. A split column can define them:

```bash
prisma data prepare data/raw.parquet \
    --split-column split \
    --output data/materials
```

Alternatively, create a reproducible valid split from train data:

```bash
prisma data prepare data/raw.parquet \
    --validation-fraction 0.1 \
    --seed 42 \
    --output data/materials
```

Inspect the prepared artifact:

```bash
prisma data inspect data/materials
```

A Hub dataset can also be normalized locally:

```bash
prisma data prepare your-org/materials \
    --source-type hub \
    --revision main \
    --output data/materials
```

Publish a prepared artifact with:

```bash
prisma data push data/materials your-org/materials --private
```

### Training a model

Create a training configuration. `dataset_name_or_path` accepts either a local
prepared dataset or a Hugging Face dataset name:

```yaml
name: bandgap-model
dataset_name_or_path: data/materials

model:
  backbone: gemnet
  pretrained_model_name_or_path: xpanceo-team/mattergen-base

conditions:
  bandgap:
    type: scalar

data:
  max_num_atoms: 20
  validation_fraction: 0.1
  split_seed: 42

training:
  max_epochs: 800
  batch_size: 32
  gradient_accumulation: 2
  learning_rate: 1.0e-4

output_dir: runs/bandgap-model
```

Start training:

```bash
prisma train training.yaml
```

Inspect the resolved configuration without loading weights or starting
training:

```bash
prisma train training.yaml --print-config
```

Override a configuration value for an individual run:

```bash
prisma train training.yaml --set training.max_epochs=100
```

Set `model.backbone` to `gemnet`, `equiformer_v2`, or `pet`. Omitting
`pretrained_model_name_or_path` trains a new model, and omitting `conditions`
trains an unconditional model. Each backbone selects its established
architecture and training defaults; values under `model.config` and `training`
override them. Complete configurations are available in `examples/training/`.

Models stored on the Hugging Face Hub are PRISMA pipelines and are selected
with `pretrained_model_name_or_path`. To resume an interrupted Lightning run,
use its training checkpoint separately:

```yaml
training:
  resume_from_checkpoint: runs/bandgap-model/last.ckpt
```

Weights & Biases logging is enabled explicitly:

```yaml
logging:
  wandb:
    project: prisma
```

After training, a checkpoint can be converted into a generation pipeline:

```python
from prisma.pipelines.mattergen.pipeline_mattergen import MatterGenPipeline
from prisma.training.module import TrainingModule


save_directory = "runs/example/"
ckpt_path = save_directory + "epoch=26-step=43092-best=loss.ckpt"

module = TrainingModule.load_from_checkpoint(ckpt_path, map_location="cpu")
pipeline = MatterGenPipeline(
    gnn=module.gnn,
    condition_encoder=module.cond_encoder,
    score_model=module.score_model,
    atomic_numbers_scheduler=module.atomic_numbers_scheduler,
    frac_coords_scheduler=module.frac_coords_scheduler,
    cell_scheduler=module.cell_scheduler,
)

pipeline.push_to_hub(
    "your-org/mattergen-bandgap",
    private=True,
    safe_serialization=False,
)
```

## DFT/VASP Validation

`vaspoperator` orchestrates high-throughput DFT validation on SLURM-managed clusters. It runs a dependency-aware workflow and propagates successful outputs between stages:

1. `REL`: full structure relaxation.
2. `SCF`: self-consistent electronic ground-state calculation.
3. `IPA`: independent-particle approximation dielectric calculation.
4. `DOS`: density-of-states calculation.
5. `BANDS`: band-structure calculation.

The workflow is configured by the YAML files in `config/`:

| File | Purpose |
| --- | --- |
| `config/vasp.yaml` | INCAR tags and common VASP parameters for each stage. |
| `config/steps.yaml` | Step ordering and dependency hierarchy. |
| `config/server.yaml` | SLURM resources, partitions, and execution parameters. |
| `config/sumo.yaml` | Plotting and band-path settings for `sumo` outputs. |

Run a single-structure validation from a POSCAR-like file:

```bash
python src/vaspoperator/pipelines/run_single_structure.py \
    --structure_path="data/raw/POSCAR_sample" \
    --material_id="Ag2O_test"
```

Run a batch workflow from a Parquet dataset containing JSON-serialized pymatgen structures:

```bash
python src/vaspoperator/pipelines/run_multi.py \
    --dataset_path="data/raw/test_structures.parquet" \
    --structure_column="structure_mattersim" \
    --id_column="material_id" \
    --num_threads=5 \
    --limit=3
```

The DFT backend writes VASP work directories under `data/vasp/` and result tables/plots under `data/results/` or `data/results_batch/`, depending on the entry point.

## Package Modules

- `prisma.models` and `prisma.schedulers` provide neural-network backbones, output heads, condition encoders, and diffusion schedulers for crystal generation.
- `prisma.pipelines` exposes operational generation interfaces, including MatterGen-compatible pipelines.
- `prisma.training` contains training modules and optimization workflows.
- `prisma.evaluation` and `prisma.validation` contain generated-structure assessment and ML validation utilities.
- `vaspoperator.calculation`, `vaspoperator.input`, `vaspoperator.slurm`, and `vaspoperator.pipelines` provide automated VASP input generation, SLURM scheduling, monitoring, and result processing.

## License

PRISMA is licensed under the MIT License.
