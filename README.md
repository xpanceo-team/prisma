# PRISMA

PRISMA is a modular closed-loop framework for crystal generation,
machine-learning screening, and DFT validation of AI-designed materials.

It integrates diffusion-based generative modeling, fast ML screening, and
automated first-principles validation into a unified workflow for computational
materials discovery. The framework supports foundation-model training,
property-conditioned fine-tuning, crystal generation, candidate screening, and
VASP validation on SLURM-managed systems.

## Workflow

PRISMA supports iterative generation and validation campaigns:

1. Train or initialize a diffusion model for crystalline materials.
2. Fine-tune the model on property-labeled structures for conditional generation.
3. Generate candidates and screen them for stability, novelty, uniqueness, and target properties.
4. Validate selected structures with first-principles calculations.
5. Incorporate validated structures and properties into subsequent training cycles.

## Installation

PRISMA requires Python 3.11 or later. PyTorch and the compiled PyTorch
Geometric extensions must match the CUDA runtime available on the target
system. Install those components first, then install PRISMA from the repository
root:

```bash
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import prisma, torch; print(prisma.__version__); print(torch.cuda.is_available())"
python -m pip check
```

Platform-specific PyTorch instructions and optional dependencies are covered in
the [installation guide](docs/installation.md).

## Fine-tuning on a custom dataset

The quickest path to a property-conditioned model is to fine-tune the
pretrained MatterGen base model on a dataset of structures and target
properties. This section follows that workflow from a local table to generation
with the resulting model.

### 1. Prepare the dataset

Create a Parquet table with one row per material. It must contain a `structure`
column and one numeric column for each conditioning property:

| material_id | structure | bandgap_hse | shg | dn |
| --- | --- | ---: | ---: | ---: |
| material-001 | pymatgen JSON or CIF text | 3.4 | 1.8 | 0.12 |
| material-002 | pymatgen JSON or CIF text | 4.1 | 2.3 | 0.08 |

Normalize the table and create a reproducible validation split:

```bash
prisma data prepare materials.parquet \
    --validation-fraction 0.1 \
    --seed 42 \
    --output data/materials
```

Check the resulting splits, row counts, and columns:

```bash
prisma data inspect data/materials
```

PRISMA also accepts CSV, JSON, JSONL, and existing Hugging Face datasets. See
[Datasets](docs/datasets.md) for the schema and supported split layouts.

### 2. Configure conditional fine-tuning

Copy [the local conditional example](examples/training/local_conditional.yaml)
and replace the condition names with the property columns in your dataset:

```bash
cp examples/training/local_conditional.yaml local_conditional.yaml
```

```yaml
name: my-materials
dataset_name_or_path: data/materials

model:
  backbone: gemnet
  pretrained_model_name_or_path: xpanceo-team/mattergen-base

conditions:
  bandgap_hse:
    type: scalar
  shg:
    type: scalar
  dn:
    type: scalar

data:
  max_num_atoms: 20

training:
  max_epochs: 800
  batch_size: 32
  gradient_accumulation: 2
  learning_rate: 1.0e-4

output_dir: runs/my-materials
```

Condition names must exactly match dataset columns. Scalar conditions are
standardized from the training split and encoded with the model's scalar
condition adapters. Structures containing more than `max_num_atoms` are
filtered before the train/validation split is used.

### 3. Check and start training

Inspect the resolved configuration without loading model weights:

```bash
prisma train local_conditional.yaml --print-config
```

Start training:

```bash
prisma train local_conditional.yaml
```

Before creating the run, PRISMA executes one representative training step with
the configured model, device, precision, and optimizer. If it runs out of GPU
memory, lower `training.batch_size` and increase
`training.gradient_accumulation` to preserve the effective batch size.

New condition encoder and adapter weights are expected when a base checkpoint
is fine-tuned on properties it did not previously contain. The remaining model
weights are loaded from `xpanceo-team/mattergen-base`.

Resume an interrupted run by adding its Lightning checkpoint to the same
configuration:

```yaml
training:
  resume_from_checkpoint: runs/my-materials/epoch=07-step=0128-last.ckpt
```

Use the most recent `*-last.ckpt` file in the run directory. Its epoch and step
numbers depend on the run.

See [Training](docs/training.md) for checkpointing, experiment logging,
backbone selection, and configuration details.

### 4. Export and use the trained model

Select the checkpoint to use for generation, reconstruct its components, and
save them as a reusable pipeline:

```python
from pathlib import Path

from prisma.pipelines.mattergen import MatterGenPipeline
from prisma.training.module import TrainingModule


checkpoint = next(Path("runs/my-materials").glob("*-best=loss.ckpt"))
module = TrainingModule.load_from_checkpoint(
    checkpoint,
    map_location="cpu",
)
pipeline = MatterGenPipeline(
    gnn=module.gnn,
    condition_encoder=module.cond_encoder,
    score_model=module.score_model,
    atomic_numbers_scheduler=module.atomic_numbers_scheduler,
    frac_coords_scheduler=module.frac_coords_scheduler,
    cell_scheduler=module.cell_scheduler,
)
pipeline.save_pretrained(
    "runs/my-materials/pipeline",
    safe_serialization=False,
)
```

Generate structures with the properties used during fine-tuning:

```python
import torch

from prisma.pipelines.mattergen import MatterGenPipeline


pipeline = MatterGenPipeline.from_pretrained(
    "runs/my-materials/pipeline",
    use_safetensors=False,
)

device = "cuda"
condition = {
    "bandgap_hse": torch.tensor([3.5, 4.0]),
    "shg": torch.tensor([2.0, 2.5]),
    "dn": torch.tensor([0.10, 0.15]),
}

structures = pipeline(
    batch_size=2,
    condition=condition,
    guidance_scale=3.0,
    device=device,
    generator=torch.Generator(device=device).manual_seed(42),
)
```

The condition values supplied for generation are in the original dataset
units. See [Generation](docs/generation.md) for loading published pipelines and
publishing the exported model.

## Documentation

- [Installation](docs/installation.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [Generation and checkpoint export](docs/generation.md)
- [DFT/VASP validation](docs/dft.md)

## License

PRISMA is licensed under the MIT License.
