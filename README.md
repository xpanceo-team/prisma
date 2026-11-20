# PRISMA

PRISMA is a framework for training diffusion models and using them to generate
crystal structures. It includes tools to screen candidate structures for
stability and target properties and to validate selected candidates with VASP.

## Installation

The installation guide covers Linux x86_64 with Python 3.11 or 3.12.
Choose the profile for your machine:

- [NVIDIA GPU (CUDA 12.4)](docs/installation.md#nvidia-gpu-cuda-124) for the training example below.
- [CPU](docs/installation.md#cpu) for machines without an NVIDIA GPU.

After installation, run `prisma --help` to see the available commands.

## Generate a crystal

Load a pretrained MatterGen model, generate one structure with the requested
composition and energy above hull, and save it to `candidate.cif`.

```python
import torch
from prisma.pipelines.mattergen import MatterGenPipeline

pipeline = MatterGenPipeline.from_pretrained(
    "xpanceo-team/mattergen-formula-e_above_hull",
    use_safetensors=False,
)

structures = pipeline(
    batch_size=1,
    condition={
        "formula": ["Na3 Mn1 Co1 Ni1 O6"],
        "e_above_hull": torch.tensor([0.043]),
    },
    guidance_scale=3.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
structures[0].to(filename="candidate.cif")
```

## Fine-tuning on a custom dataset

To fine-tune MatterGen on your own data, start with a table of crystal structures
and their target properties. The example below uses two property columns
(`dn` and `k_1064`). Use the same column names in your training configuration.

### 1. Prepare the dataset

Create a Parquet table with one row per material. It must contain a `structure`
column and one numeric column for each conditioning property:

| material_id | structure | dn | k_1064 |
| --- | --- | ---: | ---: |
| material-001 | pymatgen JSON or CIF text | 0.12 | 0.001 |
| material-002 | pymatgen JSON or CIF text | 0.08 | 0.002 |

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

Copy [the example configuration](examples/training/local_conditional.yaml):

```bash
cp examples/training/local_conditional.yaml local_conditional.yaml
```

Set the dataset path and conditioning properties in your copy:

```yaml
dataset_name_or_path: data/materials

conditions:
  dn:
    type: scalar
  k_1064:
    type: scalar
```

Use the property column names from your dataset as keys under `conditions`.

### 3. Check and start training

Check the dataset path, condition names, and training settings:

```bash
prisma train local_conditional.yaml --print-config
```

Start training:

```bash
prisma train local_conditional.yaml
```

The example configuration saves checkpoints in `runs/my-materials/`. See
[Training](docs/training.md) to resume an interrupted run.

### 4. Generate with the fine-tuned model

Follow [Generation and checkpoint export](docs/generation.md#export-a-training-checkpoint)
to export the `*-best=loss.ckpt` checkpoint from `runs/my-materials/` and load
the resulting pipeline. To generate one structure, pass `dn` and `k_1064` in
the same units as the training data.

```python
import torch

structures = pipeline(
    batch_size=1,
    condition={
        "dn": torch.tensor([0.10]),
        "k_1064": torch.tensor([0.001]),
    },
    device="cuda" if torch.cuda.is_available() else "cpu",
)
structures[0].to(filename="candidate.cif")
```

## Documentation

- [Installation](docs/installation.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [Optional embedding conditioning](docs/embedding-conditioning.md)
- [Generation and checkpoint export](docs/generation.md)
- [DFT/VASP validation](docs/dft.md)

## Data and models

The generated structures, curated datasets, trained model checkpoints, and
validation datasets used in the study are available on
[Hugging Face](https://huggingface.co/xpanceo-team).

## License

PRISMA is licensed under the [MIT License](LICENSE).
