# Optional embedding conditioning

This workflow fine-tunes a generator on a vector describing each training
structure. A separate, frozen PET or MACE model extracts that vector. The
generator backbone is selected independently; the example below fine-tunes
`xpanceo-team/mattergen-base`, which uses GemNet.

Embedding extraction and generation-based validation are opt-in. Ordinary
training recipes do not construct an embedding extractor or add a generation
metric. You can also train on a vector column without enabling this metric.

## 1. Install the extractor and obtain its checkpoint

Start with the [PRISMA installation instructions](installation.md). For PET:

```bash
python -m pip install -e ".[pet]"
```

The PET-MAD v1.0.2 workflow was checked with `metatrain==2026.2.1`,
`metatomic-torch==0.1.11`, `metatensor-torch==0.8.5`,
`metatensor-core==0.1.20`, `metatensor-learn==0.4.0`, and
`metatensor-operations==0.4.0`. Its mean energy-head features have 512 components.
PRISMA uses `model_from_checkpoint(..., context="finetune")` to upgrade old
checkpoint formats and select the saved best weights. Manually passing old
hyperparameters to the current PET constructor can fail with
`KeyError: 'cutoff_function'`.

Download the [PET-MAD v1.0.2 checkpoint](https://huggingface.co/lab-cosmo/pet-mad/tree/v1.0.2/models)
from the repository root, or place your existing file at the same path:

```bash
hf download lab-cosmo/pet-mad models/pet-mad-v1.0.2.ckpt \
    --revision v1.0.2 --local-dir .
```

Both the data extraction and validation steps must use the same checkpoint and
feature definition. PET extracts `mtt::aux::energy_last_layer_features` and
takes the arithmetic mean over atoms, separately for each structure. The raw
vectors are saved as float32; they are not L2-normalized or standardized.

## 2. Prepare train and validation splits

Each row must contain a `structure` column in a supported format. For a single
Parquet table without existing splits:

```bash
prisma data prepare materials.parquet \
    --validation-fraction 0.1 --seed 42 \
    --output data/materials
```

For an existing prepared dataset with `train` and `valid`, skip this step.
See [Datasets](datasets.md) for existing splits, other column names, and Hub
sources. Embedding extraction preserves the row order and split membership.
If you extract embeddings from a Hub dataset that has only `train`, set
`data.validation_fraction: 0.1` and `data.split_seed: 42` in the training YAML.
Training creates `valid` after atom-count filtering, without recomputing the
embeddings or saving another dataset.

## 3. Add the embedding column

The default column name is `embedding`. To use a different name, for example
`structure_descriptor`, pass `--column`:

```bash
prisma data embed data/materials \
    --extractor pet \
    --checkpoint models/pet-mad-v1.0.2.ckpt \
    --column structure_descriptor \
    --max-num-atoms 20 \
    --validation-fraction 0.1 --seed 42 \
    --batch-size 16 --device cpu \
    --output data/materials-pet

prisma data inspect data/materials-pet
```

Use `--device cuda:0` for extraction on a GPU. This batch size controls the
extractor, independently of the later training batch size. The command writes
a new dataset, prints the measured embedding dimension and a ready-to-copy
`conditions` block. For another checkpoint, use that printed dimension instead
of assuming 512. An existing column is never silently overwritten.

`--max-num-atoms` and `--validation-fraction` materialize the same data selection
that training would otherwise perform dynamically. Embedding extraction first
omits unsupported-element rows, atom filtering runs next, and the seeded split
runs last. When these options are used, keep `data.max_num_atoms` in the training
recipe as a safety check, but omit `data.validation_fraction`: the saved dataset
already contains `train` and `valid`.

If a structure contains an element absent from the extractor checkpoint,
`prisma data embed` omits that row and reports the kept and skipped counts for
each split. `prisma_metadata.json` also records counts by atomic number and up
to five source row indices per split. Check these counts before training:
generation-based validation can only compare structures retained in `valid`.
Other extraction errors still stop the command. If all rows in `train` or
`valid` are omitted, the command fails without saving an output dataset.

`data/materials-pet/prisma_metadata.json` records the checkpoint SHA-256,
feature output, pooling, dimension and library versions per embedding column.
Keep this file with the local dataset. Training checks it when present.
Datasets loaded directly from the Hub or from an external table can also be
used, but without this local metadata file PRISMA cannot verify their extractor
identity; their checkpoint and feature definition must be supplied correctly.

## 4. Configure training with the custom column name

Copy [`examples/training/pet_embedding.yaml`](../examples/training/pet_embedding.yaml)
to a working configuration. For the custom column above, save the following as
`pet_embedding_custom.yaml` in the repository root:

```yaml
name: pet-embedding-custom
dataset_name_or_path: data/materials-pet

model:
  backbone: gemnet
  pretrained_model_name_or_path: xpanceo-team/mattergen-base

conditions:
  structure_descriptor:
    type: vector
    input_dim: 512

data:
  max_num_atoms: 20

training:
  max_epochs: 800
  batch_size: 32
  gradient_accumulation: 2
  learning_rate: 1.0e-4
  embedding_validation:
    condition: structure_descriptor
    extractor: pet
    checkpoint: models/pet-mad-v1.0.2.ckpt
    every_n_epochs: 5
    sample_size: 32
    batch_size: 8
    guidance_scale: 3.0
    seed: 42

logging:
  wandb:
    project: prisma

output_dir: runs/pet-embedding-custom
```

The name must agree in three places: `--column` during extraction, the key
under `conditions`, and `training.embedding_validation.condition`. Omitting
`--column` uses `embedding`; omitting the validation `condition` also selects
`embedding`. The extractor and checkpoint always require an explicit choice.

Paths are relative to the directory from which you run `prisma`. The validation
extractor uses the training device unless you set
`training.embedding_validation.device: cpu` (or another device). The frozen
extractor uses additional memory while training. Reduce its validation
`batch_size` or select CPU if needed.

## 5. Review, authenticate and train

```bash
prisma train pet_embedding_custom.yaml --print-config
wandb login
prisma train pet_embedding_custom.yaml
```

If your pretrained generator or dataset is private, authenticate with
`hf auth login` first. The print-config command checks and resolves the recipe
without loading model weights or starting W&B. Training checks the configured
embedding column and performs the standard forward/backward preflight. Before
the first training epoch, the embedding callback loads the extractor and checks
its output dimension.

## W&B metric and best checkpoints

Enabling `training.embedding_validation` adds four W&B epoch metrics:

| Metric | Definition | Use |
| --- | --- | --- |
| `mean_embedding_l2_distance/epoch` | Mean raw Euclidean distance | Compare checkpoints in one feature space; selects the best embedding checkpoint |
| `mean_embedding_cosine_distance/epoch` | Mean `1 - cosine_similarity` | Compare embedding direction across extractors |
| `mean_embedding_relative_l2_distance/epoch` | Mean L2 distance divided by the target-vector norm | Compare error relative to each target's magnitude |
| `mean_embedding_reference_normalized_l2_distance/epoch` | Mean L2 distance divided by the mean distance between distinct validation targets | Compare error with the natural scale of each extractor |

Lower values are better for all four metrics. Ordinary training does not
calculate them: the callback is only present when the recipe explicitly has an
`embedding_validation` block. The raw `mean_embedding_l2_distance` remains the
checkpoint monitor for compatibility with existing runs.
The first point is recorded after five completed epochs
with the example configuration, then after epochs 10, 15, and so on. Lightning
uses zero-based epoch labels, so these correspond to `epoch=4,9,14,...`.
Choose this metric as the Y-axis of a W&B line plot; the `epoch` field or the
training step can be used as the X-axis. W&B logging requires the `logging.wandb`
block; omitting it still allows local metric computation and checkpointing.

At each evaluation, PRISMA:

1. Uses the same seeded sample from `valid` (all rows if fewer than `sample_size`).
2. Generates one structure per target vector, retaining each reference's atom
   count and any other configured conditions. Sampling uses a fixed seed.
3. Re-embeds generated structures with the frozen extractor.
4. Logs the raw and scale-independent distances listed above. The reference
   scale is calculated once from all distinct pairs in the fixed target sample
   and saved in `embedding_validation.json`.

This is generation-based validation on a fixed subset; `loss/val` continues to
measure the ordinary diffusion loss over the validation loader. These metrics
are not extra training losses. Raw L2 values are only comparable for the same
feature space and evaluation settings; PET and MACE distances have different
scales. Use the cosine, relative, or reference-normalized metric for
cross-extractor plots. Generation performs a full diffusion sampling run and
adds work on evaluation epochs. In distributed training each rank evaluates
the same sample with its local model and the scalar is synchronized.

The run directory contains:

| File | Purpose |
| --- | --- |
| `*-best=mean_embedding_l2_distance.ckpt` | Best generator by embedding distance |
| `*-best=loss.ckpt` | Best generator by validation diffusion loss |
| `*-last.ckpt` | Latest saved training state for resuming |
| `embedding_validation.json` | Extractor identity, sampled row indices and evaluation settings |

An embedding checkpoint is created only after a scheduled evaluation. To
resume, set `training.resume_from_checkpoint` to a `*-last.ckpt` path and keep
the evaluation settings and target vectors the same. The callback checks these
against its saved state. To export the generator selected by embedding
distance, use its `*-best=mean_embedding_l2_distance.ckpt` in the existing
[export workflow](../README.md#4-export-and-use-the-trained-model).

Extractor failures and non-finite vectors stop the run with an error. Unsupported
chemical elements are not replaced by another element to obtain a metric.

## MACE or a dataset that already has embeddings

For MACE, install `mace-torch`, use `--extractor mace` with a local MACE model,
and set `extractor: mace` in the validation block. MACE uses invariant
descriptors from all layers, averaged over atoms. Use the dimension printed
by the extraction command and the same checkpoint in both steps.

If the dataset already contains the desired vectors, skip `prisma data embed`.
Set `conditions.<existing-column>.type: vector` and its `input_dim`, and point
the validation block at that same column and its original extractor. An old
MACE column named `atlas_embedding` is configured in exactly this way. If you
only want to train on the vector, omit `training.embedding_validation`.
