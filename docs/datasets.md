# Datasets

PRISMA trains from local datasets or datasets hosted on the Hugging Face Hub.
A local dataset can be prepared and used directly without publishing it.

## Dataset schema

Each row represents one crystal structure. A training table contains:

- `structure`: required pymatgen JSON, CIF text, or POSCAR text;
- `material_id`: a recommended unique identifier;
- one column per conditioning property;
- any additional metadata columns that should be preserved.

Scalar condition columns must contain finite numeric values. Their names must
exactly match the names under `conditions` in the training YAML. Parquet is the
recommended source format because it preserves numeric types and handles
multiline structure values reliably.

## Prepare a local table

```bash
prisma data prepare materials.parquet \
    --validation-fraction 0.1 \
    --seed 42 \
    --output data/materials
```

This converts structures to pymatgen JSON and saves an Arrow-backed Hugging
Face `DatasetDict`. The source file is not modified. The validation split is
deterministic for a given input and seed.

If the source uses different column names or a known structure format:

```bash
prisma data prepare materials.csv \
    --structure-column crystal \
    --structure-format cif \
    --id-column source_id \
    --validation-fraction 0.1 \
    --output data/materials
```

Supported source files are Parquet, CSV, JSON, and JSONL. A directory may also
contain `train`, `valid`, and optional `test` files in one of those formats.

## Existing splits

PRISMA stores the canonical split names `train`, `valid`, and `test`. A test
split is optional and is not used during training.

If one table has a column containing split names:

```bash
prisma data prepare materials.parquet \
    --split-column split \
    --output data/materials
```

The input name `validation` is accepted and stored as `valid`. Do not pass
`--validation-fraction` when the source already provides a validation split.

Inspect the prepared artifact before training:

```bash
prisma data inspect data/materials
```

## Atom-count filtering

The training configuration applies `data.max_num_atoms` before deriving a
validation split. Its default is 20 atoms:

```yaml
data:
  max_num_atoms: 20
```

Increase it only when the model and available GPU memory are intended for
larger structures. Graph memory consumption can grow substantially with both
atom count and neighbor count.

## Hugging Face datasets

Use a Hub dataset directly in a training configuration:

```yaml
dataset_name_or_path: your-organization/materials
dataset_config_name: default
dataset_revision: main
```

A local artifact can be published for sharing and versioned training:

```bash
huggingface-cli login
prisma data push data/materials your-organization/materials --private
```

To normalize an existing Hub dataset locally:

```bash
prisma data prepare your-organization/materials \
    --source-type hub \
    --revision main \
    --output data/materials
```
