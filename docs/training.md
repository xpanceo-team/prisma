# Training

PRISMA can fine-tune a pretrained MatterGen model on property-labeled crystal
structures. This is the recommended starting point for adapting the model to a
custom dataset.

## Conditional fine-tuning configuration

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
  checkpoint_every_n_epochs: 5

output_dir: runs/my-materials
```

When the prepared dataset already contains `train` and `valid`, no validation
setting is necessary. For a train-only dataset, add:

```yaml
data:
  validation_fraction: 0.1
  split_seed: 42
```

Each condition configuration selects its encoder. Scalar values are scaled
using statistics from the training split and encoded sinusoidally. Adding a
condition to the base model creates condition-specific encoder and adapter
weights; messages identifying those newly initialized weights are expected.

## Review and run

Inspect the resolved training configuration:

```bash
prisma train local_conditional.yaml --print-config
```

Start training:

```bash
prisma train local_conditional.yaml
```

PRISMA performs an isolated forward and backward step before creating the
Lightning run. This catches incompatible data, model, and memory settings
before a long experiment begins.

Graph batches do not have uniform memory cost. If preflight reports a CUDA
out-of-memory error, reduce the physical batch size and increase gradient
accumulation. The approximate effective batch size is:

```text
batch_size x gradient_accumulation x number_of_devices
```

For example, batch size 16 with accumulation 4 has the same single-device
effective batch size as batch size 32 with accumulation 2.

## Command-line overrides

Override a configured value without editing the YAML:

```bash
prisma train local_conditional.yaml \
    --set training.max_epochs=100 \
    --set training.batch_size=16
```

Only fields already present in the public recipe can be overridden.

## Checkpoints and resuming

Periodic checkpointing is configured with:

```yaml
training:
  checkpoint_every_n_epochs: 5
```

Resume an interrupted Lightning run by adding its checkpoint path:

```yaml
training:
  resume_from_checkpoint: runs/my-materials/epoch=07-step=0128-last.ckpt
```

Use the most recent `*-last.ckpt` file in the run directory; its epoch and step
numbers depend on the run. Then run the same `prisma train` command. This
checkpoint contains trainer and optimizer state for resuming training. Use the
single `*-best=loss.ckpt` file selected by validation loss when exporting a
model for generation. A published PRISMA pipeline contains the components
needed for generation; the formats serve different purposes.

## Experiment logging

Weights & Biases logging is opt-in:

```yaml
logging:
  wandb:
    project: prisma
```

Authenticate with `wandb login` before starting the run.

## Backbones

`model.backbone` accepts `gemnet`, `equiformer_v2`, or `pet`. A pretrained
pipeline must contain weights compatible with the selected backbone. PET also
requires the `pet` installation extra.

The files under [`examples/training`](../examples/training/) provide ready-to-use
training configurations. [`noemd.yaml`](../examples/training/noemd.yaml) is the
reference multi-property fine-tuning experiment.
