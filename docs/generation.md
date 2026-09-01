# Generation and checkpoint export

## Load a published pipeline

```python
import torch

from prisma.pipelines.mattergen import MatterGenPipeline


pipeline = MatterGenPipeline.from_pretrained(
    "xpanceo-team/mattergen-formula-e_above_hull",
    use_safetensors=False,
)

device = "cuda"
condition = {
    "e_above_hull": torch.tensor([0.0430, 0.0430]),
    "formula": ["Na3 Mn1 Co1 Ni1 O6", "Na3 Mn1 Co1 Ni1 O6"],
}

structures = pipeline(
    batch_size=2,
    condition=condition,
    guidance_scale=3.0,
    device=device,
    generator=torch.Generator(device=device).manual_seed(42),
)
```

Condition names and tensor shapes must match the conditions used to train the
pipeline. `guidance_scale` affects conditioned generation; larger values place
more emphasis on the supplied condition and should be evaluated for the target
dataset.

## Export a training checkpoint

A Lightning checkpoint can be reconstructed and saved as a reusable PRISMA
pipeline:

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

Load the local pipeline with the same API:

```python
pipeline = MatterGenPipeline.from_pretrained(
    "runs/my-materials/pipeline",
    use_safetensors=False,
)
```

Publish it when it is ready to share:

```python
pipeline.push_to_hub(
    "your-organization/mattergen-my-materials",
    private=True,
    safe_serialization=False,
)
```
