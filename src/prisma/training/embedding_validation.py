from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from pymatgen.core import Structure
import torch

from prisma.data.persistence import load_dataset_metadata
from prisma.embeddings import create_embedder
from prisma.embeddings.base import validate_embeddings
from prisma.pipelines.mattergen.pipeline_mattergen import MatterGenPipeline

EMBEDDING_METRIC = "mean_embedding_l2_distance"


class EmbeddingValidationCallback(pl.Callback):
    """Evaluate generated structures in a fixed, explicitly selected feature space."""

    def __init__(
        self,
        datamodule,
        extractor: str,
        checkpoint: str,
        condition: str = "embedding",
        every_n_epochs: int = 5,
        sample_size: int = 32,
        batch_size: int = 8,
        guidance_scale: float = 3.0,
        seed: int = 42,
        device: str | None = None,
    ):
        self.extractor_name = extractor
        self.checkpoint = checkpoint
        self.condition = condition
        self.every_n_epochs = every_n_epochs
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.extractor_device = device
        self.embedder = None
        self._restored_state = None
        self._report = None
        self._last_evaluated_epoch = None
        if any(
            type(value) is not int or value < 1
            for value in (every_n_epochs, sample_size, batch_size)
        ):
            raise ValueError(
                "Embedding evaluation periods and sample/batch sizes must be positive integers."
            )
        if condition not in datamodule.condition:
            raise ValueError(
                f"Embedding validation condition {condition!r} is not configured for training."
            )
        self.dimension = int(datamodule.condition[condition]["input_dim"])
        for name in ("train", "valid"):
            rows = getattr(datamodule, f"{name}_dataset")
            if rows is None or not len(rows):
                raise ValueError(
                    f"Embedding validation requires a non-empty {name} split."
                )
            if condition not in rows.column_names:
                raise ValueError(
                    f"Missing embedding column {condition!r} in {name} split."
                )
            for batch in (
                rows.with_format(None).select_columns([condition]).iter(batch_size=1024)
            ):
                try:
                    validate_embeddings(
                        batch[condition],
                        count=len(batch[condition]),
                        dimension=self.dimension,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid {name} column {condition!r}: {exc}"
                    ) from exc

        raw = datamodule.valid_dataset.with_format(None)
        self.indices = (
            np.random.default_rng(seed)
            .choice(len(raw), size=min(sample_size, len(raw)), replace=False)
            .tolist()
        )
        self.samples = raw.select(self.indices)[:]
        self.targets = validate_embeddings(
            self.samples[condition], count=len(self.indices), dimension=self.dimension
        )
        self.num_atoms = (
            np.asarray(self.samples["num_atoms"]).reshape(-1).astype(int).tolist()
        )
        source = getattr(datamodule, "dataset_path", None)
        self.expected_metadata = (
            load_dataset_metadata(source).get("embeddings", {}).get(condition)
            if isinstance(source, (str, Path))
            else None
        )

    @property
    def state_key(self):
        return f"{type(self).__qualname__}[{self.condition}]"

    def state_dict(self):
        return self._report or {}

    def load_state_dict(self, state_dict):
        self._restored_state = state_dict or None

    def _initialize(self, trainer, pl_module):
        if self.embedder is not None:
            return
        device = self.extractor_device or str(pl_module.device)
        with torch.random.fork_rng(devices=[]), torch.inference_mode(False):
            self.embedder = create_embedder(
                self.extractor_name, self.checkpoint, device
            )
            reference = Structure(
                lattice=np.asarray(self.samples["cell"][0]).reshape(3, 3),
                species=self.samples["atomic_numbers"][0],
                coords=self.samples["frac_coords"][0],
            )
            validate_embeddings(
                self.embedder.encode([reference]), count=1, dimension=self.dimension
            )
        metadata = self.embedder.metadata()
        # Paths and installed package versions may differ; the feature-space identity
        # is the checkpoint, output, pooling, dtype, dimension and weight selection.
        identity_keys = (
            "extractor",
            "checkpoint_sha256",
            "output",
            "pooling",
            "dtype",
            "dimension",
            "checkpoint_context",
        )
        if self.expected_metadata is not None:
            for key in identity_keys:
                if self.expected_metadata.get(key) != metadata.get(key):
                    raise ValueError(
                        f"Embedding column {self.condition!r} was computed with a different {key}."
                    )
        sample_hash = hashlib.sha256(
            json.dumps(
                {"targets": self.targets.tolist(), "num_atoms": self.num_atoms},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self._report = {
            "condition": self.condition,
            "extractor": metadata,
            "sample_indices": self.indices,
            "sample_hash": sample_hash,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "guidance_scale": self.guidance_scale,
            "every_n_epochs": self.every_n_epochs,
        }
        if self._restored_state:
            previous = self._restored_state
            for key in (
                "condition",
                "sample_hash",
                "seed",
                "batch_size",
                "guidance_scale",
                "every_n_epochs",
            ):
                if previous.get(key) != self._report[key]:
                    raise ValueError(
                        f"Embedding validation {key} changed when resuming the run."
                    )
            for key in identity_keys:
                if previous["extractor"].get(key) != metadata.get(key):
                    raise ValueError(
                        f"Embedding extractor {key} changed when resuming the run."
                    )
        if trainer.is_global_zero:
            path = Path(trainer.default_root_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "embedding_validation.json").write_text(
                json.dumps(self._report, indent=2) + "\n", encoding="utf-8"
            )
            for logger in trainer.loggers:
                logger.log_hyperparams({"embedding_validation": self._report})

    def on_fit_start(self, trainer, pl_module):
        self._initialize(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        if self._last_evaluated_epoch == trainer.current_epoch:
            return
        self._initialize(trainer, pl_module)
        device = pl_module.device
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        # Each distributed rank evaluates the same fixed sample with its local model;
        # Lightning synchronizes the scalar, so every rank takes the same collectives.
        with (
            torch.random.fork_rng(devices=cuda_devices),
            torch.inference_mode(False),
            torch.no_grad(),
        ):
            pipeline = self._make_pipeline(pl_module)
            generator = torch.Generator(device=device).manual_seed(self.seed)
            distances = []
            for start in range(0, len(self.indices), self.batch_size):
                stop = min(start + self.batch_size, len(self.indices))
                # Keep other configured conditions at their validation-row values.
                conditions = {
                    key: self.samples[key][start:stop]
                    for key in pl_module.cond_encoder.condition_keys
                }
                structures = pipeline(
                    batch_size=stop - start,
                    condition=conditions,
                    num_atoms=self.num_atoms[start:stop],
                    guidance_scale=self.guidance_scale,
                    device=device,
                    generator=generator,
                )
                generated = validate_embeddings(
                    self.embedder.encode(structures),
                    count=stop - start,
                    dimension=self.dimension,
                )
                delta = generated.astype(np.float64) - self.targets[start:stop].astype(
                    np.float64
                )
                distances.extend(np.linalg.norm(delta, axis=1).tolist())
        pl_module.log(
            EMBEDDING_METRIC,
            float(np.mean(distances)),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(self.indices),
            sync_dist=True,
        )
        self._last_evaluated_epoch = trainer.current_epoch

    @staticmethod
    def _make_pipeline(pl_module):
        return MatterGenPipeline(
            gnn=pl_module.gnn,
            condition_encoder=pl_module.cond_encoder,
            score_model=pl_module.score_model,
            atomic_numbers_scheduler=pl_module.atomic_numbers_scheduler,
            frac_coords_scheduler=pl_module.frac_coords_scheduler,
            cell_scheduler=pl_module.cell_scheduler,
        )
