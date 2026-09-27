from types import SimpleNamespace

from datasets import Dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.loggers.logger import Logger
import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import through the training entry point to include its logger naming convention.
from prisma.training.training_utils import build_callbacks
from prisma.training.configuration import TrainingRecipe, compose_training_config
from prisma.training.embedding_validation import (
    EMBEDDING_COSINE_METRIC,
    EmbeddingValidationCallback,
    EMBEDDING_METRIC,
    EMBEDDING_REFERENCE_NORMALIZED_L2_METRIC,
    EMBEDDING_RELATIVE_L2_METRIC,
    _embedding_metrics,
)


def recipe_config(validation=True):
    config = {
        "name": "test",
        "dataset_name_or_path": "organization/data",
        "model": {"backbone": "gemnet"},
        "conditions": {"custom_vector": {"type": "vector", "input_dim": 2}},
        "training": {},
    }
    if validation:
        config["training"]["embedding_validation"] = {
            "condition": "custom_vector",
            "extractor": "pet",
            "checkpoint": "weights.ckpt",
            "every_n_epochs": 2,
            "batch_size": 2,
            "sample_size": 32,
        }
    return config


def datamodule():
    rows = Dataset.from_dict(
        {
            "cell": [[np.eye(3).tolist()]] * 3,
            "atomic_numbers": [[14]] * 3,
            "frac_coords": [[[0, 0, 0]]] * 3,
            "num_atoms": [1] * 3,
            "custom_vector": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        }
    )
    return SimpleNamespace(
        train_dataset=rows,
        valid_dataset=rows,
        dataset_path=None,
        condition={"custom_vector": {"input_dim": 2}},
    )


class ToyExtractor:
    def encode(self, structures):
        return np.array([[s.lattice.a, 0] for s in structures], dtype=np.float32)

    def metadata(self):
        return {
            "extractor": "pet",
            "checkpoint_sha256": "test-checkpoint",
            "dimension": 2,
        }


class RecordingLogger(Logger):
    def __init__(self):
        super().__init__()
        self.records = []

    @property
    def name(self):
        return "test"

    @property
    def version(self):
        return "0"

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        self.records.append(dict(metrics))


class TinyModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.cond_encoder = SimpleNamespace(condition_keys=["custom_vector"])

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("loss/val", self.weight.square(), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_metric_reaches_logger_and_selects_checkpoint_only_on_evaluation_epochs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "prisma.training.embedding_validation.create_embedder",
        lambda *args: ToyExtractor(),
    )
    cfg = compose_training_config(TrainingRecipe.from_mapping(recipe_config()))
    callbacks = build_callbacks(cfg, tmp_path, datamodule())
    callbacks = [cb for cb in callbacks if not isinstance(cb, ProgressBar)]
    evaluation = next(
        cb for cb in callbacks if isinstance(cb, EmbeddingValidationCallback)
    )
    seen = []

    def make_pipeline(module):
        def generate(**kwargs):
            seen.append(
                (module.current_epoch, kwargs["batch_size"], kwargs["condition"])
            )
            return [
                Structure(
                    Lattice.cubic(
                        target[0] + (target[0] + 1) * (5 - module.current_epoch)
                    ),
                    ["Si"],
                    [[0, 0, 0]],
                )
                for target in kwargs["condition"]["custom_vector"]
            ]

        return generate

    monkeypatch.setattr(evaluation, "_make_pipeline", make_pipeline)
    logger = RecordingLogger()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=4,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
    )
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    trainer.fit(TinyModule(), train_dataloaders=loader, val_dataloaders=loader)

    assert [(epoch, size) for epoch, size, _ in seen] == [
        (1, 2),
        (1, 1),
        (3, 2),
        (3, 1),
    ]
    assert all(set(conditions) == {"custom_vector"} for _, _, conditions in seen)
    logged_name = EMBEDDING_METRIC + "/epoch"
    metric_records = [
        record[logged_name] for record in logger.records if logged_name in record
    ]
    assert metric_records == pytest.approx([12.0, 6.0]), (
        logger.records,
        trainer.callback_metrics,
    )
    checkpoint = next(
        cb
        for cb in callbacks
        if isinstance(cb, ModelCheckpoint) and cb.monitor == EMBEDDING_METRIC
    )
    assert checkpoint.best_model_score.item() == pytest.approx(6.0)
    assert "epoch=03" in checkpoint.best_model_path
    restored = torch.load(
        checkpoint.best_model_path, weights_only=False, map_location="cpu"
    )
    assert restored["callbacks"][evaluation.state_key]["condition"] == "custom_vector"
    assert (tmp_path / "embedding_validation.json").is_file()

    for name in (
        EMBEDDING_COSINE_METRIC,
        EMBEDDING_RELATIVE_L2_METRIC,
        EMBEDDING_REFERENCE_NORMALIZED_L2_METRIC,
    ):
        assert sum(name + "/epoch" in record for record in logger.records) == 2


def test_scale_independent_embedding_metrics():
    targets = np.array([[1.0, 0.0], [0.0, 2.0]])
    generated = np.array([[2.0, 0.0], [0.0, 1.0]])

    metrics = _embedding_metrics(generated, targets, reference_l2_scale=np.sqrt(5))

    assert metrics == pytest.approx(
        {
            EMBEDDING_METRIC: 1.0,
            EMBEDDING_COSINE_METRIC: 0.0,
            EMBEDDING_RELATIVE_L2_METRIC: 0.75,
            EMBEDDING_REFERENCE_NORMALIZED_L2_METRIC: 1 / np.sqrt(5),
        }
    )


def test_default_training_has_no_embedding_callback_or_checkpoint(tmp_path):
    cfg = compose_training_config(
        TrainingRecipe.from_mapping(recipe_config(validation=False))
    )
    callbacks = build_callbacks(cfg, tmp_path, datamodule=None)
    assert not any(isinstance(cb, EmbeddingValidationCallback) for cb in callbacks)
    assert [
        cb.monitor for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb.monitor
    ] == ["loss/val"]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"condition": "missing"}, "configured vector"),
        ({"every_n_epochs": 0}, "positive integer"),
        ({"extractor": "unknown"}, "extractor"),
        ({"sample_szie": 10}, "sample_szie"),
    ],
)
def test_invalid_embedding_recipe_fails_before_loading_models(change, message):
    config = recipe_config()
    config["training"]["embedding_validation"].update(change)
    with pytest.raises(ValueError, match=message):
        TrainingRecipe.from_mapping(config)


def test_checkpoint_provenance_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "prisma.training.embedding_validation.create_embedder",
        lambda *args: ToyExtractor(),
    )
    callback = EmbeddingValidationCallback(
        datamodule(), "pet", "weights.ckpt", condition="custom_vector"
    )
    callback.expected_metadata = {
        "extractor": "pet",
        "checkpoint_sha256": "different-model",
    }
    trainer = SimpleNamespace(
        is_global_zero=True, default_root_dir=tmp_path, loggers=[]
    )
    with pytest.raises(ValueError, match="different checkpoint_sha256"):
        callback.on_fit_start(trainer, TinyModule())
