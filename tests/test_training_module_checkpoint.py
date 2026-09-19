import io

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from prisma.training import module as training_module
from prisma.training.datamodule import DataModule


class _Component(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition_keys = []
        self.config = OmegaConf.create(
            {
                "condition": {},
                "condition_dim": 8,
                "t_emb_dim": 8,
            }
        )


def test_training_hparams_are_safe_to_load_and_runtime_config_supports_dot_access(
    monkeypatch,
):
    monkeypatch.setattr(
        training_module,
        "instantiate_from_pretrained",
        lambda *args, **kwargs: _Component(),
    )

    module = training_module.TrainingModule(
        condition_stats=OmegaConf.create({"energy": {"mean": [0.0]}}),
        model=OmegaConf.create(
            {
                "cond_encoder": {"_target_": "ConditionEncoder"},
                "gnn": {"_target_": "PETWrapper"},
                "score_model": {"_target_": "MatterGenModel"},
            }
        ),
        diffusion=OmegaConf.create(
            {
                "atomic_numbers_scheduler": {"_target_": "D3PMScheduler"},
                "frac_coords_scheduler": {"_target_": "VEScheduler"},
                "cell_scheduler": {"_target_": "VPScheduler"},
            }
        ),
        optimization=OmegaConf.create({"optimizer": {"_target_": "Adam"}}),
    )

    assert not any(isinstance(value, DictConfig) for value in module.hparams.values())
    assert module.cfg.model.gnn._target_ == "PETWrapper"

    checkpoint = io.BytesIO()
    torch.save({"hyper_parameters": dict(module.hparams)}, checkpoint)
    checkpoint.seek(0)

    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded["hyper_parameters"]["model"]["gnn"]["_target_"] == "PETWrapper"


def test_datamodule_hparams_are_safe_to_load_and_runtime_config_supports_dot_access():
    datamodule = DataModule(
        dataset_name="organization/dataset",
        dataset_path=None,
        dataset_subset=None,
        revision=None,
        data_cls="prisma.data.StructureData",
        condition=OmegaConf.create({"energy": {"scale": True}}),
        max_num_atoms=20,
        validation_fraction=None,
        split_seed=42,
        persistent_workers=False,
        num_workers=0,
        batch_size=2,
    )

    assert not any(
        isinstance(value, DictConfig) for value in datamodule.hparams.values()
    )
    assert datamodule.cfg.condition.energy.scale is True

    checkpoint = io.BytesIO()
    torch.save({"datamodule_hyper_parameters": dict(datamodule.hparams)}, checkpoint)
    checkpoint.seek(0)

    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded["datamodule_hyper_parameters"]["condition"]["energy"] == {
        "scale": True
    }


def test_training_module_combines_condition_definition_and_statistics(monkeypatch):
    calls = []

    def instantiate(*args, **kwargs):
        calls.append(kwargs)
        component = _Component()
        condition = kwargs.get("condition")
        if condition:
            component.condition_keys = list(condition)
            component.config.condition = OmegaConf.create(condition)
        return component

    monkeypatch.setattr(training_module, "instantiate_from_pretrained", instantiate)

    training_module.TrainingModule(
        condition_stats={"energy": {"scale_mean": [1.0], "scale_std": [2.0]}},
        model={
            "cond_encoder": {
                "_target_": "ConditionEncoder",
                "condition": {
                    "energy": {
                        "condition_type": "adapter",
                        "encoding_type": "sinusoidal",
                        "scale": True,
                    }
                },
            },
            "gnn": {"_target_": "GemNetTWrapper"},
            "score_model": {"_target_": "MatterGenModel"},
        },
        diffusion={
            "atomic_numbers_scheduler": {"_target_": "D3PMScheduler"},
            "frac_coords_scheduler": {"_target_": "VEScheduler"},
            "cell_scheduler": {"_target_": "VPScheduler"},
        },
        optimization={"optimizer": {"_target_": "Adam"}},
    )

    assert calls[0]["condition"]["energy"] == {
        "condition_type": "adapter",
        "encoding_type": "sinusoidal",
        "scale": True,
        "scale_mean": [1.0],
        "scale_std": [2.0],
    }


def _training_module(monkeypatch) -> training_module.TrainingModule:
    monkeypatch.setattr(
        training_module,
        "instantiate_from_pretrained",
        lambda *args, **kwargs: _Component(),
    )
    return training_module.TrainingModule(
        condition_stats={},
        model={
            "cond_encoder": {"_target_": "ConditionEncoder", "condition": {}},
            "gnn": {"_target_": "GemNetTWrapper"},
            "score_model": {"_target_": "MatterGenModel"},
        },
        diffusion={
            "atomic_numbers_scheduler": {"_target_": "D3PMScheduler"},
            "frac_coords_scheduler": {"_target_": "VEScheduler"},
            "cell_scheduler": {"_target_": "VPScheduler"},
        },
        optimization={"optimizer": {"_target_": "Adam"}},
    )


def test_training_step_propagates_batch_errors(monkeypatch):
    module = _training_module(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("invalid structure")

    monkeypatch.setattr(module, "_step", fail)

    with pytest.raises(ValueError, match="invalid structure"):
        module.step({})


@pytest.mark.parametrize("stage", ["training_step", "validation_step", "test_step"])
def test_gemnet_no_neighbors_skips_only_the_affected_batch(monkeypatch, stage):
    from prisma.backbones.gemnet.gemnet import NoNeighborsError

    module = _training_module(monkeypatch)

    def no_neighbors(*args, **kwargs):
        raise NoNeighborsError([2])

    monkeypatch.setattr(module, "step", no_neighbors)
    assert getattr(module, stage)({}, 7) is None


def test_other_training_errors_still_propagate(monkeypatch):
    module = _training_module(monkeypatch)

    def invalid(*args, **kwargs):
        raise ValueError("invalid structure")

    monkeypatch.setattr(module, "step", invalid)
    with pytest.raises(ValueError, match="invalid structure"):
        module.training_step({}, 0)


def test_gemnet_empty_graph_error_does_not_require_material_ids():
    from prisma.backbones.gemnet.gemnet import GemNetT, NoNeighborsError

    with pytest.raises(NoNeighborsError, match="batch image indices=\\[1\\]"):
        GemNetT.select_edges(
            None,
            data=object(),
            edge_index=torch.tensor([[0], [0]]),
            cell_offsets=torch.zeros((1, 3)),
            neighbors=torch.tensor([1, 0]),
            edge_dist=torch.ones(1),
            edge_vector=torch.zeros((1, 3)),
        )


def test_lightning_continues_after_no_neighbor_batch(tmp_path):
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

    from prisma.backbones.gemnet.gemnet import NoNeighborsError

    class SometimesEmpty(pl.LightningModule):
        training_step = training_module.TrainingModule.training_step
        _run_step_with_batch_recovery = (
            training_module.TrainingModule._run_step_with_batch_recovery
        )
        _ensure_finite_loss = staticmethod(
            training_module.TrainingModule._ensure_finite_loss
        )

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def step(self, batch, batch_idx, dataloader_idx):
            if batch_idx == 0:
                raise NoNeighborsError([0])
            return {"loss": self.weight.square()}

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.1)

    module = SometimesEmpty()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader([0, 1], batch_size=1))
    assert trainer.global_step == 1
    assert module.weight.item() < 1.0


def test_training_step_rejects_non_finite_loss(monkeypatch):
    module = _training_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "step",
        lambda *args, **kwargs: {"loss": torch.tensor(float("nan"))},
    )

    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        module.training_step({}, 0)
