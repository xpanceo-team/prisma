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
    torch.save(
        {"datamodule_hyper_parameters": dict(datamodule.hparams)}, checkpoint
    )
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


def test_training_step_rejects_non_finite_loss(monkeypatch):
    module = _training_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "step",
        lambda *args, **kwargs: {"loss": torch.tensor(float("nan"))},
    )

    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        module.training_step({}, 0)
