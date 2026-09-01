from pathlib import Path

import pytest
from omegaconf import OmegaConf

from prisma.training.cli import main as training_main
from prisma.training.configuration import TrainingRecipe, compose_training_config


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("examples/training").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_example_training_recipe_is_valid(config_path: Path) -> None:
    recipe = TrainingRecipe.from_file(config_path)

    compose_training_config(recipe)


def test_local_conditional_pipeline_recipe(tmp_path: Path) -> None:
    dataset_path = tmp_path / "materials"
    dataset_path.mkdir()
    recipe = TrainingRecipe.from_mapping(
        {
            "name": "materials",
            "dataset_name_or_path": str(dataset_path),
            "model": {
                "backbone": "gemnet",
                "pretrained_model_name_or_path": "organization/mattergen-base",
            },
            "conditions": {
                "bandgap": {"type": "scalar"},
                "class": {"type": "categorical", "num_categories": 3},
            },
            "training": {
                "batch_size": 16,
                "gradient_accumulation": 2,
                "max_epochs": 10,
            },
        }
    )

    cfg = compose_training_config(recipe)

    assert cfg.data.dataset_path == str(dataset_path)
    assert cfg.data.dataset_name is None
    assert cfg.training.optimization.effective_batch_size == 32
    assert cfg.model.gnn["_target_"].endswith("GemNetTWrapper.from_pretrained")
    assert cfg.model.gnn.pretrained_model_name_or_path == "organization/mattergen-base"
    assert cfg.diffusion.cell_scheduler.pretrained_model_name_or_path == (
        "organization/mattergen-base"
    )
    assert cfg.model.cond_encoder.condition.bandgap.encoding_type == "sinusoidal"
    assert cfg.model.cond_encoder.condition.bandgap.scale is True
    assert cfg.model.cond_encoder.condition["class"].num_categories == 3


@pytest.mark.parametrize(
    ("backbone", "target_suffix", "atom_emb_dim"),
    [
        ("equiformer_v2", "EquiformerV2Wrapper", 96),
        ("pet", "PETWrapper", 1280),
    ],
)
def test_foundational_architecture_recipes(
    backbone: str, target_suffix: str, atom_emb_dim: int
) -> None:
    recipe = TrainingRecipe.from_mapping(
        {
            "name": f"{backbone}-foundation",
            "dataset_name_or_path": "organization/materials",
            "model": {"backbone": backbone},
        }
    )

    cfg = compose_training_config(recipe)

    assert cfg.data.dataset_name == "organization/materials"
    assert cfg.model.gnn["_target_"].endswith(target_suffix)
    assert cfg.model.gnn.atom_emb_dim == atom_emb_dim
    assert cfg.model.cond_encoder.condition == {}
    assert "pretrained_model_name_or_path" not in cfg.model.gnn


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "name": "bad",
                "dataset_name_or_path": "organization/materials",
                "model": {"backbone": "gemnet"},
                "training": {"batch_szie": 8},
            },
            "batch_szie",
        ),
        (
            {
                "name": "bad",
                "dataset_name_or_path": "organization/materials",
                "model": {"backbone": "gemnet"},
                "conditions": {"class": {"type": "categorical"}},
            },
            "num_categories",
        ),
    ],
)
def test_invalid_public_configuration_fails(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingRecipe.from_mapping(config)


def test_print_config_does_not_start_training(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "training.yaml"
    OmegaConf.save(
        {
            "name": "foundation",
            "dataset_name_or_path": "organization/materials",
            "model": {"backbone": "equiformer_v2"},
        },
        config_path,
    )
    monkeypatch.setattr(
        "prisma.training.cli.run_training",
        lambda *args, **kwargs: pytest.fail("training should not start"),
    )

    training_main([str(config_path), "--print-config"])

    output = capsys.readouterr().out
    assert "EquiformerV2Wrapper" in output
    assert "batch_size: 8" in output


def test_skip_preflight_is_forwarded_to_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "training.yaml"
    OmegaConf.save(
        {
            "name": "foundation",
            "dataset_name_or_path": "organization/materials",
            "model": {"backbone": "gemnet"},
        },
        config_path,
    )
    calls = []
    monkeypatch.setattr(
        "prisma.training.cli.run_training",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    training_main([str(config_path), "--skip-preflight"])

    assert calls == [{"preflight": False}]
