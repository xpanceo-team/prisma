from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml

from prisma.utils.resolvers import register_resolvers


_BACKBONE_TARGETS = {
    "gemnet": "prisma.models.gnns.gemnet.GemNetTWrapper",
    "equiformer_v2": "prisma.models.gnns.equiformer_v2.EquiformerV2Wrapper",
    "pet": "prisma.models.gnns.pet.PETWrapper",
}

# These are the established foundational-training architectures from the
# repository's eq2_base and pet_base experiments.
_BACKBONE_DEFAULTS = {
    "gemnet": {},
    "equiformer_v2": {
        "num_layers": 12,
        "sphere_channels": 96,
        "atom_emb_dim": 96,
        "edge_channels": 96,
        "edge_emb_dim": 96,
        "attn_hidden_channels": 48,
        "attn_alpha_channels": 48,
        "attn_value_channels": 12,
        "ffn_hidden_channels": 96,
    },
    "pet": {
        "cutoff": 7.0,
        "max_neighbors": 50,
        "graph_max_neighbors": 384,
        "use_adaptive_neighbors": False,
        "atom_emb_dim": 1280,
        "edge_emb_dim": 320,
        "head_dim": 320,
        "feedforward_dim": 640,
    },
}

_BACKBONE_TRAINING_DEFAULTS = {
    "gemnet": {"batch_size": 128, "gradient_accumulation": 4},
    "equiformer_v2": {"batch_size": 8, "gradient_accumulation": 2},
    "pet": {"batch_size": 128, "gradient_accumulation": 4},
}

_ROOT_FIELDS = {
    "name",
    "dataset_name_or_path",
    "dataset_config_name",
    "dataset_revision",
    "data",
    "model",
    "conditions",
    "training",
    "output_dir",
    "logging",
}
_DATA_FIELDS = {"max_num_atoms", "validation_fraction", "split_seed", "num_workers"}
_MODEL_FIELDS = {
    "backbone",
    "pretrained_model_name_or_path",
    "config",
    "freeze_except_new",
}
_TRAINING_FIELDS = {
    "max_epochs",
    "batch_size",
    "gradient_accumulation",
    "learning_rate",
    "weight_decay",
    "precision",
    "accelerator",
    "devices",
    "resume_from_checkpoint",
    "checkpoint_every_n_epochs",
    "seed",
}
_LOGGING_FIELDS = {"wandb"}
_CONDITION_FIELDS = {"type", "scale", "input_dim", "num_categories"}


@dataclass
class TrainingRecipe:
    name: str
    dataset_name_or_path: str
    model: dict[str, Any]
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    dataset_config_name: str | None = None
    dataset_revision: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingRecipe":
        recipe_path = Path(path).expanduser()
        if not recipe_path.is_file():
            raise FileNotFoundError(
                f"Training configuration does not exist: {recipe_path}"
            )
        content = OmegaConf.to_container(OmegaConf.load(recipe_path), resolve=True)
        if not isinstance(content, Mapping):
            raise TypeError("Training configuration must contain a YAML mapping.")
        return cls.from_mapping(content)

    @classmethod
    def from_mapping(cls, content: Mapping[str, Any]) -> "TrainingRecipe":
        content = dict(content)
        _reject_unknown(content, _ROOT_FIELDS, "configuration")
        for required in ("name", "dataset_name_or_path", "model"):
            if not content.get(required):
                raise ValueError(f"Missing required field: {required}")

        model = _mapping(content.get("model"), "model")
        data = _mapping(content.get("data", {}), "data")
        training = _mapping(content.get("training", {}), "training")
        logging = _mapping(content.get("logging", {}), "logging")
        conditions = _mapping(content.get("conditions", {}), "conditions")
        _reject_unknown(model, _MODEL_FIELDS, "model")
        _reject_unknown(data, _DATA_FIELDS, "data")
        _reject_unknown(training, _TRAINING_FIELDS, "training")
        _reject_unknown(logging, _LOGGING_FIELDS, "logging")

        backbone = model.get("backbone")
        if backbone not in _BACKBONE_TARGETS:
            choices = ", ".join(_BACKBONE_TARGETS)
            raise ValueError(f"model.backbone must be one of: {choices}")
        model["config"] = _mapping(model.get("config", {}), "model.config")

        normalized_conditions = {}
        for name, condition in conditions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Condition names must be non-empty strings.")
            condition = _mapping(condition, f"conditions.{name}")
            _reject_unknown(condition, _CONDITION_FIELDS, f"conditions.{name}")
            condition_type = condition.get("type")
            if condition_type not in {
                "scalar",
                "categorical",
                "vector",
                "space_group",
                "chemical_system",
            }:
                raise ValueError(
                    f"Unsupported condition type for {name!r}: {condition_type!r}"
                )
            if condition_type == "categorical" and not condition.get("num_categories"):
                raise ValueError(f"conditions.{name}.num_categories is required.")
            if condition_type in {"vector", "chemical_system"} and not condition.get(
                "input_dim"
            ):
                raise ValueError(f"conditions.{name}.input_dim is required.")
            normalized_conditions[name] = condition

        _validate_positive(data, "max_num_atoms")
        _validate_positive(data, "num_workers", allow_zero=True)
        fraction = data.get("validation_fraction")
        if fraction is not None and not 0 < fraction < 1:
            raise ValueError("data.validation_fraction must be between 0 and 1.")
        for key in (
            "max_epochs",
            "batch_size",
            "gradient_accumulation",
            "learning_rate",
        ):
            _validate_positive(training, key)

        return cls(
            name=str(content["name"]),
            dataset_name_or_path=str(content["dataset_name_or_path"]),
            dataset_config_name=content.get("dataset_config_name"),
            dataset_revision=content.get("dataset_revision"),
            data=data,
            model=model,
            conditions=normalized_conditions,
            training=training,
            output_dir=content.get("output_dir"),
            logging=logging,
        )


def compose_training_config(recipe: TrainingRecipe) -> DictConfig:
    """Translate a public training recipe to PRISMA's internal Hydra config."""
    register_resolvers()
    pretrained = recipe.model.get("pretrained_model_name_or_path")
    config_name = "finetune" if pretrained else "default"
    overrides = [] if pretrained else [f"model/gnn={recipe.model['backbone']}"]
    with hydra.initialize_config_module(
        config_module="prisma.configs", version_base="1.3"
    ):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    cfg.expgroup = recipe.name
    cfg.expname = recipe.name
    _configure_dataset(cfg, recipe)
    _configure_model(cfg, recipe)
    _configure_training(cfg, recipe)
    _configure_logging(cfg, recipe)

    output_dir = Path(recipe.output_dir or f"runs/{recipe.name}").expanduser()
    cfg.training.trainer.default_root_dir = str(output_dir)
    return cfg


def apply_recipe_overrides(
    recipe: TrainingRecipe, overrides: list[str]
) -> TrainingRecipe:
    if not overrides:
        return recipe
    content = OmegaConf.structured(recipe)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override!r}")
        key, raw_value = override.split("=", 1)
        if not _has_path(content, key):
            raise ValueError(f"Unknown or unset override field: {key}")
        value = yaml.safe_load(raw_value)
        OmegaConf.update(content, key, value, merge=False)
    mapping = OmegaConf.to_container(content, resolve=True)
    return TrainingRecipe.from_mapping(mapping)


def _configure_dataset(cfg: DictConfig, recipe: TrainingRecipe) -> None:
    source = Path(recipe.dataset_name_or_path).expanduser()
    if source.exists():
        cfg.data.dataset_path = str(source)
        cfg.data.dataset_name = None
    else:
        cfg.data.dataset_name = recipe.dataset_name_or_path
        cfg.data.dataset_path = None
    cfg.data.dataset_subset = recipe.dataset_config_name
    cfg.data.revision = recipe.dataset_revision
    for key, value in recipe.data.items():
        cfg.data[key] = value


def _configure_model(cfg: DictConfig, recipe: TrainingRecipe) -> None:
    backbone = recipe.model["backbone"]
    pretrained = recipe.model.get("pretrained_model_name_or_path")
    freeze_except_new = recipe.model.get("freeze_except_new", False)

    if not pretrained:
        backbone_config = OmegaConf.create(
            OmegaConf.to_container(cfg.model.gnn, resolve=False)
        )
        cfg.model.gnn = OmegaConf.merge(
            backbone_config,
            _BACKBONE_DEFAULTS[backbone],
            recipe.model["config"],
        )
    else:
        cfg.model.gnn._target_ = f"{_BACKBONE_TARGETS[backbone]}.from_pretrained"
        for component in (cfg.model.cond_encoder, cfg.model.gnn, cfg.model.score_model):
            component.pretrained_model_name_or_path = pretrained
            component.freeze_except_new = freeze_except_new
        for scheduler in cfg.diffusion.values():
            scheduler.pretrained_model_name_or_path = pretrained
        backbone_config = OmegaConf.create(
            OmegaConf.to_container(cfg.model.gnn, resolve=False)
        )
        cfg.model.gnn = OmegaConf.merge(backbone_config, recipe.model["config"])

    cfg.model.cond_encoder.condition = {
        name: _condition_config(condition)
        for name, condition in recipe.conditions.items()
    }


def _condition_config(condition: Mapping[str, Any]) -> dict[str, Any]:
    public_type = condition["type"]
    encoding_type = "sinusoidal" if public_type == "scalar" else public_type
    config = {"condition_type": "adapter", "encoding_type": encoding_type}
    if public_type == "scalar":
        config["scale"] = condition.get("scale", True)
    elif "scale" in condition:
        config["scale"] = condition["scale"]
    for key in ("input_dim", "num_categories"):
        if key in condition:
            config[key] = condition[key]
    return config


def _configure_training(cfg: DictConfig, recipe: TrainingRecipe) -> None:
    training = recipe.training
    if not recipe.model.get("pretrained_model_name_or_path"):
        architecture_defaults = _BACKBONE_TRAINING_DEFAULTS[
            recipe.model["backbone"]
        ]
        cfg.training.optimization.batch_size = architecture_defaults["batch_size"]
        cfg.training.trainer.accumulate_grad_batches = architecture_defaults[
            "gradient_accumulation"
        ]
        if recipe.model["backbone"] == "pet":
            cfg.training.optimization.lr_scheduler.config.scheduler.patience = 30

    assignments = {
        "max_epochs": (cfg.training.trainer, "max_epochs"),
        "batch_size": (cfg.training.optimization, "batch_size"),
        "gradient_accumulation": (cfg.training.trainer, "accumulate_grad_batches"),
        "learning_rate": (cfg.training.optimization.optimizer, "lr"),
        "precision": (cfg.training.trainer, "precision"),
        "accelerator": (cfg.training.trainer, "accelerator"),
        "devices": (cfg.training.trainer, "devices"),
        "checkpoint_every_n_epochs": (cfg.training.checkpoints, "every_n_epochs"),
        "seed": (cfg.training.reproducibility, "random_seed"),
    }
    for public_name, (section, internal_name) in assignments.items():
        if public_name in training:
            section[internal_name] = training[public_name]
    if "weight_decay" in training:
        cfg.training.regularization.weight_decay = training["weight_decay"]
        if "weight_decay" in cfg.training.optimization.optimizer:
            cfg.training.optimization.optimizer.weight_decay = training["weight_decay"]
    cfg.training.from_checkpoint = training.get("resume_from_checkpoint")


def _configure_logging(cfg: DictConfig, recipe: TrainingRecipe) -> None:
    wandb = recipe.logging.get("wandb", False)
    if not wandb:
        with open_dict(cfg.logging):
            if "wandb" in cfg.logging:
                del cfg.logging["wandb"]
        return
    if not isinstance(wandb, Mapping):
        raise TypeError("logging.wandb must be false or a mapping.")
    cfg.logging.wandb = OmegaConf.merge(cfg.logging.wandb, dict(wandb))


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _reject_unknown(content: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(content) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in {name}: {', '.join(unknown)}")


def _validate_positive(
    content: Mapping[str, Any], key: str, *, allow_zero: bool = False
) -> None:
    if key not in content:
        return
    value = content[key]
    if (
        not isinstance(value, (int, float))
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{key} must be {qualifier}.")


def _has_path(config: DictConfig, path: str) -> bool:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, (DictConfig, Mapping)) or part not in current:
            return False
        current = current[part]
    return True
