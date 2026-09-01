from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from prisma.training.configuration import (
    TrainingRecipe,
    apply_recipe_overrides,
    compose_training_config,
)
from prisma.training.training_utils import run_training
from prisma.utils.logging import configure_logging


def build_parser(prog: str = "prisma train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train a PRISMA crystal diffusion model.",
    )
    parser.add_argument("config", help="Training configuration YAML file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a public configuration field. May be repeated.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration and exit without training.",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = "prisma train") -> None:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    try:
        recipe = TrainingRecipe.from_file(args.config)
        recipe = apply_recipe_overrides(recipe, args.overrides)
        cfg = compose_training_config(recipe)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return

    configure_logging(level="DEBUG")
    run_dir = Path(cfg.training.trainer.default_root_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "hparams.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    run_training(cfg, str(run_dir), cfg.training.from_checkpoint)
