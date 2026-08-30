from pathlib import Path
import argparse

import hydra

from prisma.training.training_utils import run_training
from prisma.utils.logging import configure_logging


def resume_run(ckpt_path, wandb_run_id=None) -> None:
    configure_logging(level="DEBUG")

    ckpt_path = Path(ckpt_path)

    ckpt_dir = ckpt_path.parent

    ckpt_dir = str(ckpt_dir.resolve())
    with hydra.initialize_config_dir(ckpt_dir, version_base="1.3"):
        cfg = hydra.compose(config_name="hparams")

    cfg.training.trainer.default_root_dir = ckpt_dir

    cfg.logging.wandb.resume = "must"

    if wandb_run_id is not None:
        cfg.logging.wandb.id = wandb_run_id

        if "wandb" not in cfg.logging:
            raise ValueError("`wandb_run_id` is specified without wandb config.")

    run_training(cfg, ckpt_dir, ckpt_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument("--run_id", type=str, required=True, help="W&B run ID")

    args = parser.parse_args()

    resume_run(ckpt_path=args.ckpt_path, wandb_run_id=args.run_id)


if __name__ == "__main__":
    main()
