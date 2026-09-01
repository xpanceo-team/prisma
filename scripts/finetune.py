from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from prisma.training.training_utils import run_training
from prisma.utils.logging import configure_logging


@hydra.main(
    config_path=str("../src/prisma/configs"),
    config_name="finetune",
    version_base="1.3",
)
def main(cfg: DictConfig):
    configure_logging(level="DEBUG")

    # Hydra run directory
    ckpt_dir = Path(HydraConfig.get().run.dir)

    # Store the YaML config separately into the hydra dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (ckpt_dir / "hparams.yaml").write_text(yaml_conf)

    ckpt_path = cfg.training.from_checkpoint

    run_training(cfg, ckpt_dir, ckpt_path)


if __name__ == "__main__":
    main()
