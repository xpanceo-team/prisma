import datetime
import os
import warnings
from typing import Optional
import resource

import torch
from torch.nn.parameter import UninitializedParameter
from omegaconf import DictConfig, OmegaConf

import hydra

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from crystal_diffusers.training.datamodule import DataModule
from crystal_diffusers.training.callbacks import build_callbacks
from crystal_diffusers.utils.logging import logger
from crystal_diffusers.utils.resolvers import register_resolvers

# changes pytorch lightning logging format
# if it causes issues it can be safely deleted
from lightning.pytorch.trainer.connectors.logger_connector.result import _Metadata


_Metadata.forked_name = lambda self, on_step: (
    f'{self.name}/{"step" if on_step else "epoch"}'
)

def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    **kwargs,
) -> None:
    """This method controls which parameters from Hydra config are
    saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    def _safe_numel(parameter: torch.nn.Parameter) -> int:
        if isinstance(parameter, UninitializedParameter):
            return 0
        return parameter.numel()

    # save number of model parameters (skip uninitialized lazy params)
    hparams["other/params_total"] = sum(_safe_numel(p) for p in model.parameters())
    hparams["other/params_trainable"] = sum(
        _safe_numel(p) for p in model.parameters() if p.requires_grad
    )
    hparams["other/params_not_trainable"] = sum(
        _safe_numel(p) for p in model.parameters() if not p.requires_grad
    )

    for key, value in kwargs.items():
        hparams[f"other/{key}"] = value

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None


def run_training(cfg: DictConfig, ckpt_dir: str, ckpt_path: Optional[str] = None):
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /train_config
    :param ckpt_dir: path to checkpoint direcory
    :param ckpt_path: path to checkpoint
    """
    torch.set_float32_matmul_precision("medium")

    if cfg.expgroup == "test":
        print("------- checkpoint dir: -------")
        print(ckpt_dir)
        print("---------- cfg: ----------")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("--------------------------")

    if cfg.training.reproducibility.seed_everything:
        seed_everything(cfg.training.reproducibility.random_seed)

    logger.debug(f"Instantiating '{cfg.data._target_}'")
    datamodule: DataModule = hydra.utils.instantiate(cfg.data)

    datamodule.prepare_data()
    datamodule.setup("fit")
    train_batches = len(datamodule.train_dataloader())
    train_size = len(datamodule.train_dataset)
    val_batches = len(datamodule.val_dataloader())
    val_size = len(datamodule.valid_dataset)

    logger.debug(f"Instantiating '{cfg.training.module._target_}'")
    module: pl.LightningModule = hydra.utils.instantiate(
        cfg.training.module,
        condition_stats=datamodule.condition_stats,
    )
    # for weights only
    # module = TrainingModule.load_from_checkpoint(
    #     checkpoint_path=ckpt_path,
    #     **cfg.training.module
    # )

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        logger.debug("Instantiating WandbLogger")
        wandb_logger = WandbLogger(**cfg.logging.wandb)
        wandb_watch_log = cfg.logging.wandb_watch.log
        if wandb_watch_log:
            wandb_logger.watch(
                module,
                log=wandb_watch_log,
                log_freq=cfg.logging.wandb_watch.log_freq,
            )

    #  Determining number of steps between val checks
    val_check_every_n_steps_per_epoch = train_batches // cfg.logging.val_check_per_epoch

    # https://github.com/Lightning-AI/lightning/issues/12205
    # If val_check_interval will refer to global_steps, when uncomment this:
    # val_check_every_n_steps_per_epoch //= cfg.training.trainer.accumulate_grad_batches

    val_check_every_n_steps = cfg.logging.val_check_every_n_steps
    if not isinstance(val_check_every_n_steps, int):
        raise ValueError(f"Expected int, but got {type(val_check_every_n_steps)=}")

    val_check_interval = min(val_check_every_n_steps, val_check_every_n_steps_per_epoch)

    torch.multiprocessing.set_sharing_strategy("file_system")
    if cfg.training.trainer.devices == 1:
        strategy = "auto"
    else:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

        os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
        os.environ["NCCL_P2P_DISABLE"] = os.environ.get("NCCL_P2P_DISABLE", "1")
        strategy = hydra.utils.instantiate(
            cfg.training.strategy, timeout=datetime.timedelta(seconds=3600)
        )

    callbacks = build_callbacks(cfg=cfg, run_dir=ckpt_dir, datamodule=datamodule)
    logger.info(
        f"Added callbacks: " f"{', '.join(cb.__class__.__name__ for cb in callbacks)}"
    )

    # The Lightning core, the Trainer
    logger.debug("Instantiating the Trainer")
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        # val_check_interval=val_check_interval,
        check_val_every_n_epoch=1,
        detect_anomaly=False,
        strategy=strategy,
        **cfg.training.trainer,
    )
    log_hyperparameters(
        trainer=trainer,
        model=module,
        cfg=cfg,
        val_check_interval=val_check_interval,
        train_batches=train_batches,
        train_size=train_size,
        val_batches=val_batches,
        val_size=val_size,
    )

    logger.info("Starting training!")

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)

    datamodule.setup("test")
    if datamodule.test_dataset is None:
        if wandb_logger is not None:
            wandb_logger.experiment.finish()
        return

    logger.info("Starting testing on best checkpoints!")
    for checkpoint in trainer.checkpoint_callbacks:
        candidate_cfgs = []
        if isinstance(checkpoint, ModelCheckpoint) and checkpoint.monitor:
            for checkpoint_cfg in cfg.training.checkpoints.metric_checkpoints:
                if checkpoint.monitor == checkpoint_cfg.params.monitor:
                    candidate_cfgs.append(checkpoint_cfg)
        else:
            continue

        if len(candidate_cfgs) > 1:
            warnings.warn(
                f"Found more than 1 appropriate checkpoint configs "
                f"for testing with {checkpoint.monitor=}."
                f"Testing will be done with the first one."
            )

        elif len(candidate_cfgs) == 0:
            warnings.warn("No appropriate checkpoints found.")
        else:
            ckpt_path = checkpoint.best_model_path
            trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt_path)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
