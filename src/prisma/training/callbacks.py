import hydra.utils
from omegaconf import DictConfig

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    Callback,
)
from prisma.training.embedding_validation import EmbeddingValidationCallback


class EmbeddingL2MetricCallback(EmbeddingValidationCallback):
    """Compatibility with legacy MACE Hydra callback configurations."""

    def __init__(
        self,
        every_n_epochs,
        datamodule,
        mace_checkpoint,
        batch_size=16,
        guidance_scale=3.0,
        random_seed=42,
    ):
        super().__init__(
            datamodule=datamodule,
            extractor="mace",
            checkpoint=mace_checkpoint,
            condition="atlas_embedding",
            every_n_epochs=every_n_epochs,
            sample_size=batch_size,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            seed=random_seed,
        )


def build_callbacks(cfg: DictConfig, run_dir, datamodule) -> list[Callback]:
    callbacks = []

    if "lr_monitor" in cfg.logging:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "change_lr" in cfg.training and cfg.training.change_lr is not None:
        callbacks.append(ChangeLearningRateCallback(new_lr=cfg.training.change_lr))

    if "early_stopping" in cfg.training.optimization:
        callbacks.append(EarlyStopping(**cfg.training.regularization.early_stopping))

    for model_checkpoint in cfg.training.checkpoints.metric_checkpoints:
        checkpoint_params = dict(model_checkpoint.params)
        if "callback" in model_checkpoint:
            callback = hydra.utils.instantiate(model_checkpoint.callback)
            callback = callback(datamodule=datamodule)
            callbacks.append(callback)
            if isinstance(callback, EmbeddingValidationCallback):
                # Also align legacy Hydra configurations with the evaluation schedule.
                checkpoint_params["every_n_epochs"] = callback.every_n_epochs
                checkpoint_params["save_on_train_epoch_end"] = False

        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                auto_insert_metric_name=False,
                filename=f"epoch={{epoch:02d}}-step={{step:04d}}-"
                f"best={model_checkpoint.metric_name}",
                save_weights_only=False,
                **checkpoint_params,
            )
        )

    every_n = getattr(cfg.training.checkpoints, "every_n_epochs", 0)
    if every_n > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                auto_insert_metric_name=False,
                filename="epoch={epoch:02d}-step={step:04d}",
                monitor=None,
                save_on_train_epoch_end=True,
                every_n_epochs=every_n,
                save_top_k=-1,
                save_weights_only=True,
            )
        )

    if cfg.training.checkpoints.save_last:
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                auto_insert_metric_name=False,
                filename="epoch={epoch:02d}-step={step:04d}-last",
                save_last=False,
            )
        )

    callbacks.append(
        TQDMProgressBar(refresh_rate=cfg.logging.progress_bar_refresh_rate)
    )

    return callbacks


class ChangeLearningRateCallback(Callback):
    def __init__(self, new_lr):
        self.new_lr = new_lr

    def on_train_start(self, trainer, pl_module):
        lightning_optimizer = pl_module.optimizers()
        for param_group in lightning_optimizer.optimizer.param_groups:
            old_lr = param_group["lr"]
            param_group["lr"] = self.new_lr
            print(f"Learning rate changed from {old_lr} to {self.new_lr}")
