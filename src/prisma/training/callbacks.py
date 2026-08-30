import hydra.utils
import torch
from omegaconf import DictConfig

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    Callback,
)
from pymatgen.core import Structure
import numpy as np

from prisma import MatterGenPipeline
from prisma.training.module import TrainingModule
from prisma.utils.logging import logger


class EmbeddingL2MetricCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int,
        datamodule,
        mace_checkpoint: str,
        batch_size=16,
        guidance_scale: float = 3.0,
        random_seed: int = 42,
    ):
        super().__init__()
        from mace.calculators import MACECalculator

        self.every_n = every_n_epochs
        self.random_seed = random_seed

        logger.debug("initializing MACE")
        dtype = str(torch.get_default_dtype()).replace("torch.", "")
        self.calculator = MACECalculator(
            model_paths=mace_checkpoint,
            device="cuda",
            default_dtype=dtype,
        )

        dataset = datamodule.valid_dataset
        previous_format = dataset.format

        dataset.reset_format()

        rng = np.random.default_rng(random_seed)
        indices = rng.choice(len(dataset), size=batch_size, replace=False)

        batch = dataset.select(indices)

        self.generation_params = {
            "batch_size": batch_size,
            "condition": {
                "atlas_embedding": batch["atlas_embedding"],
            },
            "num_atoms": batch["num_atoms"],
            "guidance_scale": guidance_scale,
        }
        self.mace_atomic_numbers = np.array(list(range(1, 84)) + [89, 90, 91, 92, 93, 94])

        dataset.set_format(**previous_format)


    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: TrainingModule):

        if trainer.current_epoch % self.every_n == 0:
            pipeline = MatterGenPipeline(
                gnn=pl_module.gnn,
                condition_encoder=pl_module.cond_encoder,
                score_model=pl_module.score_model,
                atomic_numbers_scheduler=pl_module.atomic_numbers_scheduler,
                frac_coords_scheduler=pl_module.frac_coords_scheduler,
                cell_scheduler=pl_module.cell_scheduler,
            )

            device = pl_module.device

            generator = torch.Generator(device=device).manual_seed(self.random_seed)

            try:
                structures = pipeline(
                    device=device,
                    generator=generator,
                    **self.generation_params,
                )
            except Exception as e:
                logger.error(f"Error in callback generation: {e}")
                return

            s_embs = []
            for s in structures:
                atomic_numbers = np.array(s.atomic_numbers)
                mace_numbers_mask = np.isin(atomic_numbers, self.mace_atomic_numbers)

                if not mace_numbers_mask.all():
                    # if some atomic numbers are not supported by MACE, change them to H
                    logger.debug("Changing unsupported atomic numbers to hydrogen to calculate MACE embeddings")
                    atomic_numbers = np.where(mace_numbers_mask, atomic_numbers, 1)
                    s = Structure(
                        lattice=s.lattice,
                        species=atomic_numbers,
                        coords=s.frac_coords,
                        coords_are_cartesian=False,
                    )

                structure_ase = s.to_ase_atoms()

                with torch.enable_grad():
                    descriptors = self.calculator.get_descriptors(structure_ase)

                emb = np.mean(descriptors, axis=0)

                s_embs.append(emb)

            generated_embs = np.vstack(s_embs)

            embs = np.array(self.generation_params["condition"]["atlas_embedding"])

            aed = (
                ((embs - generated_embs) ** 2).sum(-1) ** (1 / 2)
            ).mean()

            pl_module.log(
                "mean_embedding_l2_distance",
                aed,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
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
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir,
                auto_insert_metric_name=False,
                filename=f"epoch={{epoch:02d}}-step={{step:04d}}-"
                f"best={model_checkpoint.metric_name}",
                save_weights_only=False,
                **model_checkpoint.params,
            )
        )

        if "callback" in model_checkpoint:
            callback = hydra.utils.instantiate(model_checkpoint.callback)
            callbacks.append(
                callback(datamodule=datamodule)
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
