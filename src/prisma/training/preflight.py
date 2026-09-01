from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import torch

from prisma.training.datamodule import DataModule
from prisma.utils.logging import logger


class _StopAfterFirstBatch(pl.Callback):
    def __init__(self) -> None:
        self.completed = False
        self.loss: float | None = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.completed = True
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            self.loss = float(loss.detach().cpu())
        trainer.should_stop = True


@dataclass
class PreflightReport:
    timestamp: str
    batch_size: int
    total_atoms: int
    max_atoms: int
    loss: float | None
    peak_gpu_memory_bytes: int | None


def instantiate_training_module(
    cfg: DictConfig,
    datamodule: DataModule,
) -> pl.LightningModule:
    return hydra.utils.instantiate(
        cfg.training.module,
        condition_stats=datamodule.condition_stats,
    )


def run_preflight(
    cfg: DictConfig,
    datamodule: DataModule,
    *,
    ckpt_path: str | None,
    run_dir: str | Path,
) -> PreflightReport:
    """Run one isolated optimizer step using a conservative training batch."""
    dataloader = datamodule.preflight_dataloader()
    batch = next(iter(dataloader))
    num_atoms = batch["data"].num_atoms
    report = PreflightReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        batch_size=len(num_atoms),
        total_atoms=int(num_atoms.sum()),
        max_atoms=int(num_atoms.max()),
        loss=None,
        peak_gpu_memory_bytes=None,
    )

    module = instantiate_training_module(cfg, datamodule)
    _log_summary(cfg, datamodule, module, report)

    uses_cuda = _uses_cuda(cfg)
    if uses_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    stop_after_batch = _StopAfterFirstBatch()
    trainer = pl.Trainer(
        accelerator=cfg.training.trainer.accelerator,
        devices=cfg.training.trainer.devices,
        precision=cfg.training.trainer.precision,
        strategy=_instantiate_strategy(cfg),
        max_epochs=cfg.training.trainer.max_epochs,
        accumulate_grad_batches=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[stop_after_batch],
    )

    try:
        trainer.fit(module, train_dataloaders=dataloader, ckpt_path=ckpt_path)
        if not stop_after_batch.completed:
            raise RuntimeError(
                "Preflight did not execute a training batch. Check whether the "
                "resumed checkpoint has already reached training.max_epochs."
            )
        report.loss = stop_after_batch.loss
        if uses_cuda:
            report.peak_gpu_memory_bytes = torch.cuda.max_memory_allocated()
        if trainer.is_global_zero:
            OmegaConf.save(asdict(report), Path(run_dir) / "preflight.yaml")
        logger.info("Preflight passed: forward, backward, and optimizer step succeeded.")
        return report
    except torch.cuda.OutOfMemoryError as exc:
        exc.add_note(
            "Training preflight failed with "
            f"batch_size={report.batch_size}, max_atoms={report.max_atoms}, "
            f"precision={cfg.training.trainer.precision}. Reduce training.batch_size "
            "and increase training.gradient_accumulation if you need to preserve "
            "the effective batch size."
        )
        raise
    finally:
        del trainer
        del module
        gc.collect()
        if uses_cuda:
            torch.cuda.empty_cache()


def _instantiate_strategy(cfg: DictConfig) -> str | pl.strategies.Strategy:
    if cfg.training.trainer.devices == 1:
        return "auto"
    return hydra.utils.instantiate(cfg.training.strategy)


def _uses_cuda(cfg: DictConfig) -> bool:
    accelerator = str(cfg.training.trainer.accelerator).lower()
    return torch.cuda.is_available() and accelerator in {"auto", "cuda", "gpu"}


def _log_summary(
    cfg: DictConfig,
    datamodule: DataModule,
    module: pl.LightningModule,
    report: PreflightReport,
) -> None:
    dataset_source = datamodule.dataset_path or datamodule.dataset_name
    conditions = ", ".join(datamodule.condition) or "none"
    device = str(cfg.training.trainer.accelerator)
    if _uses_cuda(cfg):
        properties = torch.cuda.get_device_properties(0)
        device = f"{properties.name} ({properties.total_memory / 2**30:.1f} GiB)"
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    accumulation = int(cfg.training.trainer.accumulate_grad_batches)
    devices = int(cfg.training.trainer.devices)
    effective_batch = int(cfg.data.batch_size) * accumulation * devices
    logger.info(
        "Training preflight\n"
        f"  dataset: {dataset_source}\n"
        f"  splits: train={len(datamodule.train_dataset)}, "
        f"validation={len(datamodule.valid_dataset)}\n"
        f"  conditions: {conditions}\n"
        f"  atom limit: {datamodule.max_num_atoms}\n"
        f"  device: {device}\n"
        f"  precision: {cfg.training.trainer.precision}\n"
        f"  batch: {cfg.data.batch_size} x {accumulation} accumulation "
        f"x {devices} devices = {effective_batch} effective\n"
        f"  preflight batch: {report.batch_size} structures, "
        f"{report.total_atoms} atoms, max {report.max_atoms}\n"
        f"  trainable parameters: {trainable:,}"
    )
