import copy
from omegaconf import OmegaConf
from typing import Dict, Optional, Union, TYPE_CHECKING

import hydra
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer

from prisma.data import StructureData
from prisma.models.mattergen.modeling_mattergen import MatterGenOutput
from prisma.utils.logging import logger

if TYPE_CHECKING:
    from prisma.models.cond_encoder import ConditionEncoder


def freeze_parameters(model: nn.Module, skip: list[str] | None = None) -> None:
    if skip is None:
        skip = []

    for p in model.parameters():
        p.requires_grad = False

    param_dict = dict(model.named_parameters())

    unfrozen = []
    for key in skip:
        if key not in param_dict:
            continue
        param_dict[key].requires_grad = True
        unfrozen.append(key)

    if len(unfrozen):
        logger.info(
            f"{model.__class__.__name__}: "
            f"training {len(unfrozen)} tensors, total {len(param_dict)}."
        )


def initialize_weights_from_new_instance(instance, keys):
    logger.debug(f"Initializing weights: {keys}")

    instance_from_config = instance.__class__.from_config(instance.config)

    missing_state_dict = {
        k: v for k, v in instance_from_config.state_dict().items() if k in keys
    }

    # Could be some problems with tied weights!
    instance.load_state_dict(missing_state_dict, assign=True, strict=False)


def instantiate_from_pretrained(module, *args, **kwargs):
    logger.debug(f"Instantiating {module._target_}")

    check_missing_weights = module._target_.endswith("from_pretrained")
    if check_missing_weights:
        kwargs["output_loading_info"] = True

    freeze_except_new = kwargs.pop("freeze_except_new", False) or module.get(
        "freeze_except_new", False
    )

    instance = hydra.utils.instantiate(
        module,
        *args,
        **kwargs,
    )

    if check_missing_weights and isinstance(instance, tuple):
        instance, loading_info = instance

        missing_keys = loading_info["missing_keys"]
        if len(missing_keys) > 0:
            initialize_weights_from_new_instance(instance, missing_keys)

            if freeze_except_new:
                freeze_parameters(instance, skip=missing_keys)

    return instance


def _get_target_factory_name(target: str) -> str:
    parts = [part for part in target.split(".") if part]
    if len(parts) >= 2 and parts[-1] == "from_pretrained":
        return parts[-2]
    return parts[-1] if parts else ""


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        *args,
        condition_stats: dict[str, dict[str, list[str]]],
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        logger.debug(f"{condition_stats=}")
        self.cond_encoder: ConditionEncoder = instantiate_from_pretrained(
            self.hparams.model.cond_encoder,
            condition=condition_stats,
        )

        # TODO: rename attributes
        condition_adapter_keys = [
            k
            for k in self.cond_encoder.condition_keys
            if self.cond_encoder.config.condition[k]["condition_type"] == "adapter"
        ]

        gnn_kwargs = {
            "t_emb_dim": self.cond_encoder.config.t_emb_dim,
            "condition_keys": condition_adapter_keys,
        }
        gnn_target = self.hparams.model.gnn.get("_target_", "")
        gnn_target_name = _get_target_factory_name(gnn_target)
        if gnn_target_name in {"PETWrapper", "PETMADWrapper", "EquiformerV2Wrapper"}:
            gnn_kwargs["condition_dim"] = self.cond_encoder.config.condition_dim

        self.gnn = instantiate_from_pretrained(
            self.hparams.model.gnn,
            **gnn_kwargs,
        )

        self.score_model = instantiate_from_pretrained(
            self.hparams.model.score_model,
        )

        self.atomic_numbers_scheduler = instantiate_from_pretrained(
            self.hparams.diffusion.atomic_numbers_scheduler,
        )

        self.frac_coords_scheduler = instantiate_from_pretrained(
            self.hparams.diffusion.frac_coords_scheduler,
        )

        self.cell_scheduler = instantiate_from_pretrained(
            self.hparams.diffusion.cell_scheduler,
        )

        self._problem_batch = False

    def on_fit_start(self) -> None:
        pass

    def step(
        self,
        batch: dict[str, any],
        batch_idx: Optional[int] = None,
        dataloader_idx: int = 0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        self._problem_batch = False

        try:
            losses = self._step(batch, batch_idx, dataloader_idx)
        except Exception as e:
            logger.error(
                f"Skipping batch due to error at epoch {self.current_epoch}: {e}"
            )

            # continue training
            # raise e
            self._problem_batch = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            losses = None

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            losses = self._skip_if_problem_batch_on_any_device(losses)

        return losses

    def _skip_if_problem_batch_on_any_device(self, losses):
        if losses is None:
            flag_skip = torch.ones((), device=self.device, dtype=torch.bool)
        else:
            flag_skip = torch.zeros((), device=self.device, dtype=torch.bool)

        # print(f"{flag_skip=}")
        # suboptimal but will do,
        # till they fix it in https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1552650013
        world_size = torch.distributed.get_world_size()
        torch.distributed.barrier()
        # now gather
        result = [torch.zeros_like(flag_skip) for _ in range(world_size)]
        torch.distributed.all_gather(result, flag_skip)
        any_invalid = torch.sum(torch.stack(result)).bool().item()

        if any_invalid:
            self._problem_batch = True
            return None
        else:
            return losses

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if self._problem_batch:
            return

        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def optimizer_step(self, *args, **kwargs):
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                valid_gradients = not (torch.isnan(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(
                "detected inf or nan values in gradients. not updating model parameters"
            )
            self.zero_grad()

        super().optimizer_step(*args, **kwargs)

    def _step(
        self,
        batch: dict[str, any],
        batch_idx: Optional[int] = None,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        num_timesteps = 1000

        data = batch["data"]
        timesteps = torch.randint(
            low=0,
            high=num_timesteps,
            size=(len(data),),
            dtype=torch.int32,
            device=self.device,
        )

        noised_atomic_numbers = self.atomic_numbers_scheduler.add_noise(
            original_samples=data.atomic_numbers,
            num_atoms=data.num_atoms,
            timesteps=timesteps,
        )

        noised_frac_coords = self.frac_coords_scheduler.add_noise(
            original_samples=data.frac_coords,
            num_atoms=data.num_atoms,
            timesteps=timesteps,
        )

        noised_cell = self.cell_scheduler.add_noise(
            original_samples=data.cell,
            num_atoms=data.num_atoms,
            timesteps=timesteps,
        )

        noised_data = copy.copy(data)
        noised_data.update_structure(
            atomic_numbers=noised_atomic_numbers,
            frac_coords=noised_frac_coords,
            cell=noised_cell,
        )

        t_emb, added_cond, cond_mask = self.cond_encoder(batch["condition"], timesteps)
        gnn_output = self.gnn(
            noised_data,
            t_emb=t_emb,
            added_cond=added_cond,
            cond_mask=cond_mask,
            output_edges_hidden_states=True,
        )
        score_model_output = self.score_model(noised_data, gnn_output)

        losses = self.get_loss(data, noised_data, score_model_output, timesteps)

        return losses

    def get_loss(
        self,
        data: StructureData,
        noised_data: StructureData,
        score_model_output: MatterGenOutput,
        timesteps: torch.Tensor,
    ):
        losses = {}

        (
            atom_types_loss,
            types_vb_loss,
            types_ce_loss,
            types_accuracy,
        ) = self.atomic_numbers_scheduler.get_loss(
            model_output=score_model_output.atom_types_logits,
            timestep=timesteps,
            sample=data.atomic_numbers,
            noised_sample=noised_data.atomic_numbers,
            num_atoms=data.num_atoms,
            batch_idx=data.batch,
        )
        losses.update(
            atom_types_loss=atom_types_loss,
            types_vb_loss=types_vb_loss,
            types_ce_loss=types_ce_loss,
            types_accuracy=types_accuracy,
        )

        coords_loss = self.frac_coords_scheduler.get_loss(
            model_output=score_model_output.frac_coords_score,
            timestep=timesteps,
            sample=data.frac_coords,
            noised_sample=noised_data.frac_coords,
            num_atoms=data.num_atoms,
            batch_idx=data.batch,
        )
        losses.update(
            coords_loss=coords_loss,
        )

        lattice_loss = self.cell_scheduler.get_loss(
            model_output=score_model_output.lattice_score,
            timestep=timesteps,
            sample=data.cell,
            noised_sample=noised_data.cell,
            num_atoms=data.num_atoms,
            batch_idx=data.batch,
        )
        losses.update(
            lattice_loss=lattice_loss,
        )

        loss = (
            losses.get("atom_types_loss", 0)
            + 0.1 * losses.get("coords_loss", 0)
            + losses.get("lattice_loss", 0)
        )

        losses.update(
            loss=loss,
        )

        return losses

    def training_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        outputs = self.step(batch, batch_idx, dataloader_idx)

        if outputs is None:
            return None

        outputs = {f"{key}/train": value for key, value in outputs.items()}

        self.log_dict(
            outputs,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch),
            sync_dist=True,
        )

        return outputs["loss/train"]

    def validation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        outputs = self.step(batch, batch_idx, dataloader_idx)

        if outputs is None:
            return None

        outputs = {f"{key}/val": value for key, value in outputs.items()}

        self.log_dict(
            outputs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch),
            sync_dist=True,
        )

        return outputs["loss/val"]

    def test_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        outputs = self.step(batch, batch_idx, dataloader_idx)

        if outputs is None:
            return None

        outputs = {f"{key}/test": value for key, value in outputs.items()}

        self.log_dict(
            outputs,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch),
            sync_dist=True,
        )

        return outputs["loss/test"]

    def get_test_prefix(self, dataset_name=""):
        return self._test_prefix_format.format(dataset_name=dataset_name)

    def get_test_postfix(self, dataset_name=""):
        return self._test_postfix_format.format(dataset_name=dataset_name)

    def test_dataset_name(self, dataloader_idx: int = 0):
        datamodule = getattr(self.trainer, "datamodule")
        if datamodule is None:
            raise ValueError("Datamodule is not passted to pl.Trainer")

        name = datamodule.test_datasets[dataloader_idx].path

        return name

    def configure_optimizers(
        self,
    ) -> dict[str, Optimizer | object]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might
        have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers,
        the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        logger.debug(f"Instantiating '{self.hparams.optimization.optimizer._target_}'")
        opt = hydra.utils.instantiate(
            self.hparams.optimization.optimizer,
            params=[p for p in self.parameters() if p.requires_grad],
            _convert_="partial",
        )
        if not self.hparams.optimization.lr_scheduler.use_lr_scheduler:
            return {"optimizer": opt}

        logger.debug(
            f"Instantiating "
            f"'{self.hparams.optimization.lr_scheduler.config.scheduler._target_}'"
        )
        lr_scheduler_cfg = OmegaConf.to_container(
            self.hparams.optimization.lr_scheduler.config,
            resolve=True,
        )
        scheduler_cfg = lr_scheduler_cfg.pop("scheduler")
        lr_scheduler_config = {
            "scheduler": hydra.utils.instantiate(
                scheduler_cfg,
                optimizer=opt,
            ),
            **lr_scheduler_cfg,
        }

        return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
