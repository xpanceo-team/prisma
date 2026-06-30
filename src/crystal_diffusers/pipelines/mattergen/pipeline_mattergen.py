import copy

import numpy as np
import torch
from pymatgen.core import Structure
from tqdm.auto import tqdm
from torch_geometric.data import Batch

from crystal_diffusers.pipelines.pipeline_utils import DiffusionPipeline
from crystal_diffusers.data import StructureData
from crystal_diffusers.data.data import sample_num_atoms, symmetrize_cell
import torch.nn as nn
from crystal_diffusers.models.mattergen.modeling_mattergen import MatterGenModel
from crystal_diffusers.models.cond_encoder import ConditionEncoder
from crystal_diffusers.schedulers import (
    D3PMScheduler,
    VariancePreservingScheduler,
    VarianceExplodingScheduler,
)


def _infer_len(obj, skip_tensor=False) -> int | None:
    """
    Return len(obj) if it is a (nested) list or a 1-D tensor, else None.
    """
    if obj is None:
        return None
    elif isinstance(obj, (list, tuple)):
        return len(obj)
    elif torch.is_tensor(obj):
        if not skip_tensor and obj.ndim >= 1:
            return obj.size(0)

    return None


class MatterGenPipeline(DiffusionPipeline):
    """
    Pipeline for crystal generation using MatterGen.
    Original paper: https://arxiv.org/abs/2312.03687
    """

    def __init__(
        self,
        gnn: nn.Module,
        score_model: MatterGenModel,
        condition_encoder: ConditionEncoder,
        atomic_numbers_scheduler: D3PMScheduler,
        frac_coords_scheduler: VarianceExplodingScheduler,
        cell_scheduler: VariancePreservingScheduler,
    ):
        super().__init__()

        self.register_modules(
            gnn=gnn,
            score_model=score_model,
            condition_encoder=condition_encoder,
            atomic_numbers_scheduler=atomic_numbers_scheduler,
            frac_coords_scheduler=frac_coords_scheduler,
            cell_scheduler=cell_scheduler,
        )

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int | None = None,
        num_atoms: int | list[int] | torch.Tensor | None = None,
        atomic_numbers: torch.Tensor | None = None,
        frac_coords: torch.Tensor | None = None,
        cell: torch.Tensor | None = None,
        condition: dict[str, any] | None = None,
        device: str | torch.device = "cpu",
        guidance_scale: float = 3.0,
        generator: torch.Generator | None = None,
        output_trajectories: bool = False,
    ):
        batch, condition = self.prepare_inputs(
            batch_size=batch_size,
            num_atoms=num_atoms,
            atomic_numbers=atomic_numbers,
            frac_coords=frac_coords,
            cell=cell,
            condition=condition,
            device=device,
            generator=generator,
        )

        self._guidance_scale = guidance_scale if len(condition) > 0 else 0.0

        self.gnn.to(device)
        self.score_model.to(device)
        self.condition_encoder.to(device)
        self.condition_encoder.eval()

        cur_atom_types = batch.atomic_numbers
        cur_frac_coords = batch.frac_coords
        cur_cell = batch.cell

        generated_list = [
            {
                "coords": cur_frac_coords.cpu().numpy(),
                "atom_types": cur_atom_types.cpu().numpy(),
                "lattice": cur_cell.cpu().numpy(),
                "t": "init",
            }
        ]

        num_timesteps = 1000
        for i in tqdm(reversed(range(num_timesteps)), leave=False, total=num_timesteps):
            t = torch.full(
                [batch.num_atoms.shape[0]], i, dtype=torch.long, device=device
            )

            # ------------------------------------------------------------------
            # Predictor
            # ------------------------------------------------------------------

            # TODO: condition pass before diffusion steps
            t_emb, added_cond, cond_mask = self.condition_encoder(condition, t)

            gnn_output = self.gnn(
                batch,
                t_emb=t_emb,
                added_cond=added_cond,
                cond_mask=cond_mask,
                output_edges_hidden_states=True,
            )
            score_model_output = self.score_model(batch, gnn_output)

            if self.do_classifier_free_guidance:
                gnn_output_uncond = self.gnn(
                    batch, t_emb=t_emb, output_edges_hidden_states=True
                )
                score_model_output_uncond = self.score_model(batch, gnn_output_uncond)

                score_model_output.frac_coords_score = (
                    score_model_output_uncond.frac_coords_score
                    + self.guidance_scale
                    * (
                        score_model_output.frac_coords_score
                        - score_model_output_uncond.frac_coords_score
                    )
                )
                score_model_output.lattice_score = (
                    score_model_output_uncond.lattice_score
                    + self.guidance_scale
                    * (
                        score_model_output.lattice_score
                        - score_model_output_uncond.lattice_score
                    )
                )
                score_model_output.atom_types_logits = (
                    score_model_output_uncond.atom_types_logits
                    + self.guidance_scale
                    * (
                        score_model_output.atom_types_logits
                        - score_model_output_uncond.atom_types_logits
                    )
                )

            if frac_coords is None:
                cur_frac_coords = self.frac_coords_scheduler.step(
                    model_output=score_model_output.frac_coords_score,
                    timestep=t,
                    sample=batch.frac_coords,
                    num_atoms=batch.num_atoms,
                    batch_idx=batch.batch,
                    corrector=False,
                    generator=generator,
                ).prev_sample

            if cell is None:
                cur_cell = self.cell_scheduler.step(
                    model_output=score_model_output.lattice_score,
                    timestep=t,
                    sample=batch.cell,
                    num_atoms=batch.num_atoms,
                    batch_idx=batch.batch,
                    corrector=False,
                    generator=generator,
                ).prev_sample

            if atomic_numbers is None:
                cur_atom_types = self.atomic_numbers_scheduler.step(
                    model_output=score_model_output.atom_types_logits,
                    timestep=t,
                    sample=batch.atomic_numbers,
                    num_atoms=batch.num_atoms,
                    batch_idx=batch.batch,
                    generator=generator,
                ).prev_sample

            batch.update_structure(
                atomic_numbers=cur_atom_types,
                frac_coords=cur_frac_coords,
                cell=cur_cell,
            )

            # ------------------------------------------------------------------
            # Corrector
            # ------------------------------------------------------------------

            gnn_output = self.gnn(
                batch,
                t_emb=t_emb,
                added_cond=added_cond,
                cond_mask=cond_mask,
                output_edges_hidden_states=True,
            )
            score_model_output = self.score_model(batch, gnn_output)

            if self.do_classifier_free_guidance:
                gnn_output_uncond = self.gnn(
                    batch, t_emb=t_emb, output_edges_hidden_states=True
                )
                score_model_output_uncond = self.score_model(batch, gnn_output_uncond)

                score_model_output.frac_coords_score = (
                    score_model_output_uncond.frac_coords_score
                    + self.guidance_scale
                    * (
                        score_model_output.frac_coords_score
                        - score_model_output_uncond.frac_coords_score
                    )
                )
                score_model_output.lattice_score = (
                    score_model_output_uncond.lattice_score
                    + self.guidance_scale
                    * (
                        score_model_output.lattice_score
                        - score_model_output_uncond.lattice_score
                    )
                )
                score_model_output.atom_types_logits = (
                    score_model_output_uncond.atom_types_logits
                    + self.guidance_scale
                    * (
                        score_model_output.atom_types_logits
                        - score_model_output_uncond.atom_types_logits
                    )
                )

            if frac_coords is None:
                cur_frac_coords = self.frac_coords_scheduler.step(
                    model_output=score_model_output.frac_coords_score,
                    timestep=t,
                    sample=batch.frac_coords,
                    num_atoms=batch.num_atoms,
                    batch_idx=batch.batch,
                    corrector=True,
                    generator=generator,
                ).prev_sample

            if cell is None:
                cur_cell = self.cell_scheduler.step(
                    model_output=score_model_output.lattice_score,
                    timestep=t,
                    sample=batch.cell,
                    num_atoms=batch.num_atoms,
                    batch_idx=batch.batch,
                    corrector=True,
                    generator=generator,
                ).prev_sample

            if output_trajectories:
                generated_list.append(
                    {
                        "coords": cur_frac_coords.cpu().numpy(),
                        "atom_types": cur_atom_types.cpu().numpy(),
                        "lattice": cur_cell.cpu().numpy(),
                        "t": i,
                    }
                )

            batch.update_structure(
                frac_coords=cur_frac_coords,
                cell=cur_cell,
            )

        batch.cpu()
        structures = []
        for batch_idx in range(len(batch)):
            mask = batch.batch == batch_idx

            structure = Structure(
                lattice=batch.cell[batch_idx],
                species=batch.atomic_numbers[mask],
                coords=batch.frac_coords[mask],
                coords_are_cartesian=False,
            )

            structures.append(structure)

        if output_trajectories:
            return structures, generated_list

        return structures

    def prepare_inputs(
        self,
        batch_size: int | None = None,
        num_atoms: int | list[int] | torch.Tensor | None = None,
        atomic_numbers: list[list[int]] | torch.Tensor | None = None,
        frac_coords: list[list[list[float]]] | torch.Tensor | None = None,
        cell: list[list[list[float]]] | torch.Tensor | None = None,
        condition: dict[str, any] | None = None,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ):
        # ------------------------------------------------------------------
        # 1. Determine batch_size (collect every candidate, then reconcile)
        # ------------------------------------------------------------------
        if atomic_numbers or frac_coords:
            if num_atoms is None:
                raise ValueError(
                    "If 'atomic_numbers' or 'frac_coords' are passed then "
                    "'num_atoms' must be specified."
                )

        candidate_sizes = [
            _infer_len(num_atoms),
            _infer_len(atomic_numbers, skip_tensor=True),
            _infer_len(frac_coords, skip_tensor=True),
            _infer_len(cell),
        ]
        candidate_sizes = [s for s in candidate_sizes if s is not None]

        if candidate_sizes:
            if len(set(candidate_sizes)) != 1:
                raise ValueError(
                    f"Inconsistent batch lengths {candidate_sizes},"
                    " cannot deduce batch_size."
                )

        if batch_size is None:
            if candidate_sizes:
                batch_size = candidate_sizes[0]
            else:
                batch_size = 1
        elif candidate_sizes and batch_size != candidate_sizes[0]:
            raise ValueError(
                f"Inconsistent batch lengths: input has {candidate_sizes[0]},"
                f" but specified {batch_size=}."
            )

        # ------------------------------------------------------------------
        # 2. num_atoms – ensure 1-D tensor of length batch_size
        # ------------------------------------------------------------------
        if num_atoms is None:
            num_atoms = sample_num_atoms(batch_size, generator=generator)
        else:
            if isinstance(num_atoms, int):
                num_atoms = torch.tensor([num_atoms] * batch_size)
            elif isinstance(num_atoms, (list, tuple)):
                num_atoms = torch.tensor(num_atoms, dtype=torch.long)
            elif torch.is_tensor(num_atoms):
                num_atoms = num_atoms.to(dtype=torch.long)
            else:
                raise TypeError("'num_atoms' must be int, list, or Tensor.")

        if num_atoms.ndim != 1 or num_atoms.size(0) != batch_size:
            raise ValueError(
                f"'num_atoms' must be 1-D with length {batch_size}, "
                f"got shape {tuple(num_atoms.shape)}."
            )

        # ------------------------------------------------------------------
        # 3. atomic_numbers – flatten → 1-D tensor or keep None
        # ------------------------------------------------------------------
        if atomic_numbers is not None:
            if isinstance(atomic_numbers, (list, tuple)):
                flat = [z for sub in atomic_numbers for z in sub]
                atomic_numbers = torch.tensor(flat, dtype=torch.long)
            elif torch.is_tensor(atomic_numbers):
                atomic_numbers = atomic_numbers.to(dtype=torch.long)
            else:
                raise TypeError("atomic_numbers must be list of lists or Tensor.")

            if atomic_numbers.ndim != 1:
                raise ValueError("`atomic_numbers` must be 1-D after flattening.")

            if atomic_numbers.size(0) != num_atoms.sum().item():
                raise ValueError(
                    "`atomic_numbers` length "
                    f"({atomic_numbers.size(0)}) does not equal "
                    f"sum(num_atoms) ({num_atoms.sum().item()})."
                )

        # ------------------------------------------------------------------
        # 4. frac_coords – flatten rows → [N, 3] tensor or keep None
        # ------------------------------------------------------------------
        if frac_coords is not None:
            if isinstance(frac_coords, (list, tuple)):
                flat = [c for sub in frac_coords for c in sub]
                frac_coords = torch.tensor(flat, dtype=torch.float32)
            elif torch.is_tensor(frac_coords):
                frac_coords = frac_coords.to(dtype=torch.float32)
            else:
                raise TypeError("frac_coords must be list of lists or Tensor.")

            if frac_coords.ndim != 2 or frac_coords.size(1) != 3:
                raise ValueError("`frac_coords` must have shape [N, 3].")

            if frac_coords.size(0) != num_atoms.sum().item():
                raise ValueError(
                    "`frac_coords` rows "
                    f"({frac_coords.size(0)}) do not equal "
                    f"sum(num_atoms) ({num_atoms.sum().item()})."
                )

        # ------------------------------------------------------------------
        # 5. cell – [batch_size, 3, 3] tensor with symmetrisation
        # ------------------------------------------------------------------
        if cell is not None:
            if isinstance(cell, (list, tuple)):
                cell_list = cell
            elif torch.is_tensor(cell):
                cell_list = cell.cpu().numpy()
            else:
                raise TypeError("cell must be list of lists or Tensor.")

            if len(cell_list) != batch_size:
                raise ValueError(
                    f"'cell' list length {len(cell_list)} "
                    f"does not match batch_size={batch_size}."
                )

            # symmetrise each cell and stack
            sym_cells = [
                torch.tensor(symmetrize_cell(np.asarray(c)), dtype=torch.float32)
                for c in cell_list
            ]
            cell = torch.stack(sym_cells, dim=0)

        # ------------------------------------------------------------------
        # 6. Condition
        # ------------------------------------------------------------------

        condition = copy.deepcopy(condition)
        if condition is None:
            condition = {}
        else:
            for cond_name, cond_value in condition.items():
                if len(cond_value) != batch_size:
                    raise ValueError(
                        f"Condition must be a dict with values as lists or tensors of "
                        f"batch size length {batch_size}, got length "
                        f"{len(cond_value)} for {cond_name} condition."
                    )

                if isinstance(cond_value, (list, tuple)):
                    if torch.is_tensor(cond_value[0]):
                        condition[cond_name] = torch.stack(cond_value, 0).to(device)
                    elif isinstance(cond_value[0], str):
                        pass
                    else:
                        condition[cond_name] = torch.tensor(cond_value).to(device)
                elif torch.is_tensor(cond_value):
                    condition[cond_name] = cond_value.to(device)
                else:
                    raise TypeError("condition values must be list of lists or Tensor.")

        # ------------------------------------------------------------------
        # 7. Move to device & return
        # ------------------------------------------------------------------

        num_atoms = num_atoms.to(device)
        if atomic_numbers is None:
            atomic_numbers = self.atomic_numbers_scheduler.sample_prior(
                num_atoms=num_atoms,
            )

        if frac_coords is None:
            frac_coords = self.frac_coords_scheduler.sample_prior(
                num_atoms=num_atoms,
                generator=generator,
            )

        if cell is None:
            cell = self.cell_scheduler.sample_prior(
                num_atoms=num_atoms,
                generator=generator,
            )

        batch = Batch.from_data_list([StructureData(num_nodes=n) for n in num_atoms])

        batch.update_structure(
            atomic_numbers=atomic_numbers,
            frac_coords=frac_coords,
            cell=cell,
            num_atoms=num_atoms,
        )

        batch = batch.to(device)

        return batch, condition

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return (
            self._guidance_scale > 1
            and len(self.condition_encoder.condition_encdoings) > 0
        )
