import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
from ase import Atoms
from datasets import Dataset, concatenate_datasets
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# prevent mattersim from chaning logging behavior
# it must be the first import of mattersim
from loguru import logger as loguru_logger
import mattersim.utils.logger_utils as mu
import tqdm

mu.get_logger = lambda: loguru_logger

from mattersim.applications.batch_relax import BatchRelaxer  # noqa: E402
from mattersim.forcefield import Potential  # noqa: E402

from crystal_diffusers.utils.logging import logger


@contextmanager
def no_tqdm():
    real_tqdm = tqdm.tqdm
    def silent_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return real_tqdm(*args, **kwargs)
    tqdm.tqdm = silent_tqdm
    try:
        yield
    finally:
        tqdm.tqdm = real_tqdm


class MattersimBatchRelaxer(BatchRelaxer):
    def relax(
        self,
        atoms_list: list[Atoms],
    ) -> dict[int, list[Atoms]]:
        self.trajectories = {}
        pointer = 0
        atoms_list_ = []
        for i in range(len(atoms_list)):
            atoms_list_.append(atoms_list[i].copy())
            atoms_list_[i].info["structure_index"] = i

        while (
                pointer < len(atoms_list) or not self.finished
        ):  # While there are unfinished instances or atoms left to insert
            while pointer < len(atoms_list) and (
                    sum([len(opt.atoms) for opt in self.optimizer_instances])
                    + len(atoms_list[pointer])
                    <= self.max_natoms_per_batch
            ):
                # While there are enough n_atoms slots in the
                # batch and we have not reached the end of the list.
                self.insert(
                    atoms_list_[pointer]
                )  # Insert new structure to fire instances
                pointer += 1
            self.step_batch()

        return self.trajectories


class Relaxer(ABC):
    @abstractmethod
    def relax(self, structures: list[Structure]) -> tuple[list[Structure], np.ndarray]:
        raise NotImplementedError


class MattersimRelaxer(Relaxer):
    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:
        """
        Args:
            checkpoint (str): Path to the MatterSim checkpoint file.
            device (str): Device to use ('cuda' or 'cpu').
        """

        potential = Potential.from_checkpoint(
            device=device, load_path=checkpoint, load_training_state=False
        )
        self.max_z = potential.model.max_z

        self._relaxer = MattersimBatchRelaxer(potential=potential, filter="EXPCELLFILTER", **kwargs)

    def relax(self, structures: list[Structure]) -> tuple[list[Structure], np.ndarray]:
        """
        Relax structures using MatterSim and return
        relaxed structures and their energies.

        Args:
            structures (list[Structure]): List of pymatgen.Structure objects.

        Returns:
            relaxed_structures (list[Structure]): Relaxed structures.
            total_energies (np.ndarray): Relaxed total energies (in eV).
        """

        # Convert pymatgen.Structure -> ASE Atoms
        atoms_list = []
        for s in structures:
            s_max_z = max([el.Z for el in s.elements])
            if s_max_z > self.max_z:
                raise ValueError(
                    f"Can't relax structure with maximum element number {s_max_z}. "
                    f"MatterSim supports only elements with number below or equal {self.max_z}."
                )

            atoms_list.append(s.to_ase_atoms())

        # Relax structures
        logger.debug(f"Relaxing {len(structures)} structures...")
        relaxation_trajectories = self._relaxer.relax(atoms_list)
        logger.debug(f"Relaxing {len(structures)} structures done!")

        # Extract final relaxed atoms & energies
        relaxed_atoms = [traj[-1] for traj in relaxation_trajectories.values()]
        relaxed_structures = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]

        # convert to cif and back to get normal python types in Structure
        # without this it wouldn't be possible
        # to convert to json as there are numpy types
        relaxed_structures = [
            Structure.from_str(s.to(fmt="cif"), fmt="cif") for s in relaxed_structures
        ]
        total_energies = np.array([a.info["total_energy"] for a in relaxed_atoms])

        return relaxed_structures, total_energies


def relax_structures(examples, model_path, device, batch_size):
    structures = [Structure.from_str(s, fmt="json") for s in examples["structure"]]

    relaxer = MattersimRelaxer(
        checkpoint=model_path,
        device=device,
        max_natoms_per_batch=batch_size,
        max_n_steps=5000,
    )

    try:
        relaxed_structures, energies = relaxer.relax(structures)
    except Exception as e:
        logger.warning(f"Skipping relaxing {len(structures)} structures: {e}")

        return {"relaxed_structure": [None] * len(structures), "relaxed_energy": [None] * len(structures)}

    return {
        "relaxed_structure": [s.to(fmt="json") for s in relaxed_structures],
        "relaxed_energy": energies.tolist()
    }


def relax_f(
    examples,
    device,
    relaxation_batch_size,
    checkpoint_path: str,
    structure_column: str = "structure",
    relaxed_structure_column: str = "relaxed_structure",
    relaxed_energy_column: str = "relaxed_energy",
    **relaxer_params
):
    structures = [Structure.from_str(s, fmt="json") for s in examples[structure_column]]

    relaxer = MattersimRelaxer(
        checkpoint=checkpoint_path,
        device=device,
        max_natoms_per_batch=relaxation_batch_size,
        **relaxer_params,
    )

    try:
        relaxed_structures, energies = relaxer.relax(structures)
    except Exception as e:
        print(f"Skipping relaxing {len(structures)} structures: {e}")
        return {
            relaxed_structure_column: [None] * len(structures),
            relaxed_energy_column: [None] * len(structures)
        }

    return {
        relaxed_structure_column: [s.to(fmt="json") for s in relaxed_structures],
        relaxed_energy_column: energies.tolist()
    }


def check_structure_for_relaxation(x, structure_column):
    s = Structure.from_str(x[structure_column], fmt="json")

    mattersim_max_z = 94
    if max([el.Z for el in s.elements]) > mattersim_max_z:
        return False

    return True


def relax_dataset(
    ds,
    device="cuda",
    relaxation_batch_size=2048,
    checkpoint_path: str = "/home/sankek/models/mattersim/mattersim-v1.0.0-5M.pth",
    structure_column: str = "structure",
    relaxed_structure_column: str = "relaxed_structure",
    relaxed_energy_column: str = "relaxed_energy",
    num_proc: int | None = None,
    relaxer_params: dict | None = None,
):
    if relaxer_params is None:
        relaxer_params = {"max_n_steps": 5000}

    # Save the initial ordering once.
    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))
    else:
        ds = ds.remove_columns("idx").add_column("idx", list(range(len(ds))))

    if relaxed_structure_column not in ds.column_names:
        ds = ds.add_column(relaxed_structure_column, [None] * len(ds))

    if relaxed_energy_column not in ds.column_names:
        ds = ds.add_column(relaxed_energy_column, [None] * len(ds))

    # Keep only rows that still need relaxation.
    need_relax = ds.filter(
        lambda x: (
            (x[relaxed_structure_column] is None)
            or (x[relaxed_energy_column] is None)
        ),
        num_proc=num_proc,
    )

    need_relax = need_relax.filter(
        lambda x: check_structure_for_relaxation(x, structure_column=structure_column), num_proc=num_proc,
    )

    need_relax = need_relax.map(
        partial(
            relax_f,
            device=device,
            relaxation_batch_size=relaxation_batch_size,
            checkpoint_path=checkpoint_path,
            structure_column=structure_column,
            relaxed_structure_column=relaxed_structure_column,
            relaxed_energy_column=relaxed_energy_column,
            **relaxer_params
        ),
        batched=True,
        batch_size=relaxation_batch_size,
    )

    # Grab the rows that were already fine.
    dataset_indices = np.arange(len(ds))
    need_relax_indices = np.array(need_relax["idx"])

    ok_indices = dataset_indices[~np.isin(dataset_indices, need_relax_indices)]
    already_ok = ds.select(ok_indices)

    # Stitch everything back together and drop the helper column.
    ds = (
        concatenate_datasets([need_relax, already_ok])
        .sort("idx")  # restores the original order
        .remove_columns("idx")
    )

    return ds
