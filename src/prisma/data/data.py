from typing import Dict, Any, Optional

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, Batch
from torch_geometric.data.storage import GlobalStorage

from prisma.utils.functions import (
    frac_to_cart_coords_from_lattice_matrix,
    radius_graph_pbc,
)


def symmetrize_cell(cell: np.ndarray | list[list[float]]) -> np.ndarray:
    cell = np.array(cell)

    _, s, vh = np.linalg.svd(cell.T)
    symmetric_cell = vh.T @ np.diag(s) @ vh

    return symmetric_cell


def sample_num_atoms(
    batch_size: int = 1, generator: torch.Generator | None = None
) -> torch.Tensor:
    num_atoms_distribution = [
        0.0002303828963737732,
        0.002804088967292211,
        0.019342289742695216,
        0.1636343889258233,
        0.04668051158167732,
        0.07808005476530565,
        0.027247714272549548,
        0.1150400537121267,
        0.048984340545415055,
        0.12620539622566992,
        0.03577352703049611,
        0.14591300741832927,
        0.0060031200426537475,
        0.028628366058675234,
        0.02022761830161729,
        0.04473213051520198,
        0.0013033089566287742,
        0.038699389814443035,
        0.0070135136024644384,
        0.04345679662456145,
    ]

    device = generator.device if generator else None
    probs = torch.tensor(num_atoms_distribution, device=device)

    indices = torch.multinomial(
        probs, num_samples=batch_size, replacement=True, generator=generator
    )

    num_atoms = indices + 1

    return num_atoms


class KeyMappedStorage(GlobalStorage):
    def __init__(
        self,
        _key_mapping: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._key_mapping: Dict[str, Any] = _key_mapping

    def __getattr__(self, key: str) -> Any:
        if key == "_key_mapping":
            self._key_mapping = {}
            return self._key_mapping
        else:
            return super().__getattr__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_key_mapping":
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key: str) -> Any:
        key = self._key_mapping.get(key, key)

        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        key = self._key_mapping.get(key, key)

        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        key = self._key_mapping.get(key, key)

        super().__delitem__(key)


class MappingData(Data):
    _key_mapping: Dict[str, str] = {}

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.__dict__["_store"] = KeyMappedStorage(
            _key_mapping=self._key_mapping, _parent=self
        )

        for key, value in kwargs.items():
            setattr(self, key, value)


class StructureData(MappingData):
    cart_coords: torch.Tensor
    frac_coords: torch.Tensor
    atomic_numbers: torch.LongTensor
    cell: torch.Tensor
    num_atoms: torch.Tensor
    neighbors: torch.Tensor
    cell_offsets: torch.Tensor
    condition: dict[str, torch.Tensor | str]

    _key_mapping: Dict[str, str] = {
        "pos": "cart_coords",
        "positions": "cart_coords",
        "natoms": "num_atoms",
    }

    def update_structure(
        self,
        atomic_numbers=None,
        frac_coords=None,
        cell=None,
        num_atoms=None,
    ):
        if num_atoms is not None:
            self.num_atoms = num_atoms

        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers

        if frac_coords is not None:
            self.frac_coords = frac_coords

        if cell is not None:
            self.cell = cell

        if isinstance(self, Batch) and (cell is not None or frac_coords is not None):
            self.cart_coords = frac_to_cart_coords_from_lattice_matrix(
                frac_coords=self.frac_coords,
                lattice_matrix=self.cell,
                num_atoms=self.num_atoms,
            )

    def build_graph(self, cutoff=6.0, max_neighbors=50):
        data = self if isinstance(self, Batch) else Batch.from_data_list([self])

        self.edge_index, self.cell_offsets, self.neighbors = radius_graph_pbc(
            data,
            cutoff,
            max_neighbors,
            True,
        )

    @classmethod
    def from_pymatgen(cls, structure, condition: dict[str, Any] | None = None):
        # niggli
        structure = structure.get_reduced_structure()

        symmetric_lattice = symmetrize_cell(structure.lattice.matrix)

        structure = Structure(
            lattice=symmetric_lattice,
            species=structure.species,
            coords=structure.frac_coords,
            coords_are_cartesian=False,
        )

        frac_coords = torch.tensor(
            structure.frac_coords,
            dtype=torch.get_default_dtype(),
        )
        atomic_numbers = torch.LongTensor(structure.atomic_numbers)

        num_atoms = atomic_numbers.shape[0]

        cell = torch.tensor(
            structure.lattice.matrix,
            dtype=torch.get_default_dtype(),
        ).unsqueeze(0)

        cart_coords = frac_to_cart_coords_from_lattice_matrix(
            frac_coords, cell, num_atoms
        )

        data = cls(
            frac_coords=frac_coords,
            cart_coords=cart_coords,
            atomic_numbers=atomic_numbers,
            num_atoms=num_atoms,
            cell=cell,
            num_nodes=num_atoms,
        )

        if condition:
            data.condition = condition

        return data


class MACEStructureData(StructureData):
    cart_coords: torch.Tensor
    frac_coords: torch.Tensor
    atomic_numbers: torch.LongTensor
    cell: torch.Tensor
    num_atoms: torch.Tensor
    neighbors: torch.Tensor
    cell_offsets: torch.Tensor
    shifts: torch.Tensor
    node_attributes: torch.Tensor

    _key_mapping: Dict[str, str] = {
        "pos": "cart_coords",
        "positions": "cart_coords",
        "natoms": "num_atoms",
        "unit_shifts": "cell_offsets",
    }

    def build_graph(self, cutoff=6.0, max_neighbors=50):
        super().build_graph(cutoff=cutoff, max_neighbors=max_neighbors)

        self.shifts = frac_to_cart_coords_from_lattice_matrix(
            self.cell_offsets,
            self.cell,
            self.neighbors,
        )
