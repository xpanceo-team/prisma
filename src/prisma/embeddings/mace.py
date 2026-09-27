from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
import torch

from prisma.embeddings.base import StructureEmbedder


class MACEEmbedder(StructureEmbedder):
    extractor = "mace"
    output = "invariant_descriptors_all_layers"
    packages = ("mace-torch",)
    mace_supported_atomic_numbers = frozenset((*range(1, 84), *range(89, 95)))

    def __init__(self, checkpoint: str, device: str = "cpu"):
        super().__init__(checkpoint, device)
        try:
            from mace.calculators import MACECalculator
        except ImportError as exc:
            raise ImportError(
                "MACE embeddings require mace-torch: pip install mace-torch"
            ) from exc
        self.calculator = MACECalculator(
            model_paths=str(self.checkpoint), device=device, default_dtype="float32"
        )

    def encode(self, structures):
        vectors = []
        for structure in structures:
            atomic_numbers = np.asarray(structure.atomic_numbers)
            supported = np.isin(
                atomic_numbers, tuple(self.mace_supported_atomic_numbers)
            )
            if not supported.all():
                # Preserve the historical Atlas/MACE embedding definition used by
                # this project: unsupported species are represented as hydrogen.
                structure = Structure(
                    lattice=structure.lattice,
                    species=np.where(supported, atomic_numbers, 1),
                    coords=structure.frac_coords,
                    coords_are_cartesian=False,
                )
            with (
                torch.inference_mode(False),
                torch.enable_grad(),
                torch.autocast(
                    device_type=torch.device(self.device).type, enabled=False
                ),
            ):
                descriptors = self.calculator.get_descriptors(
                    structure.to_ase_atoms(), invariants_only=True, num_layers=-1
                )
            vectors.append(np.asarray(descriptors).mean(axis=0))
        return self._validate(vectors, len(structures))

    def metadata(self):
        return {
            **super().metadata(),
            "unsupported_species_policy": "replace_with_hydrogen",
            "supported_atomic_numbers": sorted(self.mace_supported_atomic_numbers),
        }
