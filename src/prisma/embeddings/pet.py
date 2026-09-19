from __future__ import annotations

import numpy as np
import torch

from prisma.embeddings.base import StructureEmbedder


class PETEmbedder(StructureEmbedder):
    extractor = "pet"
    output = "mtt::aux::energy_last_layer_features"
    packages = (
        "metatrain",
        "metatomic-torch",
        "metatensor-torch",
        "metatensor-core",
        "metatensor-learn",
        "metatensor-operations",
    )

    def __init__(self, checkpoint: str, device: str = "cpu"):
        super().__init__(checkpoint, device)
        try:
            from metatrain.utils.io import model_from_checkpoint
        except ImportError as exc:
            raise ImportError(
                'PET embeddings require the optional dependencies: pip install -e ".[pet]"'
            ) from exc

        # The official loader upgrades old hypers AND state-dict keys. In particular,
        # an old PET-MAD checkpoint needs Cosine cutoff, not the new Bump default.
        checkpoint_data = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        )
        self.model = model_from_checkpoint(checkpoint_data, context="finetune")
        self.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if self.output not in self.model.outputs:
            raise ValueError(f"The PET checkpoint does not provide {self.output!r}.")
        self.dimension = int(self.model.last_layer_feature_size)
        self.supported_atomic_numbers = frozenset(
            int(number) for number in self.model.atomic_types
        )

    def encode(self, structures):
        from metatomic.torch import ModelOutput, System
        from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

        if not structures:
            return np.empty((0, self.dimension), dtype=np.float32)
        systems = []
        supported = self.supported_atomic_numbers
        # Construct CPU neighbor lists, then move the entire System to the model device.
        for index, structure in enumerate(structures):
            atoms = structure.to_ase_atoms()
            unsupported = {int(number) for number in atoms.numbers} - supported
            if unsupported:
                raise ValueError(
                    f"PET does not support atomic numbers {sorted(unsupported)} in structure {index}."
                )
            system = System(
                types=torch.tensor(atoms.numbers, dtype=torch.long),
                positions=torch.tensor(atoms.positions, dtype=torch.float32),
                cell=torch.tensor(np.asarray(atoms.cell), dtype=torch.float32),
                pbc=torch.tensor(atoms.pbc, dtype=torch.bool),
            )
            system = get_system_with_neighbor_lists(
                system, self.model.requested_neighbor_lists()
            )
            systems.append(system.to(device=self.device))

        with (
            torch.inference_mode(False),
            torch.no_grad(),
            torch.autocast(device_type=torch.device(self.device).type, enabled=False),
        ):
            result = self.model(
                systems=systems, outputs={self.output: ModelOutput(per_atom=True)}
            )
            features = result[self.output].block()
            system_ids = features.samples.column("system")
            vectors = []
            for index in range(len(systems)):
                atom_features = features.values[system_ids == index]
                if len(atom_features) != len(structures[index]):
                    raise ValueError(
                        f"PET returned an unexpected atom count for structure {index}."
                    )
                vectors.append(atom_features.mean(dim=0).cpu().numpy())
        return self._validate(np.stack(vectors), len(structures))

    def metadata(self):
        return {**super().metadata(), "checkpoint_context": "finetune"}
