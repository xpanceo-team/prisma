from __future__ import annotations

import numpy as np
import torch

from prisma.embeddings.base import StructureEmbedder


class MACEEmbedder(StructureEmbedder):
    extractor = "mace"
    output = "invariant_descriptors_all_layers"
    packages = ("mace-torch",)

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
            # Unsupported species must fail; replacing them by hydrogen would change
            # the structure whose distance we claim to measure.
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
