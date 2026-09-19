from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

import numpy as np
from pymatgen.core import Structure


def validate_embeddings(
    values, *, count: int, dimension: int | None = None
) -> np.ndarray:
    """Validate one vector per structure without normalizing the feature space."""
    try:
        values = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embeddings must be numeric vectors of equal length.") from exc
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] == 0:
        raise ValueError(
            f"Expected embeddings with shape [{count}, D], got {values.shape}."
        )
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError(
            f"Embedding dimension is {values.shape[1]}, expected {dimension}. "
            "Check conditions.<column>.input_dim and the extractor checkpoint."
        )
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain NaN or infinity.")
    return values


class StructureEmbedder(ABC):
    """A frozen model returning raw, mean-pooled structure embeddings."""

    extractor: str
    output: str
    packages: tuple[str, ...] = ()
    supported_atomic_numbers: frozenset[int] | None = None

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Embedding checkpoint does not exist: {self.checkpoint}"
            )
        self.device = device
        self.dimension: int | None = None

    @abstractmethod
    def encode(self, structures: Sequence[Structure]) -> np.ndarray:
        """Return float32 values of shape [len(structures), dimension] in input order."""

    def _validate(self, values, count: int) -> np.ndarray:
        values = validate_embeddings(values, count=count, dimension=self.dimension)
        self.dimension = values.shape[1]
        return values

    def metadata(self) -> dict:
        with self.checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "extractor": self.extractor,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": digest,
            "output": self.output,
            "pooling": "mean",
            "dtype": "float32",
            "dimension": self.dimension,
            "versions": {name: version(name) for name in ("torch", *self.packages)},
        }


def create_embedder(
    extractor: str, checkpoint: str, device: str = "cpu"
) -> StructureEmbedder:
    """Import optional model dependencies only when the extractor is requested."""
    if extractor == "pet":
        from prisma.embeddings.pet import PETEmbedder

        return PETEmbedder(checkpoint, device)
    if extractor == "mace":
        from prisma.embeddings.mace import MACEEmbedder

        return MACEEmbedder(checkpoint, device)
    raise ValueError(
        f"Unknown embedding extractor {extractor!r}; choose 'pet' or 'mace'."
    )
