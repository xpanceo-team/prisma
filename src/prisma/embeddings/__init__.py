"""Optional structure embedding extractors, independent of generation backbones."""

from prisma.embeddings.base import StructureEmbedder, create_embedder

__all__ = ["StructureEmbedder", "create_embedder"]
