from __future__ import annotations

import uuid

from datasets import Dataset, DatasetDict, Sequence, Value
from pymatgen.core import Structure

from prisma.embeddings.base import StructureEmbedder, validate_embeddings


def embed_dataset(
    dataset: DatasetDict,
    embedder: StructureEmbedder,
    *,
    column: str = "embedding",
    batch_size: int = 16,
    skip_report: dict | None = None,
    replace_column: bool = False,
) -> DatasetDict:
    """Add embeddings, omitting structures with elements unsupported by the extractor."""
    structural_columns = {
        "structure",
        "num_atoms",
        "num_nodes",
        "atomic_numbers",
        "frac_coords",
        "cart_coords",
        "cell",
    }
    if not column or column in structural_columns:
        raise ValueError(
            "Choose a non-empty embedding column that is not a structural field."
        )
    if batch_size < 1:
        raise ValueError("Embedding batch_size must be positive.")
    for split, rows in dataset.items():
        if "structure" not in rows.column_names:
            raise ValueError(
                f"Split {split!r} has no 'structure' column; run prisma data prepare first."
            )
        if column in rows.column_names and not replace_column:
            raise ValueError(
                f"Column {column!r} already exists in split {split!r}; choose a new column."
            )
    if not any(len(rows) for rows in dataset.values()):
        raise ValueError("Cannot embed an empty dataset.")

    if replace_column:
        dataset = DatasetDict(
            {
                split: (
                    rows.remove_columns(column) if column in rows.column_names else rows
                )
                for split, rows in dataset.items()
            }
        )

    result = DatasetDict()
    supported = embedder.supported_atomic_numbers
    if supported is not None and embedder.dimension is None:
        raise ValueError("An extractor with known elements must report its dimension.")
    for split, rows in dataset.items():
        skipped = {"count": 0, "by_atomic_number": {}, "example_row_indices": []}

        def encode_batch(batch, indices):
            try:
                structures = [
                    Structure.from_str(value, fmt="json")
                    for value in batch["structure"]
                ]
                if supported is not None:
                    kept = []
                    for offset, structure in enumerate(structures):
                        unsupported = {
                            int(number) for number in structure.atomic_numbers
                        } - supported
                        if unsupported:
                            skipped["count"] += 1
                            if len(skipped["example_row_indices"]) < 5:
                                skipped["example_row_indices"].append(
                                    int(indices[offset])
                                )
                            for number in unsupported:
                                key = str(number)
                                counts = skipped["by_atomic_number"]
                                counts[key] = counts.get(key, 0) + 1
                        else:
                            kept.append(offset)
                    vectors = (
                        validate_embeddings(
                            embedder.encode([structures[i] for i in kept]),
                            count=len(kept),
                        ).tolist()
                        if kept
                        else []
                    )
                    return {
                        **{
                            key: [values[i] for i in kept]
                            for key, values in batch.items()
                        },
                        column: vectors,
                    }
                vectors = validate_embeddings(
                    embedder.encode(structures), count=len(structures)
                )
                return {column: vectors.tolist()}
            except Exception as exc:
                raise ValueError(
                    f"Embedding failed in split {split!r}, rows {indices[0]}–{indices[-1]}: {exc}"
                ) from exc

        if len(rows):
            map_options = {}
            if supported is not None:
                features = rows.features.copy()
                features[column] = Sequence(Value("float32"), length=embedder.dimension)
                map_options = {
                    "remove_columns": rows.column_names,
                    "features": features,
                }
            result[split] = rows.with_format(None).map(
                encode_batch,
                batched=True,
                batch_size=batch_size,
                with_indices=True,
                load_from_cache_file=False,
                new_fingerprint=uuid.uuid4().hex,
                desc=f"{embedder.extractor} embeddings: {split}",
                **map_options,
            )
        else:
            result[split] = rows.with_format(None).add_column(column, [])
        if skip_report is not None:
            skip_report[split] = skipped

    for split in ("train", "valid"):
        if split in result and not len(result[split]):
            raise ValueError(
                f"No structures remain in split {split!r} after embedding. "
                "Use a checkpoint that supports this dataset's elements."
            )

    dimension = embedder.dimension
    if dimension is None:
        raise ValueError("The extractor did not report its embedding dimension.")
    feature = Sequence(Value("float32"), length=dimension)
    for split, rows in result.items():
        if len(rows):
            result[split] = rows.cast_column(column, feature)
        else:
            info = rows.info.copy()
            info.features[column] = feature
            result[split] = Dataset.from_dict(
                {key: [] for key in rows.column_names}, info=info, split=rows.split
            )
    return result
