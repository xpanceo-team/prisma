from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, Value, concatenate_datasets

from prisma.data.loading import SourceType, load_dataset_source
from prisma.data.structures import StructureFormat, normalize_structure


_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "valid",
    "valid": "valid",
    "val": "valid",
    "test": "test",
}


def prepare_dataset(
    source: Any,
    *,
    source_type: SourceType = "auto",
    config_name: str | None = None,
    revision: str | None = None,
    structure_column: str = "structure",
    structure_format: StructureFormat = "auto",
    id_column: str | None = "material_id",
    split_column: str | None = None,
    validation_fraction: float | None = None,
    seed: int = 42,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and normalize a dataset for PRISMA training."""

    if split_column is not None and validation_fraction is not None:
        raise ValueError(
            "split_column and validation_fraction cannot be used together."
        )
    if validation_fraction is not None and not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    loaded = load_dataset_source(
        source,
        source_type=source_type,
        config_name=config_name,
        revision=revision,
        id_column=id_column,
    )
    datasets = _as_dataset_dict(loaded)
    if split_column is not None:
        if len(datasets) != 1 or "train" not in datasets:
            raise ValueError(
                "split_column can only be used with a single, unsplit dataset."
            )
        datasets = _split_by_column(datasets["train"], split_column)

    prepared = DatasetDict(
        {
            split: _prepare_split(
                dataset,
                split=split,
                structure_column=structure_column,
                structure_format=structure_format,
                id_column=id_column,
                num_proc=num_proc,
            )
            for split, dataset in datasets.items()
        }
    )

    if validation_fraction is not None:
        if "valid" in prepared:
            raise ValueError(
                "validation_fraction cannot be used when valid already exists."
            )
        if "train" not in prepared:
            raise ValueError("A train split is required to create validation data.")
        split = prepared["train"].train_test_split(
            test_size=validation_fraction,
            seed=seed,
        )
        prepared["train"] = split["train"]
        prepared["valid"] = split["test"]

    if "train" not in prepared:
        raise ValueError("Prepared datasets must contain a train split.")
    return prepared


def _as_dataset_dict(dataset: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})

    normalized = {}
    for original_name, split_dataset in dataset.items():
        split_name = _normalize_split_name(original_name)
        if split_name in normalized:
            normalized[split_name] = concatenate_datasets(
                [normalized[split_name], split_dataset]
            )
        else:
            normalized[split_name] = split_dataset
    return DatasetDict(normalized)


def _split_by_column(dataset: Dataset, split_column: str) -> DatasetDict:
    if split_column not in dataset.column_names:
        raise ValueError(f"Dataset has no {split_column!r} split column.")

    split_values = dataset.unique(split_column)
    grouped = {}
    for split_value in split_values:
        if not isinstance(split_value, str):
            raise ValueError("Split names must be strings.")
        split_name = _normalize_split_name(split_value)
        subset = dataset.filter(
            lambda value, expected=split_value: value == expected,
            input_columns=split_column,
        ).remove_columns(split_column)
        if split_name in grouped:
            grouped[split_name] = concatenate_datasets([grouped[split_name], subset])
        else:
            grouped[split_name] = subset
    return DatasetDict(grouped)


def _prepare_split(
    dataset: Dataset,
    *,
    split: str,
    structure_column: str,
    structure_format: StructureFormat,
    id_column: str | None,
    num_proc: int | None,
) -> Dataset:
    if structure_column not in dataset.column_names:
        raise ValueError(
            f"Split {split!r} has no {structure_column!r} structure column."
        )
    if structure_column != "structure" and "structure" in dataset.column_names:
        raise ValueError(
            "Dataset already contains 'structure'; choose which column to keep first."
        )
    if (
        id_column is not None
        and id_column != "material_id"
        and id_column in dataset.column_names
        and "material_id" in dataset.column_names
    ):
        raise ValueError(
            "Dataset already contains 'material_id'; choose which ID column to keep."
        )

    def convert_structure(value, index):
        return {
            "structure": normalize_structure(
                value,
                structure_format=structure_format,
                context=f"split={split}, row={index}",
            )
        }

    prepared = dataset.map(
        convert_structure,
        input_columns=structure_column,
        with_indices=True,
        num_proc=num_proc,
        desc=f"normalize {split} structures",
    )
    if structure_column != "structure":
        prepared = prepared.remove_columns(structure_column)

    if id_column is not None and id_column in prepared.column_names:
        if id_column != "material_id":
            prepared = prepared.rename_column(id_column, "material_id")
        prepared = prepared.cast_column("material_id", Value("string"))
    return prepared


def _normalize_split_name(name: str) -> str:
    normalized = _SPLIT_ALIASES.get(name.lower())
    if normalized is None:
        supported = ", ".join(sorted(_SPLIT_ALIASES))
        raise ValueError(f"Unsupported split {name!r}. Expected one of: {supported}.")
    return normalized
