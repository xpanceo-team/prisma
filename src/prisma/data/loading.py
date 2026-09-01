from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


SourceType = Literal["auto", "hub"]


def load_dataset_source(
    source: Any,
    *,
    source_type: SourceType = "auto",
    config_name: str | None = None,
    revision: str | None = None,
    id_column: str | None = "material_id",
) -> Dataset | DatasetDict:
    """Load a supported dataset source without changing its schema or splits."""

    if source_type not in {"auto", "hub"}:
        raise ValueError("source_type must be 'auto' or 'hub'.")

    if isinstance(source, (Dataset, DatasetDict)):
        if source_type != "auto":
            raise ValueError("source_type is only used with string sources.")
        return source

    if isinstance(source, pd.DataFrame):
        if source_type != "auto":
            raise ValueError("source_type is only used with string sources.")
        return Dataset.from_pandas(source, preserve_index=False)

    if not isinstance(source, (str, Path)):
        raise TypeError(
            "source must be a path, Hub dataset name, pandas DataFrame, "
            "Dataset, or DatasetDict."
        )

    if source_type == "hub":
        if isinstance(source, Path):
            source = str(source)
        return load_dataset(source, name=config_name, revision=revision)

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Local dataset source does not exist: {path}. "
            "Use source_type='hub' for a Hub dataset."
        )
    if path.is_dir():
        return _load_local_directory(path, id_column=id_column)
    return _load_tabular_file(path, id_column=id_column)


def _load_local_directory(
    path: Path,
    *,
    id_column: str | None,
) -> Dataset | DatasetDict:
    if (path / "dataset_dict.json").is_file() or (path / "state.json").is_file():
        return load_from_disk(str(path))

    split_files = {}
    for split, names in {
        "train": ("train.parquet", "train.csv", "train.json", "train.jsonl"),
        "valid": (
            "valid.parquet",
            "valid.csv",
            "valid.json",
            "valid.jsonl",
            "validation.parquet",
            "validation.csv",
            "validation.json",
            "validation.jsonl",
        ),
        "test": ("test.parquet", "test.csv", "test.json", "test.jsonl"),
    }.items():
        matches = [path / name for name in names if (path / name).is_file()]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple files found for the {split!r} split: {matches}."
            )
        if matches:
            split_files[split] = matches[0]

    if "train" not in split_files:
        raise ValueError(
            f"Directory is not a saved dataset and contains no train data: {path}."
        )

    suffixes = {file.suffix.lower() for file in split_files.values()}
    if len(suffixes) != 1:
        raise ValueError("All split files in a directory must use the same format.")
    loader_name = _loader_name(next(iter(split_files.values())))
    load_kwargs = {}
    if loader_name == "csv" and id_column is not None:
        load_kwargs["converters"] = {id_column: str}
    return load_dataset(
        loader_name,
        data_files={split: str(file) for split, file in split_files.items()},
        **load_kwargs,
    )


def _load_tabular_file(path: Path, *, id_column: str | None) -> Dataset:
    loader_name = _loader_name(path)
    load_kwargs = {}
    if loader_name == "csv" and id_column is not None:
        load_kwargs["converters"] = {id_column: str}
    return load_dataset(
        loader_name,
        data_files=str(path),
        split="train",
        **load_kwargs,
    )


def _loader_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise ValueError(
        f"Unsupported dataset file {path}. Use Parquet, CSV, JSON, or JSONL."
    )
