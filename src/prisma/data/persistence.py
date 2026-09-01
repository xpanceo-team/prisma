from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def save_dataset(
    dataset: Dataset | DatasetDict,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    max_shard_size: str | int | None = None,
    num_proc: int | None = None,
) -> Path:
    """Atomically save a Dataset or DatasetDict in Datasets disk format."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass overwrite=True to replace it."
        )

    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    backup_path = output_path.parent / f".{output_path.name}.backup-{uuid.uuid4().hex}"
    try:
        dataset.save_to_disk(
            temporary_path,
            max_shard_size=max_shard_size,
            num_proc=num_proc,
        )
        if output_path.exists():
            output_path.rename(backup_path)
        try:
            temporary_path.rename(output_path)
        except Exception:
            if backup_path.exists():
                backup_path.rename(output_path)
            raise
        _remove_path(backup_path)
    finally:
        _remove_path(temporary_path)
    return output_path


def load_saved_dataset(path: str | Path) -> DatasetDict:
    dataset = load_from_disk(str(Path(path).expanduser()))
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    return dataset


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
