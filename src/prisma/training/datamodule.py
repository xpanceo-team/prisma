import functools
import heapq
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

from datasets import (
    Dataset,
    DatasetDict,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)
import numpy as np
import lightning.pytorch as pl
from omegaconf import OmegaConf
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import default_collate

from prisma.utils.logging import logger
from prisma.training.dataset import preprocess_dataset
from prisma.utils.functions import get_class_from_string


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def _transform_condition(value: float | str | Sequence | None):
    if isinstance(value, (float, int, Sequence)):
        return torch.tensor(value)
    elif isinstance(value, str):
        return value
    elif value is None:
        return torch.tensor(float("nan"))
    else:
        raise TypeError(f"Unsupported condition type: {type(value)}")


def dataset_transform(batch, condition_keys, data_cls):
    data_keys = [key for key in batch.keys() if key not in condition_keys]

    batch_size = len(batch[data_keys[0]])

    data = [
        data_cls.from_dict(
            {
                **{k: torch.tensor(batch[k][idx]) for k in data_keys},
            }
        )
        for idx in range(batch_size)
    ]

    condition = [
        {key: _transform_condition(batch[key][idx]) for key in condition_keys}
        for idx in range(batch_size)
    ]

    return {"data": data, "condition": condition}


class DataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        plain_hparams = OmegaConf.to_container(
            OmegaConf.create(kwargs),
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )
        self.save_hyperparameters(plain_hparams)
        self.cfg = OmegaConf.create(plain_hparams)

        self.dataset_name = self.cfg.get("dataset_name")
        self.dataset_path = self.cfg.get("dataset_path")
        self.dataset_subset = self.cfg.dataset_subset
        self.revision = self.cfg.revision
        self.max_num_atoms = self.cfg.max_num_atoms
        self.validation_fraction = self.cfg.validation_fraction
        self.split_seed = self.cfg.split_seed

        self.condition = self.cfg.condition
        self.condition_stats = {}

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self._prepared_datasets: Optional[DatasetDict] = None
        self._test_split_warning_emitted = False

        self.data_cls = get_class_from_string(self.cfg.data_cls)

        if (self.dataset_name is None) == (self.dataset_path is None):
            raise ValueError(
                "Configure exactly one dataset source: dataset_name for a "
                "Hugging Face dataset or dataset_path for local data."
            )

    def _load_datasets(self) -> DatasetDict:
        if self.dataset_path is None:
            return self._normalize_split_names(
                load_dataset(
                    self.dataset_name,
                    name=self.dataset_subset,
                    revision=self.revision,
                )
            )

        path = Path(self.dataset_path).expanduser()
        if path.is_dir() and (
            (path / "dataset_dict.json").is_file()
            or (path / "state.json").is_file()
        ):
            datasets = load_from_disk(str(path))
            if isinstance(datasets, Dataset):
                datasets = DatasetDict({"train": datasets})
            return self._normalize_split_names(datasets)

        data_files = self._resolve_local_data_files(self.dataset_path)
        datasets = load_dataset("parquet", data_files=data_files)
        return self._normalize_split_names(datasets)

    @staticmethod
    def _normalize_split_names(datasets: DatasetDict) -> DatasetDict:
        if "validation" not in datasets:
            return datasets
        if "valid" in datasets:
            raise ValueError("Dataset contains both 'valid' and 'validation' splits.")
        normalized = DatasetDict(datasets)
        normalized["valid"] = normalized.pop("validation")
        return normalized

    @staticmethod
    def _resolve_local_data_files(
        dataset_path: str | Mapping[str, str],
    ) -> dict[str, str]:
        if isinstance(dataset_path, Mapping):
            data_files = {
                ("valid" if split == "validation" else split): str(
                    Path(path).expanduser()
                )
                for split, path in dataset_path.items()
            }
            if "train" not in data_files:
                raise ValueError("Local dataset split mapping must contain 'train'.")
            return data_files

        path = Path(dataset_path).expanduser()
        if path.is_file():
            if path.suffix.lower() != ".parquet":
                raise ValueError("Local dataset files must use the .parquet extension.")
            return {"train": str(path)}

        if not path.is_dir():
            raise FileNotFoundError(f"Local dataset path does not exist: {path}")

        data_files = {}
        split_filenames = {
            "train": ("train.parquet",),
            "valid": ("valid.parquet", "validation.parquet"),
            "test": ("test.parquet",),
        }
        for split, filenames in split_filenames.items():
            matches = [
                path / filename
                for filename in filenames
                if (path / filename).is_file()
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple local files found for the {split!r} split: {matches}"
                )
            if matches:
                data_files[split] = str(matches[0])

        if "train" not in data_files:
            raise ValueError(
                f"Local dataset directory must contain train.parquet: {path}"
            )
        return data_files

    def prepare_data(self) -> None:
        if self.dataset_name is None:
            return

        logger.info("Downloading dataset")
        builder = load_dataset_builder(
            self.dataset_name,
            name=self.dataset_subset,
            revision=self.revision,
        )
        builder.download_and_prepare()

    def setup(self, stage: Optional[str] = None):
        if stage not in {None, "fit", "validate", "test", "predict"}:
            raise ValueError(f"Unknown setup stage: {stage}")

        self._ensure_datasets_prepared()

        if (
            stage in {None, "test"}
            and self.test_dataset is None
            and not self._test_split_warning_emitted
        ):
            logger.warning("Dataset has no test split; skipping post-training test.")
            self._test_split_warning_emitted = True

    def _ensure_datasets_prepared(self) -> None:
        if self._prepared_datasets is not None:
            return

        logger.info("Preparing datasets")
        datasets = self._load_datasets()
        logger.debug("Loaded datasets")

        condition_keys = list(self.condition.keys())
        logger.debug("Preprocessing datasets")
        datasets = preprocess_dataset(
            datasets,
            condition_keys=condition_keys,
            structure_json_col="structure",
            data_cls=self.data_cls,
            num_proc=self.cfg.num_workers,
            max_num_atoms=self.max_num_atoms,
        )
        logger.debug("Preprocessed datasets")

        if "train" not in datasets:
            raise ValueError("Dataset has no 'train' split.")

        if "valid" not in datasets:
            if self.validation_fraction is None:
                raise ValueError(
                    "Dataset has no 'valid' split and validation_fraction is not set."
                )
            split_datasets = datasets["train"].train_test_split(
                test_size=self.validation_fraction,
                seed=self.split_seed,
            )
            datasets["train"] = split_datasets["train"]
            datasets["valid"] = split_datasets["test"]

        self.condition_stats = self._get_condition_stats(datasets["train"])

        transform_fn = functools.partial(
            dataset_transform,
            condition_keys=condition_keys,
            data_cls=self.data_cls,
        )
        for dataset in datasets.values():
            dataset.set_transform(transform_fn)

        self.train_dataset = datasets["train"]
        self.valid_dataset = datasets["valid"]
        self.test_dataset = datasets.get("test")
        self._prepared_datasets = datasets

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def preflight_dataloader(self) -> DataLoader:
        """Return one conservative batch for checking a training configuration."""
        if self.train_dataset is None:
            raise ValueError("Dataset wasn't initialized. Use DataModule.setup().")
        if len(self.train_dataset) == 0:
            raise ValueError(
                "The training split is empty after preprocessing and filtering."
            )

        raw_dataset = self.train_dataset.with_format(
            None,
            columns=["num_atoms"],
            output_all_columns=False,
        )
        num_atoms = raw_dataset["num_atoms"]
        batch_size = min(self.cfg.batch_size, len(num_atoms))
        largest_indices = heapq.nlargest(
            batch_size,
            range(len(num_atoms)),
            key=lambda index: num_atoms[index],
        )
        dataset = self.train_dataset.select(largest_indices)

        return self._make_dataloader(dataset, shuffle=False, num_workers=0)

    def _get_condition_stats(
        self,
        dataset: Dataset,
    ) -> dict[str, dict[str, list[float]]]:
        logger.info("Preparing condition statistics")

        previous_format = dataset.format
        dataset.set_format("torch", columns=list(self.condition.keys()))

        condition_stats = {}
        for cond_name, cond_info in self.condition.items():
            if not cond_info.get("scale"):
                continue

            if not cond_info.get("scale_mean") or not cond_info.get("scale_std"):
                values = dataset[cond_name]

                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values)

                if values.ndim == 1:
                    values = values[~values.isnan()]
                    values = values.unsqueeze(-1)

                stats_dict = {
                    "scale_mean": values.mean(0).tolist(),
                    "scale_std": values.std(0).tolist(),
                }

                condition_stats[cond_name] = stats_dict

        dataset.set_format(**previous_format)

        return condition_stats

    def _get_dataloader(self, split: str) -> DataLoader:
        if split == "train":
            dataset = self.train_dataset
            shuffle = True
        elif split == "valid":
            dataset = self.valid_dataset
            shuffle = False
        elif split == "test":
            dataset = self.test_dataset
            shuffle = False
        else:
            raise ValueError(f"Unknown split: {split}")

        if dataset is None:
            raise ValueError("Dataset wasn't initialized. Use DataModule.setup().")

        return self._make_dataloader(dataset, shuffle=shuffle)

    def _make_dataloader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        num_workers: int | None = None,
    ) -> DataLoader:
        if num_workers is None:
            num_workers = self.cfg.num_workers

        def collate_fn(examples):
            return {
                "data": Batch.from_data_list([example["data"] for example in examples]),
                "condition": default_collate(
                    [example["condition"] for example in examples]
                ),
            }

        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.cfg.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.cfg.persistent_workers and num_workers > 0,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.cfg=})"
