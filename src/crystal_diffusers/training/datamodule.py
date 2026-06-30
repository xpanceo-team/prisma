import random
from collections.abc import Sequence
from typing import Optional
import functools

from datasets import load_dataset, Dataset
import numpy as np
import lightning.pytorch as pl
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.data.dataloader import default_collate

from crystal_diffusers.utils.logging import logger
from crystal_diffusers.training.dataset import preprocess_dataset
from crystal_diffusers.utils.functions import get_class_from_string


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
        self.save_hyperparameters()

        self.dataset_name = self.hparams.dataset_name
        self.dataset_subset = self.hparams.dataset_subset
        self.revision = self.hparams.revision

        self.condition = self.hparams.condition
        self.condition_stats = {}

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.data_cls = get_class_from_string(self.hparams.data_cls)

    def prepare_data(self) -> None:
        logger.info("Preparing datasets")
        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_subset,
            revision=self.revision,
        )
        logger.debug("Loaded datasets")

        condition_keys = list(self.condition.keys())

        logger.debug("Preprocessing datasets")
        preprocess_dataset(
            ds,
            condition_keys=condition_keys,
            structure_json_col="structure",
            data_cls=self.data_cls,
            num_proc=self.hparams.num_workers,
        )
        logger.debug("Preprocessed datasets")

    def setup(self, stage: Optional[str] = None):
        if stage is None or stage == "fit":
            self.train_dataset = self._get_dataset("train")
            self.condition_stats = self._get_condition_stats(self.train_dataset)

            self.valid_dataset = self._get_dataset("valid")

        if stage is None or stage == "test":
            try:
                self.test_dataset = self._get_dataset("test")
            except (ValueError, KeyError) as exc:
                message = str(exc).lower()
                if "split" in message and "test" in message:
                    logger.warning(
                        "Dataset has no test split; skipping post-training test."
                    )
                    self.test_dataset = None
                else:
                    raise

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

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

    def _get_dataset(self, split: str) -> Dataset:
        if split == "train":
            if self.train_dataset is not None:
                return self.train_dataset
        elif split == "valid":
            if self.valid_dataset is not None:
                return self.valid_dataset
        elif split == "test":
            if self.test_dataset is not None:
                return self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")

        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_subset,
            revision=self.revision,
            split=split,
        )

        condition_keys = list(self.condition.keys())

        ds = preprocess_dataset(
            ds,
            condition_keys=condition_keys,
            structure_json_col="structure",
            data_cls=self.data_cls,
            num_proc=self.hparams.num_workers,
        )

        transform_fn = functools.partial(
            dataset_transform,
            condition_keys=condition_keys,
            data_cls=self.data_cls,
        )

        ds.set_transform(transform_fn)

        return ds

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

        def collate_fn(examples):
            return {
                "data": Batch.from_data_list([example["data"] for example in examples]),
                "condition": default_collate(
                    [example["condition"] for example in examples]
                ),
            }

        # TODO: why not torch DataLoader?
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.hparams.persistent_workers,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.hparams=})"
