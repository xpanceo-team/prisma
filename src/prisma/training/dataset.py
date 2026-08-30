import functools

import os
import pickle
from pathlib import Path
from typing import Type, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from pymatgen.core import Structure
from tqdm.auto import tqdm

from prisma.data import StructureData


def get_data_dict_from_structure_json(structure_str, data_cls: StructureData):
    s = Structure.from_str(structure_str, fmt="json")

    return dict(data_cls.from_pymatgen(s))


def structure_within_atom_limit(structure_str: str, max_num_atoms: int) -> bool:
    structure = Structure.from_str(structure_str, fmt="json")
    return len(structure) <= max_num_atoms


def preprocess_dataset(
    ds: Dataset | DatasetDict,
    condition_keys: list[str],
    structure_json_col: str,
    data_cls: Type[StructureData],
    num_proc: int | None = None,
    max_num_atoms: int | None = 20,
) -> Dataset | DatasetDict:
    if isinstance(ds, DatasetDict):
        dataset_columns = ds.column_names["train"]
    else:
        dataset_columns = ds.column_names

    remove_columns = [
        col
        for col in dataset_columns
        if col not in condition_keys + [structure_json_col]
    ]

    missing_columns = set(condition_keys) - set(dataset_columns)
    if missing_columns:
        raise ValueError(
            f"Columns {list(missing_columns)} not in the dataset."
            f" Current columns in the dataset: {dataset_columns}"
        )

    if max_num_atoms is not None:
        if max_num_atoms < 1:
            raise ValueError("max_num_atoms must be positive or None.")

        filter_fn = functools.partial(
            structure_within_atom_limit,
            max_num_atoms=max_num_atoms,
        )
        ds = ds.filter(
            filter_fn,
            num_proc=num_proc,
            input_columns=structure_json_col,
            desc=f"filter structures with at most {max_num_atoms} atoms",
        )

    preprocess_fn = functools.partial(
        get_data_dict_from_structure_json,
        data_cls=data_cls,
    )

    new_ds = ds.map(
        preprocess_fn,
        num_proc=num_proc,
        input_columns=structure_json_col,
        desc="pymatgen -> tensors",
        remove_columns=remove_columns,
    ).remove_columns(structure_json_col)

    return new_ds


class StructureDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        cache_dir: str = "data_cache",
        name: Optional[str] = None,
        num_workers: int = 8,
        data_cls: Type[StructureData] = None,
        condition_columns: list[str] | None = None,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.split = split
        self.cache_dir = cache_dir
        self.cache_path = None
        self.num_workers = num_workers
        self.condition_columns = condition_columns

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            path_text = f"{path.replace('/', '_').replace('.', '_')}"

            if self.name is not None:
                path_text += f"_{name}"

            self.cache_path = Path(self.cache_dir) / f"{path_text}_{split}.pkl"

        if self.cache_dir is not None and self.cache_path.exists():
            self.data = self._load_data(data_cls=data_cls)
        else:
            data = self._prepare(data_cls=data_cls, condition_columns=condition_columns)

            self.data = data
            self._save_data(data)

    def _prepare(
        self,
        data_cls: Type[StructureData] | None = None,
        condition_columns: list[str] | None = None,
    ):
        if condition_columns is None:
            condition_columns = []

        path = self.path

        if os.path.exists(path):
            if path.endswith(".json"):
                loaded_df = pd.read_json(path)
            elif path.endswith(".csv"):
                loaded_df = pd.read_csv(path)
            else:
                raise ValueError(f"Unable to load {path}. Invalid file exstension.")
        else:
            loaded_df = load_dataset(path, name=self.name, split=self.split).to_pandas()

        if data_cls is None:
            data_cls = StructureData

        data = []
        for idx in tqdm(range(len(loaded_df))):
            row = loaded_df.iloc[idx]

            s = Structure.from_str(row["structure"], fmt="json")

            condition = {name: row[name] for name in condition_columns}
            try:
                data.append(data_cls.from_pymatgen(s, condition=condition))
            except Exception as e:
                print(f"Exception occured while processing {s}: {e}")

        return data

    def _save_data(self, data):
        if self.cache_dir is not None:
            with open(self.cache_path, "wb") as f:
                data_list = [dict(item) for item in data]
                pickle.dump(data_list, f)

    def _load_data(
        self,
        data_cls: Type[StructureData] = None,
    ):
        if data_cls is None:
            data_cls = StructureData

        print(f"Loading from cache: {self.cache_path}")
        with open(self.cache_path, "rb") as f:
            data_list = pickle.load(f)

        data = [data_cls(**item) for item in data_list]

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path=})"
