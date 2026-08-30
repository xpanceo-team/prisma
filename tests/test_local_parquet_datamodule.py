from pathlib import Path

import pytest
from datasets import Dataset
from pymatgen.core import Lattice, Structure

from prisma.training.datamodule import DataModule


def _datamodule_kwargs(**overrides):
    kwargs = {
        "dataset_name": None,
        "dataset_path": "materials.parquet",
        "dataset_subset": None,
        "revision": None,
        "data_cls": "prisma.data.StructureData",
        "condition": {"property": {"scale": True}},
        "max_num_atoms": 20,
        "validation_fraction": 0.25,
        "split_seed": 42,
        "persistent_workers": False,
        "num_workers": 0,
        "batch_size": 2,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("dataset_name", "dataset_path"),
    [(None, None), ("organization/dataset", "materials.parquet")],
)
def test_datamodule_requires_exactly_one_dataset_source(
    dataset_name, dataset_path
):
    with pytest.raises(ValueError, match="exactly one dataset source"):
        DataModule(
            **_datamodule_kwargs(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
            )
        )


def test_local_dataset_directory_resolves_conventional_splits(tmp_path):
    for filename in ("train.parquet", "validation.parquet", "test.parquet"):
        (tmp_path / filename).touch()

    assert DataModule._resolve_local_data_files(tmp_path) == {
        "train": str(tmp_path / "train.parquet"),
        "valid": str(tmp_path / "validation.parquet"),
        "test": str(tmp_path / "test.parquet"),
    }

    (tmp_path / "valid.parquet").touch()
    with pytest.raises(ValueError, match="Multiple local files"):
        DataModule._resolve_local_data_files(tmp_path)


def test_local_parquet_runs_through_training_data_pipeline(tmp_path):
    structures = [
        Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3]),
        Structure(
            Lattice.cubic(5.0),
            ["C"] * 3,
            [[0, 0, 0], [0.25] * 3, [0.5] * 3],
        ),
        Structure(Lattice.cubic(3.8), ["Ge"], [[0, 0, 0]]),
    ]
    dataset_path = tmp_path / "materials.parquet"
    Dataset.from_dict(
        {
            "structure": [structure.to(fmt="json") for structure in structures],
            "property": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(dataset_path)

    datamodule = DataModule(
        **_datamodule_kwargs(
            dataset_path=str(dataset_path),
            max_num_atoms=2,
        )
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    assert len(datamodule.train_dataset) == 2
    assert len(datamodule.valid_dataset) == 1
    assert datamodule.condition_stats["property"]["scale_mean"]

    batch = next(iter(datamodule.train_dataloader()))
    assert batch["data"].num_graphs == 2
    assert batch["condition"]["property"].shape == (2,)
