from datasets import Dataset, DatasetDict
import numpy as np
from pymatgen.core import Lattice, Structure
import pytest

from prisma.data.cli import main as data_main
from prisma.data.embeddings import embed_dataset
from prisma.data.persistence import (
    load_dataset_metadata,
    load_saved_dataset,
    save_dataset,
)
from prisma.embeddings.base import StructureEmbedder, validate_embeddings


class ToyEmbedder(StructureEmbedder):
    extractor = "toy"
    output = "cell_and_count"

    def encode(self, structures):
        return self._validate(
            [[s.lattice.a, len(s)] for s in structures], len(structures)
        )


def test_custom_column_cli_preserves_splits_and_writes_dimension_and_provenance(
    tmp_path, monkeypatch, capsys
):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test weights")
    structures = [Structure(Lattice.cubic(a), ["Si"], [[0, 0, 0]]) for a in (3, 4, 5)]
    source = tmp_path / "source"
    save_dataset(
        DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "structure": [s.to(fmt="json") for s in structures[:2]],
                        "id": ["a", "b"],
                    }
                ),
                "valid": Dataset.from_dict(
                    {"structure": [structures[2].to(fmt="json")], "id": ["c"]}
                ),
            }
        ),
        source,
    )
    monkeypatch.setattr(
        "prisma.embeddings.create_embedder", lambda *args: ToyEmbedder(str(checkpoint))
    )
    output = tmp_path / "embedded"

    data_main(
        [
            "embed",
            str(source),
            "--extractor",
            "pet",
            "--checkpoint",
            str(checkpoint),
            "--column",
            "custom_vector",
            "--batch-size",
            "1",
            "--output",
            str(output),
        ]
    )

    result = load_saved_dataset(output)
    assert result["train"]["id"] == ["a", "b"]
    assert result["valid"]["custom_vector"] == [[5, 1]]
    assert result["train"].features["custom_vector"].length == 2
    assert "custom_vector" not in load_saved_dataset(source)["train"].column_names
    metadata = load_dataset_metadata(output)["embeddings"]["custom_vector"]
    assert metadata["dimension"] == 2
    assert len(metadata["checkpoint_sha256"]) == 64
    printed = capsys.readouterr().out
    assert "custom_vector:" in printed and "input_dim: 2" in printed


def test_embedding_column_defaults_to_embedding_and_rejects_overwrite(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")
    embedder = ToyEmbedder(str(checkpoint))
    s = Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])
    dataset = DatasetDict(train=Dataset.from_dict({"structure": [s.to(fmt="json")]}))
    result = embed_dataset(dataset, embedder)
    assert result["train"]["embedding"] == [[4, 1]]
    with pytest.raises(ValueError, match="already exists"):
        embed_dataset(result, embedder)


def test_replace_column_preserves_rows_and_recomputes_embeddings(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")
    structures = [Structure(Lattice.cubic(a), ["Si"], [[0, 0, 0]]) for a in (3, 4)]
    dataset = DatasetDict(
        train=Dataset.from_dict(
            {
                "structure": [structure.to(fmt="json") for structure in structures],
                "material_id": ["first", "second"],
                "embedding": [[99.0, 99.0], [99.0, 99.0]],
            }
        )
    )

    result = embed_dataset(
        dataset,
        ToyEmbedder(str(checkpoint)),
        replace_column=True,
    )

    assert result["train"]["material_id"] == ["first", "second"]
    assert result["train"]["embedding"] == [[3, 1], [4, 1]]
    assert dataset["train"]["embedding"] == [[99.0, 99.0], [99.0, 99.0]]


def test_embed_cli_materializes_training_filter_and_validation_split(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")
    structures = [
        Structure(
            Lattice.cubic(4),
            ["Si"] * count,
            [[i / count] * 3 for i in range(count)],
        )
        for count in (1, 2, 3, 1, 2, 3)
    ]
    source = tmp_path / "source"
    save_dataset(
        DatasetDict(
            train=Dataset.from_dict(
                {
                    "structure": [structure.to(fmt="json") for structure in structures],
                    "id": list("abcdef"),
                }
            )
        ),
        source,
    )
    monkeypatch.setattr(
        "prisma.embeddings.create_embedder", lambda *args: ToyEmbedder(str(checkpoint))
    )
    output = tmp_path / "embedded"

    data_main(
        [
            "embed",
            str(source),
            "--extractor",
            "pet",
            "--checkpoint",
            str(checkpoint),
            "--max-num-atoms",
            "2",
            "--validation-fraction",
            "0.25",
            "--seed",
            "42",
            "--output",
            str(output),
        ]
    )

    result = load_saved_dataset(output)
    expected = Dataset.from_dict(
        {
            "id": ["a", "b", "d", "e"],
        }
    ).train_test_split(test_size=0.25, seed=42)
    assert result["train"]["id"] == expected["train"]["id"]
    assert result["valid"]["id"] == expected["test"]["id"]
    report = load_dataset_metadata(output)["materialized_training_splits"]
    assert report == {
        "source_rows": {"train": 6},
        "max_num_atoms": 2,
        "eligible_rows": {"train": 4},
        "validation_fraction": 0.25,
        "split_seed": 42,
        "output_rows": {"train": 3, "valid": 1},
    }


def test_empty_optional_split_preserves_embedding_schema(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")
    s = Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])
    train = Dataset.from_dict({"structure": [s.to(fmt="json")]})
    dataset = DatasetDict(train=train, test=train.select([]))
    result = embed_dataset(dataset, ToyEmbedder(str(checkpoint)))
    assert len(result["test"]) == 0
    assert result["test"].features["embedding"] == result["train"].features["embedding"]


def test_unsupported_elements_are_removed_without_losing_other_rows(
    tmp_path, monkeypatch, capsys
):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")

    class LimitedEmbedder(ToyEmbedder):
        supported_atomic_numbers = frozenset({14})

        def __init__(self, path):
            super().__init__(path)
            self.dimension = 2

    structures = [
        Structure(Lattice.cubic(4), [element], [[0, 0, 0]])
        for element in ("Th", "Th", "Si", "Si")
    ]
    source = tmp_path / "source"
    save_dataset(
        DatasetDict(
            train=Dataset.from_dict(
                {
                    "structure": [s.to(fmt="json") for s in structures],
                    "id": ["a", "b", "c", "d"],
                }
            )
        ),
        source,
    )
    monkeypatch.setattr(
        "prisma.embeddings.create_embedder",
        lambda *args: LimitedEmbedder(str(checkpoint)),
    )
    output = tmp_path / "embedded"
    data_main(
        [
            "embed",
            str(source),
            "--extractor",
            "pet",
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            "2",
            "--output",
            str(output),
        ]
    )

    saved = load_saved_dataset(output)["train"]
    assert saved["id"] == ["c", "d"]
    assert saved["embedding"] == [[4, 1], [4, 1]]
    report = load_dataset_metadata(output)["embedding_filter"]["embedding"]["train"]
    assert report == {
        "count": 2,
        "by_atomic_number": {"90": 2},
        "example_row_indices": [0, 1],
    }
    assert "2 skipped" in capsys.readouterr().out


def test_all_unsupported_train_rows_fail_without_publishing(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")
    embedder = ToyEmbedder(str(checkpoint))
    embedder.supported_atomic_numbers = frozenset({14})
    embedder.dimension = 2
    th = Structure(Lattice.cubic(4), ["Th"], [[0, 0, 0]])
    dataset = DatasetDict(train=Dataset.from_dict({"structure": [th.to(fmt="json")]}))
    with pytest.raises(ValueError, match="No structures remain in split 'train'"):
        embed_dataset(dataset, embedder)


@pytest.mark.parametrize("values", [[[float("nan"), 0]], [[1, 2], [3]], [1, 2]])
def test_invalid_vectors_are_rejected(values):
    with pytest.raises(ValueError):
        validate_embeddings(values, count=1, dimension=2)


def test_failed_extraction_does_not_publish_a_partial_dataset(tmp_path, monkeypatch):
    source = tmp_path / "source"
    save_dataset(
        DatasetDict(train=Dataset.from_dict({"structure": ["invalid structure"]})),
        source,
    )
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(
        "prisma.embeddings.create_embedder", lambda *args: ToyEmbedder(str(checkpoint))
    )
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        data_main(
            [
                "embed",
                str(source),
                "--extractor",
                "pet",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()
