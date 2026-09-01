import pandas as pd
from datasets import Dataset, DatasetDict
from pymatgen.core import Lattice, Structure

from prisma.data import load_saved_dataset, prepare_dataset, save_dataset


def test_prepare_dataframe_normalizes_structures_and_split_column():
    structures = [
        Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]]),
        Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5] * 3]),
    ]
    frame = pd.DataFrame(
        {
            "source_id": ["001", "002"],
            "crystal": [
                structures[0].to(fmt="cif"),
                structures[1].to(fmt="poscar"),
            ],
            "bandgap": [1.1, 5.0],
            "split": ["train", "valid"],
        }
    )

    prepared = prepare_dataset(
        frame,
        structure_column="crystal",
        structure_format="auto",
        id_column="source_id",
        split_column="split",
    )

    assert list(prepared) == ["train", "valid"]
    assert prepared["train"][0]["material_id"] == "001"
    assert "crystal" not in prepared["train"].column_names
    assert Structure.from_str(
        prepared["valid"][0]["structure"], fmt="json"
    ) == structures[1]


def test_prepare_save_and_load_datasetdict_round_trip(tmp_path):
    structure = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    source = Dataset.from_dict(
        {
            "structure": [structure.to(fmt="json")] * 4,
            "property": [1.0, 2.0, 3.0, 4.0],
        }
    )
    prepared = prepare_dataset(source, validation_fraction=0.25, seed=42)

    output_path = save_dataset(prepared, tmp_path / "materials")
    restored = load_saved_dataset(output_path)

    assert isinstance(restored, DatasetDict)
    assert set(restored) == {"train", "valid"}
    assert len(restored["train"]) == 3
    assert len(restored["valid"]) == 1


def test_prepare_csv_preserves_string_ids_and_multiline_structures(tmp_path):
    structure = Structure(Lattice.cubic(3.5), ["Si"], [[0, 0, 0]])
    source_path = tmp_path / "materials.csv"
    pd.DataFrame(
        {
            "material_id": ["001"],
            "structure": [structure.to(fmt="cif")],
            "property": [1.0],
        }
    ).to_csv(source_path, index=False)

    prepared = prepare_dataset(source_path, structure_format="cif")

    assert prepared["train"][0]["material_id"] == "001"
    assert Structure.from_str(
        prepared["train"][0]["structure"], fmt="json"
    ) == structure
