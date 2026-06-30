from typing import cast

from pymatgen.core import Structure


def _convert_structure_json(structure_json: str) -> Structure:
    s = cast(Structure, Structure.from_str(structure_json, fmt="json"))
    return s
