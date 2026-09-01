from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pymatgen.core import Structure


StructureFormat = Literal["auto", "pymatgen-json", "cif", "poscar"]

_PYMATGEN_FORMATS = {
    "pymatgen-json": "json",
    "cif": "cif",
    "poscar": "poscar",
}


def normalize_structure(
    value: Any,
    *,
    structure_format: StructureFormat = "auto",
    context: str | None = None,
) -> str:
    """Convert a pymatgen mapping or supported structure text to JSON."""

    if structure_format not in {"auto", *_PYMATGEN_FORMATS}:
        supported = ", ".join(("auto", *_PYMATGEN_FORMATS))
        raise ValueError(
            f"Unsupported structure format {structure_format!r}. "
            f"Expected one of: {supported}."
        )

    context_text = f" ({context})" if context else ""
    if isinstance(value, Mapping):
        try:
            return Structure.from_dict(dict(value)).to(fmt="json")
        except Exception as exc:
            raise ValueError(
                f"Unable to parse pymatgen structure mapping{context_text}: {exc}"
            ) from exc

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Structure{context_text} must be non-empty text or a mapping."
        )

    formats = (
        tuple(_PYMATGEN_FORMATS)
        if structure_format == "auto"
        else (structure_format,)
    )
    errors = []
    for candidate in formats:
        try:
            structure = Structure.from_str(value, fmt=_PYMATGEN_FORMATS[candidate])
            return structure.to(fmt="json")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    attempted = "; ".join(errors)
    raise ValueError(
        f"Unable to parse structure{context_text} as {', '.join(formats)}. "
        f"Parser errors: {attempted}"
    )
