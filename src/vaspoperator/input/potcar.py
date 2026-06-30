import logging
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Potcar

# Standard logging setup
logger = logging.getLogger(__name__)


def get_potcar_symbols(
    structure: Structure, potcar_mapping: dict[str, str] | None = None
) -> list[str]:
    """
    Determines the appropriate POTCAR symbols for elements in a structure.

    The search order follows a preference for semi-core states if not specified:
    Custom mapping -> Standard -> _sv -> _s -> _pv.

    Args:
        structure (Structure): Pymatgen Structure object.
        potcar_mapping (Dict[str, str], optional): Custom mapping of element
            symbols to specific POTCAR variants (e.g., {"Li": "Li_sv"}).

    Returns:
        List[str]: A list of POTCAR symbols corresponding to the unique
            elements in the structure, in the order VASP expects.
    """
    mapping = potcar_mapping or {}
    # structure.symbol_set provides unique elements in the order they appear
    elements = structure.symbol_set
    symbols = []

    for el in elements:
        if el in mapping:
            chosen_symbol = mapping[el]
            logger.debug(f"Using mapped POTCAR for {el}: {chosen_symbol}")
            symbols.append(chosen_symbol)
            continue

        # Define search priority for VASP variants
        variants = [el, f"{el}_sv", f"{el}_s", f"{el}_pv"]
        found_variant = None

        for variant in variants:
            try:
                # Check if the POTCAR variant exists in the local environment
                Potcar([variant])
                found_variant = variant
                break
            except Exception:
                continue

        if found_variant:
            logger.info(f"Selected POTCAR variant for {el}: {found_variant}")
            symbols.append(found_variant)
        else:
            logger.warning(
                f"No special variant found for {el}, defaulting to standard."
            )
            symbols.append(el)

    return symbols


def create_potcar(
    structure: Structure, potcar_mapping: dict[str, str] | None = None
) -> Potcar:
    """
    Creates a Potcar object based on the elements present in the structure.

    Args:
        structure (Structure): Pymatgen Structure object.
        potcar_mapping (Dict[str, str], optional): Custom element-to-variant mapping.

    Returns:
        Potcar: A pymatgen Potcar object containing the concatenated potentials.
    """
    symbols = get_potcar_symbols(structure, potcar_mapping)
    return Potcar(symbols)


def save_potcar(potcar: Potcar, folder: Path) -> None:
    """
    Saves the POTCAR to the specified folder.

    Args:
        potcar (Potcar): The Potcar object to save.
        folder (Path): The directory path where 'POTCAR' will be written.
    """
    target_path = folder / "POTCAR"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        potcar.write_file(target_path)
        logger.info(f"Successfully saved POTCAR to {target_path}")
    except Exception as e:
        logger.error(f"Failed to write POTCAR to {folder}: {e}")
        raise


def create_and_save_potcar(
    structure: Structure,
    folder: Path,
    potcar_mapping: dict[str, str] | None = None,
) -> None:
    """
    High-level wrapper to resolve symbols, generate, and save a POTCAR file.

    Args:
        structure (Structure): Pymatgen Structure object.
        folder (Path): Destination directory.
        potcar_mapping (Dict[str, str], optional): Custom element-to-variant mapping.
    """
    logger.info(f"Generating POTCAR for {structure.formula}")
    potcar = create_potcar(structure=structure, potcar_mapping=potcar_mapping)
    save_potcar(potcar=potcar, folder=folder)
