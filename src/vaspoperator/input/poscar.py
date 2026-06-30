import logging
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# Standard logging setup
logger = logging.getLogger(__name__)


def create_poscar(structure: Structure, comment: str | None = None) -> Poscar:
    """
    Generates a VASP POSCAR object from a pymatgen Structure.

    Args:
        structure (Structure): The pymatgen Structure object to convert.
        comment (str, optional): A comment line for the top of the POSCAR file.
            Defaults to None.

    Returns:
        Poscar: A pymatgen Poscar object.
    """
    logger.debug(f"Creating POSCAR for structure: {structure.formula}")
    return Poscar(structure, comment=comment)


def save_poscar(poscar: Poscar, folder: Path) -> None:
    """
    Saves a POSCAR object to a file named 'POSCAR' in the target directory.

    Args:
        poscar (Poscar): The Poscar object to write.
        folder (Path): The directory path where the POSCAR will be saved.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    target_path = folder / "POSCAR"
    try:
        # Create directory if it doesn't exist
        folder.mkdir(parents=True, exist_ok=True)

        poscar.write_file(target_path)
        logger.info(f"POSCAR successfully saved to {target_path}")
    except Exception as e:
        logger.error(f"Failed to save POSCAR to {folder}: {e}", exc_info=True)
        raise


def create_and_save_poscar(structure: Structure, folder: Path) -> None:
    """
    High-level wrapper to generate and save a POSCAR file in one operation.

    Args:
        structure (Structure): The pymatgen Structure object.
        folder (Path): The directory where the POSCAR file should be stored.
    """
    logger.info(f"Processing POSCAR for {structure.formula} in {folder}")
    poscar = create_poscar(structure=structure)
    save_poscar(poscar=poscar, folder=folder)
