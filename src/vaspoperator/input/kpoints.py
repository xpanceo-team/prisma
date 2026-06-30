import logging
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints

# Logging configuration
logger = logging.getLogger(__name__)


def create_kpoints(structure: Structure, kppa: int) -> Kpoints:
    """
    Generates a VASP KPOINTS object based on a target K-point density per reciprocal atom.

    Args:
        structure (Structure): The pymatgen Structure object.
        kppa (int): K-points per atom.

    Returns:
        Kpoints: A pymatgen Kpoints object configured for automatic density.
    """
    logger.debug(
        f"Generating K-points with KPPA: {kppa} for structure: {structure.formula}"
    )
    kpoints = Kpoints.automatic_density(structure, kppa=kppa)
    return kpoints


def save_kpoints(kpoints: Kpoints, folder: Path) -> None:
    """
    Writes the KPOINTS object to a file in the specified directory.

    Args:
        kpoints (Kpoints): The Kpoints object to save.
        folder (Path): The directory path where the KPOINTS file will be written.

    Raises:
        OSError: If there is an issue creating the directory or writing the file.
    """
    target_path = folder / "KPOINTS"
    try:
        # Ensure directory exists
        folder.mkdir(parents=True, exist_ok=True)

        kpoints.write_file(target_path)
        logger.info(f"KPOINTS successfully saved to {target_path}")
    except Exception as e:
        logger.error(f"Failed to save KPOINTS to {folder}: {e}")
        raise


def create_and_save_kpoints(
    structure: Structure, kppa: int, folder: Path
) -> None:
    """
    A high-level wrapper to generate and save K-points in one step.

    Args:
        structure (Structure): The pymatgen Structure object.
        kppa (int): K-points per atom.
        folder (Path): The directory path where the KPOINTS file will be written.
    """
    logger.info(f"Creating and saving K-points in {folder}")
    kpoints = create_kpoints(structure=structure, kppa=kppa)
    save_kpoints(kpoints=kpoints, folder=folder)
