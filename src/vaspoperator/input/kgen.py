import logging
from pathlib import Path
from typing import Any

from sumo.cli.kgen import kgen

# Standard logging setup
logger = logging.getLogger(__name__)


def create_and_save_kgen(
    folder: Path, sumo_kgen_params: dict[str, Any]
) -> None:
    """
    Generates and saves a high-symmetry KPOINTS file using Sumo's k-point generator.

    This function acts as a wrapper around `sumo.cli.kgen.kgen`. It expects
    a valid POSCAR file to already exist in the target folder, as Sumo uses it
    to determine the standard high-symmetry k-point path for band structures.

    Args:
        folder (Path): The directory containing the 'POSCAR' file. The generated
            'KPOINTS' and 'band.conf' (if requested) will be saved here.
        sumo_kgen_params (dict[str, Any]): Additional keyword arguments to pass
            to the Sumo `kgen` function (e.g., density, line_density, mode).

    Raises:
        FileNotFoundError: If the 'POSCAR' file is not found in the specified folder.
        Exception: If the Sumo kgen execution fails.
    """
    poscar_path = folder / "POSCAR"

    # Pre-execution check: Sumo will crash if the structure file is missing
    if not poscar_path.exists():
        logger.error(
            f"POSCAR not found in {folder}. Sumo kgen requires a structure file."
        )
        raise FileNotFoundError(
            f"Missing required file: {poscar_path.absolute()}"
        )

    logger.info(f"Running Sumo kgen in {folder}")
    logger.debug(f"Sumo parameters: {sumo_kgen_params}")

    try:
        # Sumo's kgen handles both the creation and saving of the file internally
        kgen(
            filename=str(poscar_path), directory=str(folder), **sumo_kgen_params
        )
        logger.info(
            f"Sumo k-point generation completed successfully in {folder}"
        )
    except Exception as e:
        logger.error(f"Sumo kgen failed for {folder}: {e}", exc_info=True)
        raise
