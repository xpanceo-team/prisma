"""Module for creating VASP calculation README files."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from pymatgen.core import Structure

# Standard logging setup
logger = logging.getLogger(__name__)


def create_readme(step: str, structure: Structure, id: str, kppa: int) -> str:
    """
    Create a formatted README.md string with calculation details.

    Generates a Markdown string containing information about the VASP
    calculation setup, structure details, and calculation parameters.

    Args:
        step (str): Calculation step (e.g., 'REL', 'SCF', 'DOS').
        structure (Structure): Pymatgen Structure object containing the crystal.
        id (str): Structure identifier or job name.
        kppa (int): K-points density used for the calculation.

    Returns:
        str: Formatted README content as a string.
    """
    logger.debug(f"Generating README content for step: {step}, ID: {id}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        space_group = structure.get_space_group_info()[0]
    except Exception as e:
        space_group = "Unknown"
        logger.warning(
            f"Could not determine space group for {structure.formula}: {e}"
        )

    readme_content = f"""# VASP Calculation - {step} Step

## Calculation Type
{step}

## Structure Information
- **Formula**: {structure.formula}
- **Number of atoms**: {len(structure)}
- **Volume**: {structure.volume:.2f} Å³
- **Space group**: {space_group}

## Calculation Parameters
- **K-points density**: {kppa} kppa
- **ENCUT**: 500 eV
- **Precision**: Accurate

## Files Description
- **POSCAR**: Crystal structure
- **INCAR**: Calculation parameters
- **KPOINTS**: K-points mesh
- **POTCAR**: Pseudopotentials
- **README.md**: This file

## Next Steps
- Verify input files before submission
- Check for any warnings or errors
- Submit job to queue system

## Notes
- Generated automatically by VASP preparation script
- Structure ID: {id}
- Timestamp: {timestamp}
"""

    logger.info(
        f"README content generated successfully ({len(readme_content)} chars)"
    )
    return readme_content


def save_readme(readme: str, folder: Path) -> None:
    """
    Saves the README string to a file named 'README.md' in the target directory.

    Args:
        readme (str): The markdown string content.
        folder (Path): The directory path where the README will be saved.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    target_path = folder / "README.md"
    try:
        # Create directory if it doesn't exist
        folder.mkdir(parents=True, exist_ok=True)

        target_path.write_text(readme)
        logger.info(f"README successfully saved to {target_path}")
    except Exception as e:
        logger.error(f"Failed to save README to {folder}: {e}", exc_info=True)
        raise


def create_and_save_readme(
    step: str, structure: Structure, id: str, kppa: int, folder: Path
) -> None:
    """
    High-level wrapper to generate and save a README.md file in one operation.

    Args:
        step (str): Calculation step (e.g., 'REL', 'SCF', 'DOS').
        structure (Structure): Pymatgen Structure object.
        id (str): Structure identifier.
        kppa (int): K-points density used for the calculation.
        folder (Path): Destination directory for the README file.
    """
    logger.info(
        f"Processing README for {structure.formula} (ID: {id}) in {folder}"
    )
    readme = create_readme(step=step, structure=structure, id=id, kppa=kppa)
    save_readme(readme=readme, folder=folder)


def main(
    structure: Structure,
    id: str = "test",
    kppa: int = 500,
    step: str = "REL",
    folder: Path | None = None,
) -> None:
    """
    Main entry point to initialize logging, create, and save a VASP README.

    Args:
        structure (Structure): Pymatgen Structure object.
        id (str, optional): Structure identifier. Defaults to "test".
        kppa (int, optional): K-points density. Defaults to 500.
        step (str, optional): Calculation step. Defaults to "REL".
        folder (Path, optional): Output directory. Defaults to the current directory.

    Raises:
        ValueError: If invalid parameters are provided (e.g., negative kppa).
    """
    # Input validation
    if kppa <= 0:
        raise ValueError(f"kppa must be positive, got {kppa}")
    if not isinstance(structure, Structure):
        raise ValueError(
            f"structure must be a pymatgen Structure object, got {type(structure)}"
        )

    target_folder = folder or Path(".")

    # Custom logger setup attempt
    try:
        sys.path.append("/korotnev/lcdm/vaspoperator/src/")
        from vaspoperator.globals.logger import setup_logging

        setup_logging()
        logger.info("Custom logging configured successfully")
    except ImportError as e:
        # Fallback if the custom environment isn't available
        logging.basicConfig(level=logging.INFO)
        logger.warning(
            f"Could not import custom logger, using fallback. Error: {e}"
        )

    logger.info(
        f"Starting main README generation process for ID: {id}, Step: {step}"
    )

    try:
        # Utilize the unified wrapper function
        create_and_save_readme(
            step=step,
            structure=structure,
            id=id,
            kppa=kppa,
            folder=target_folder,
        )
    except Exception as e:
        logger.error(f"Failed to execute main README generation: {e}")
        raise
