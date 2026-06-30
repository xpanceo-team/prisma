import logging
import shutil
from pathlib import Path

# Standard logging setup
logger = logging.getLogger(__name__)


def clear_from_dat(directory: Path) -> dict[str, list[str]]:
    """
    Remove all .dat files from the specified directory.

    Args:
        directory (Path): Path to the directory for cleanup.

    Returns:
        dict[str, list[str]]: Summary containing 'deleted' and 'errors' lists.

    Raises:
        TypeError: If directory is not a Path object.
        FileNotFoundError: If the directory does not exist.
    """
    if not isinstance(directory, Path):
        raise TypeError(f"Expected Path object, got {type(directory).__name__}")
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found or is not a folder: {directory}"
        )

    logger.info(f"Cleaning .dat files in: {directory}")
    summary = {"deleted": [], "errors": []}

    for file_path in directory.glob("*.dat"):
        try:
            file_path.unlink()
            summary["deleted"].append(str(file_path))
            logger.debug(f"Deleted: {file_path.name}")
        except Exception as e:
            err = f"Failed to delete {file_path.name}: {e}"
            summary["errors"].append(err)
            logger.error(err)

    logger.info(
        f"Cleanup finished. Deleted: {len(summary['deleted'])}, Errors: {len(summary['errors'])}"
    )
    return summary


def copy_file_between_stages_multi(
    folder: Path, filename: str, step_initial: str, steps_to_copy: list[str]
) -> dict[str, list[str]]:
    """
    Copies a specific file from an initial stage to multiple destination stages.

    This is commonly used in VASP workflows to pass the CONTCAR (relaxed structure)
    of one step to the POSCAR (input structure) of subsequent steps.

    Args:
        folder (Path): The root project directory.
        filename (str): Name of the file to copy (e.g., 'CONTCAR').
        step_initial (str): The source directory name (e.g., 'REL').
        steps_to_copy (list[str]): list of target directory names (e.g., ['SCF', 'DOS']).

    Returns:
        dict[str, list[str]]: Summary of 'successful' and 'errors' copy operations.
    """
    # Validation
    if not isinstance(folder, Path):
        raise TypeError(
            f"Expected Path for folder, got {type(folder).__name__}"
        )

    source_path = folder / step_initial / filename
    if not source_path.is_file():
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file missing: {source_path}")

    summary = {"successful": [], "errors": []}
    logger.info(
        f"Propagating {filename} from {step_initial} to {len(steps_to_copy)} stages."
    )

    for step_dest in steps_to_copy:
        dest_dir = folder / step_dest
        dest_path = dest_dir / filename

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, dest_path)

            msg = f"Copied {filename} to {step_dest}"
            summary["successful"].append(msg)
            logger.debug(msg)
        except Exception as e:
            err = f"Error copying to {step_dest}: {e}"
            summary["errors"].append(err)
            logger.error(err)

    logger.info(
        f"Copying complete. Success: {len(summary['successful'])}, Errors: {len(summary['errors'])}"
    )
    return summary
