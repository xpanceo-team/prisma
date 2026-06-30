import logging
from pathlib import Path
from typing import Any

# Standard logging setup
logger = logging.getLogger(__name__)


def create_incar(params: dict[str, Any]) -> str:
    """
    Generates the content for a VASP INCAR file from a dictionary of parameters.

    Converts Python booleans to VASP format (.TRUE. / .FALSE.) and formats
    each key-value pair as 'KEY = VALUE'.

    Args:
        params (Dict[str, Any]): A dictionary containing INCAR tags as keys
            and their corresponding values.

    Returns:
        str: A formatted string representing the INCAR file content.
    """
    lines = []
    logger.debug(f"Processing {len(params)} parameters for INCAR generation.")

    for key, value in params.items():
        # Convert Python booleans to VASP logical format (.TRUE. / .FALSE.)
        if isinstance(value, bool):
            formatted_value = ".TRUE." if value else ".FALSE."
        else:
            formatted_value = str(value)

        lines.append(f"{key.upper()} = {formatted_value}")

    return "\n".join(lines) + "\n"


def save_incar(incar_content: str, folder: Path) -> None:
    """
    Saves the INCAR content to a file named 'INCAR' in the specified directory.

    Args:
        incar_content (str): The string content to be written to the file.
        folder (Path): The directory path where the INCAR file will be saved.

    Raises:
        OSError: If the directory does not exist or the file cannot be written.
    """
    target_path = folder / "INCAR"

    try:
        # Ensure the directory exists
        folder.mkdir(parents=True, exist_ok=True)

        target_path.write_text(incar_content)
        logger.info(f"Successfully saved INCAR to: {target_path.absolute()}")

    except Exception as e:
        logger.error(
            f"Critical failure writing INCAR to {folder}: {e}", exc_info=True
        )
        raise


def create_and_save_incar(params: dict[str, Any], folder: Path) -> None:
    """
    A wrapper function that creates INCAR content and saves it to a folder.

    Args:
        params (Dict[str, Any]): Dictionary of INCAR parameters.
        folder (Path): Directory where the file should be stored.
    """
    logger.info(f"Starting INCAR generation for folder: {folder}")
    incar_content = create_incar(params=params)
    save_incar(incar_content=incar_content, folder=folder)
