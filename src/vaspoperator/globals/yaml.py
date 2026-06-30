"""Module for loading YAML configuration files."""

import logging
from pathlib import Path
from typing import Any

import yaml

# Standard logging setup
logger = logging.getLogger(__name__)


def load_yaml_dict(yaml_path: str | Path) -> dict[str, Any]:
    """
    Load parameters from a YAML file into a Python dictionary.

    Uses safe_load to prevent arbitrary code execution. Handles Path objects
    and strings interchangeably.

    Args:
        yaml_path (Union[str, Path]): Path to the YAML file to load.

    Returns:
        fict[str, Any]: Dictionary containing the parsed YAML content.
            Returns an empty dictionary if the YAML file is empty.

    Raises:
        TypeError: If yaml_path is not a string or Path object.
        FileNotFoundError: If the specified YAML file does not exist.
        PermissionError: If the YAML file cannot be read.
        yaml.YAMLError: If the YAML file is malformed.
    """
    # Ensure we are working with a Path object
    path = Path(yaml_path)

    if not isinstance(yaml_path, (str, Path)):
        logger.error(f"Invalid yaml_path type: {type(yaml_path).__name__}")
        raise TypeError(f"Expected str or Path, got {type(yaml_path).__name__}")

    logger.info(f"Loading YAML file: {path}")

    try:
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f)

        if data is None:
            logger.warning(f"YAML file {path} is empty; returning empty dict.")
            return {}

        if not isinstance(data, dict):
            logger.warning(
                f"YAML content in {path} is {type(data).__name__}, wrapping in dict."
            )
            data = {"content": data}

        logger.info(
            f"Successfully loaded {len(data)} top-level keys from {path.name}"
        )
        return data

    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"I/O error loading {path}: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {path}: {e}")
        raise
