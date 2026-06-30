"""Module for generating Slurm submission scripts for VASP calculations."""

import logging
import re
from pathlib import Path

# Standard logging setup
logger = logging.getLogger(__name__)


def check_vasp_input_params(
    id: str,
    n_cpus: int,
    n_nodes: int,
    max_duration: str = "01:30:00",
    cluster_part: str = "mpi",
    unavailable_nodes: str | None = None,
) -> None:
    """
    Validate input parameters for VASP job generation.

    Args:
        id (str): Job identifier.
        n_cpus (int): Number of CPU cores.
        n_nodes (int): Number of compute nodes.
        max_duration (str, optional): Maximum walltime in HH:MM:SS. Defaults to "01:30:00".
        cluster_part (str, optional): Slurm partition. Defaults to "mpi".
        unavailable_nodes (str, optional): Nodes to exclude. Defaults to None.

    Raises:
        ValueError: If ID is empty, or if CPU/Node counts are less than 1.
    """
    if not id or not id.strip():
        logger.error("Job ID cannot be empty.")
        raise ValueError("Job ID cannot be empty.")

    if n_cpus <= 0:
        logger.error(f"Invalid n_cpus value: {n_cpus}. Must be > 0.")
        raise ValueError(f"n_cpus must be > 0, got {n_cpus}.")

    if n_nodes <= 0:
        logger.error(f"Invalid n_nodes value: {n_nodes}. Must be > 0.")
        raise ValueError(f"n_nodes must be > 0, got {n_nodes}.")

    if not re.match(r"^\d{2}:\d{2}:\d{2}$", max_duration):
        logger.warning(
            f"max_duration '{max_duration}' may not be in exact HH:MM:SS format."
        )


def create_run_script(
    id: str,
    n_cpus: int,
    n_nodes: int,
    directory: str,
    step: str,
    max_duration: str = "01:30:00",
    cluster_part: str = "mpi",
    unavailable_nodes: str | None = None,
    is_exclusive: bool = True,
) -> str:
    """
    Generate a Slurm job submission script for VASP calculations.

    Args:
        id (str): Job identifier, used in the job name (vasp_{id}).
        n_cpus (int): Number of CPU cores to allocate.
        n_nodes (int): Number of compute nodes to request.
        directory (str): The working directory for the job.
        step (str): The calculation step (e.g., 'REL', 'SCF'). Used for logging/context.
        max_duration (str, optional): Maximum walltime (HH:MM:SS). Defaults to "01:30:00".
        cluster_part (str, optional): Slurm partition/queue. Defaults to "mpi".
        unavailable_nodes (str, optional): Nodes to exclude (e.g., "cnode05"). Defaults to None.
        is_exclusive (bool, optional): Whether to request exclusive node access. Defaults to True.

    Returns:
        str: The complete Slurm submission bash script.
    """
    check_vasp_input_params(
        id=id,
        n_cpus=n_cpus,
        n_nodes=n_nodes,
        max_duration=max_duration,
        cluster_part=cluster_part,
        unavailable_nodes=unavailable_nodes,
    )

    # Build script lines dynamically to prevent empty spaces for omitted flags
    lines = ["#!/bin/sh", "#SBATCH -e slurm-%j.err"]

    if unavailable_nodes:
        lines.append(f"#SBATCH --exclude={unavailable_nodes}")
        logger.info(f"Excluding nodes: {unavailable_nodes}")

    lines.extend(
        [
            f"#SBATCH -p {cluster_part}",
            f"#SBATCH -J vasp_{id}",
            f"#SBATCH -N {n_nodes}",
            f"#SBATCH -n {n_cpus}",
            f"#SBATCH --time={max_duration}",
        ]
    )

    if is_exclusive:
        lines.append("#SBATCH --exclusive")

    lines.extend(
        [
            f"#SBATCH --chdir={directory}/",
            "",
            f"srun vasp_std > vasp_{id}.log",
            "",  # Trailing newline
        ]
    )

    script = "\n".join(lines)

    logger.info(
        f"Slurm script generated successfully for step '{step}' ({len(lines)} lines)"
    )
    logger.debug(f"Script content:\n{script}")

    return script


def save_run_script(script: str, folder: Path) -> None:
    """
    Saves the generated Slurm script to 'run.sh'.

    Args:
        script (str): The bash script content.
        folder (Path): The target directory to save the file.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    target_path = folder / "run.sh"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target_path.write_text(script)
        logger.info(f"Submission script saved to {target_path}")
    except Exception as e:
        logger.error(f"Failed to save run.sh to {folder}: {e}", exc_info=True)
        raise


def create_and_save_run_script(
    id: str,
    n_cpus: int,
    n_nodes: int,
    folder: Path,
    step: str,
    max_duration: str = "01:30:00",
    cluster_part: str = "mpi",
    unavailable_nodes: str | None = None,
    is_exclusive: bool = True,
) -> None:
    """
    High-level wrapper to generate and save a Slurm run script.

    Args:
        id (str): Job identifier.
        n_cpus (int): Number of CPU cores.
        n_nodes (int): Number of compute nodes.
        folder (Path): Target directory.
        step (str): The calculation step.
        max_duration (str, optional): Walltime. Defaults to "01:30:00".
        cluster_part (str, optional): Slurm partition. Defaults to "mpi".
        unavailable_nodes (str, optional): Excluded nodes. Defaults to None.
        is_exclusive (bool, optional): Request exclusive access. Defaults to True.
    """
    logger.info(f"Generating and saving run script for {id} in {folder}")
    run_script = create_run_script(
        id=id,
        n_cpus=n_cpus,
        n_nodes=n_nodes,
        directory=str(folder.absolute()),
        step=step,
        max_duration=max_duration,
        cluster_part=cluster_part,
        unavailable_nodes=unavailable_nodes,
        is_exclusive=is_exclusive,
    )
    save_run_script(script=run_script, folder=folder)


def main(
    id: str = "test",
    n_cpus: int = 24,
    n_nodes: int = 2,
    max_duration: str = "01:30:00",
    cluster_part: str = "mpi",
    unavailable_nodes: str | None = "[17-22,40]",
    log_level: str = "INFO",
) -> None:
    """
    Main function for the CLI interface.
    """
    # Attempt to load custom logger, fallback to basic logging
    try:
        from vaspoperator.globals.logger import setup_logging

        setup_logging()
    except ImportError:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO)
        )
        logger.warning("Custom logger not found; using fallback basicConfig.")

    current_dir = Path.cwd()

    # Use the combined generation and saving utility so output is visible
    create_and_save_run_script(
        id=id,
        n_cpus=n_cpus,
        n_nodes=n_nodes,
        folder=current_dir,
        step="CLI_TEST",
        max_duration=max_duration,
        cluster_part=cluster_part,
        unavailable_nodes=unavailable_nodes,
    )


if __name__ == "__main__":
    try:
        import fire

        fire.Fire(main)
    except ImportError:
        logger.warning(
            "The 'fire' library is missing. Running main with default arguments."
        )
        main()
