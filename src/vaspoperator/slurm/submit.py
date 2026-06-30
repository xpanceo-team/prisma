import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("SLURM-Monitor")


def submit_job(script_path: str | Path) -> int | None:
    """
    Submits a bash script to the SLURM scheduler and returns the Job ID.
    """
    script_path = str(script_path)
    try:
        result = subprocess.run(
            ["sbatch", script_path], capture_output=True, text=True, check=True
        )

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = int(match.group(1))
            logger.info(
                f"Successfully submitted job {job_id} for {script_path}"
            )
            return job_id

        logger.error(f"Failed to parse SLURM output: {result.stdout}")
        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"SBATCH Command Failed: {e.stderr}")
        return None
