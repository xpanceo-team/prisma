import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("SLURM-Monitor")


def monitor_job(
    job_id: int, timeout_seconds: int = 36000, check_interval: int = 30
) -> dict[str, Any]:
    """
    Monitor a Slurm job with state detection and stuck-process warnings.
    """
    start_time = time.time()
    last_job_state = None
    no_change_count = 0
    max_no_change_checks = (
        100  # Alert if no change for ~50 mins at 30s interval
    )

    logger.info(f"Monitoring Job ID: {job_id} | Timeout: {timeout_seconds}s")

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            logger.error(f"Job {job_id} timed out after {elapsed:.1f}s")
            return {
                "completed": False,
                "timed_out": True,
                "exit_code": None,
                "state": "TIMEOUT",
                "runtime": elapsed,
                "stuck": False,
            }

        try:
            # Using scontrol provides more detail than squeue
            result = subprocess.run(
                ["scontrol", "show", "job", str(job_id)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Extract State
            state_match = re.search(r"JobState=(\S+)", result.stdout)
            current_state = state_match.group(1) if state_match else "UNKNOWN"

            # Handle State Transitions
            if current_state == last_job_state:
                no_change_count += 1
            else:
                no_change_count = 0
                last_job_state = current_state
                logger.debug(
                    f"Job {job_id} transitioned to state: {current_state}"
                )

            # Terminal States
            terminal_success = ["COMPLETED"]
            terminal_error = [
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "PREEMPTED",
                "OUT_OF_MEMORY",
            ]

            if current_state in (terminal_success + terminal_error):
                # Attempt to parse ExitCode (format is usually Code:Signal)
                exit_match = re.search(r"ExitCode=(\d+):(\d+)", result.stdout)
                exit_code = (
                    int(exit_match.group(1))
                    if exit_match
                    else (0 if current_state == "COMPLETED" else -1)
                )

                runtime = time.time() - start_time
                logger.info(
                    f"Job {job_id} finished. State: {current_state} | ExitCode: {exit_code}"
                )

                return {
                    "completed": (current_state == "COMPLETED"),
                    "timed_out": False,
                    "exit_code": exit_code,
                    "state": current_state,
                    "runtime": runtime,
                    "stuck": False,
                }

            # Stuck Detection for active jobs
            if no_change_count >= max_no_change_checks:
                logger.warning(
                    f"Job {job_id} likely stuck in '{current_state}' for {no_change_count * check_interval}s"
                )

        except subprocess.CalledProcessError as e:
            # If scontrol fails, the job might have been purged from the active controller
            # and moved to the accounting database (sacct).
            if "Invalid job id" in e.stderr:
                logger.info(
                    f"Job {job_id} cleared from active queue. Assuming completion."
                )
                return {
                    "completed": True,
                    "timed_out": False,
                    "exit_code": 0,
                    "state": "COMPLETED (purged)",
                    "runtime": time.time() - start_time,
                    "stuck": False,
                }
            logger.error(f"Scontrol error: {e.stderr}")

        time.sleep(check_interval)


def submit_and_monitor(
    script_path: Path, timeout_seconds: int = 36000, check_interval: int = 30
) -> dict[str, Any] | None:
    """
    Submits via sbatch and immediately begins monitoring.
    """
    try:
        sub_result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        match = re.search(r"Submitted batch job (\d+)", sub_result.stdout)
        if not match:
            logger.error("Could not parse Job ID from sbatch output.")
            return None

        job_id = int(match.group(1))
        logger.info(f"Job {job_id} submitted successfully.")

        res = monitor_job(job_id, timeout_seconds, check_interval)
        res["job_id"] = job_id
        return res

    except subprocess.CalledProcessError as e:
        logger.error(f"Submission failed: {e.stderr}")
        return None
