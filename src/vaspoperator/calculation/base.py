from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from pymatgen.core import Structure

from vaspoperator.globals.logger import logged


@dataclass
class StepConfigBase:
    """Configuration container for a single VASP calculation step."""

    material_id: str
    incar: dict[str, Any]
    server_config: dict[str, Any]
    sumo_config: dict[str, Any]
    calculation_dir: Path
    results_dir: Path
    step_prefix: str
    kppa: int
    date: datetime = field(default_factory=datetime.now)


@logged(name="StepBase")
class StepBase:
    """
    Base class for all VASP calculation steps (e.g., REL, SCF, BANDS).

    Provides the standard interface for generating inputs, managing job
    submission, and defining data schemas for results.
    """

    def __init__(self, structure: Structure, config: StepConfigBase):
        """
        Initializes the step with a structure and configuration.

        Args:
            structure (Structure): The crystal structure to use.
            config (StepConfigBase): Configuration parameters for the step.
        """
        self.debug(
            f"Initializing {self.__class__.__name__} for {config.material_id}"
        )
        self.structure_initial = structure
        self.config = config
        self.debug(f"{self.__class__.__name__} initialized.")

    def generate_input(self) -> None:
        """Abstract method to generate VASP input files (INCAR, POSCAR, etc.)."""
        pass

    def submit_and_monitor(self) -> None:
        """Abstract method to submit the Slurm job and wait for completion."""
        pass

    def process_data(self) -> bool:
        """
        Processes the results of the calculation.

        Returns:
            bool: True if data was processed successfully, False otherwise.
        """
        return True

    @staticmethod
    def get_polars_schema() -> dict[str, pl.Schema]:
        """
        Defines the Polars schemas for calculation statistics and physical data.

        Returns:
            dict[str, pl.Schema]: A dictionary containing 'run_stats' and 'data' schemas.
        """
        return {
            "run_stats": pl.Schema(
                {
                    "material_id": pl.String,
                    "step": pl.String,
                    "TS": pl.Datetime(time_unit="us"),
                    "Average memory used (kb)": pl.Float64,
                    "Maximum memory used (kb)": pl.Float64,
                    "Elapsed time (sec)": pl.Float64,
                    "System time (sec)": pl.Float64,
                    "User time (sec)": pl.Float64,
                    "Total CPU time used (sec)": pl.Float64,
                    "cores": pl.Int64,
                }
            ),
            "data": pl.Schema(
                {
                    "material_id": pl.String,
                    "step": pl.String,
                    "TS": pl.Datetime(time_unit="us"),
                    "is_succeed": pl.Boolean,
                    "energy": pl.Float64,
                    "is_spin": pl.Boolean,
                    "e_fermi": pl.Float64,
                    "incar": pl.String,
                    "total_mag": pl.Float64,
                    "bandgap": pl.Float64,
                    "bandgap_direct": pl.Float64,
                    "bandgap_cbm": pl.Float64,
                    "bandgap_vbm": pl.Float64,
                    "is_gap_direct": pl.Boolean,
                    "structure_final": pl.String,
                    "structure_initial": pl.String,
                }
            ),
        }
