import json
from dataclasses import dataclass
from typing import Any

import polars as pl
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun

from vaspoperator.calculation.base import StepBase, StepConfigBase
from vaspoperator.globals.logger import logged
from vaspoperator.input.incar import create_and_save_incar
from vaspoperator.input.kgen import create_and_save_kgen
from vaspoperator.input.kpoints import create_and_save_kpoints
from vaspoperator.input.poscar import create_and_save_poscar
from vaspoperator.input.potcar import create_and_save_potcar
from vaspoperator.input.readme import create_and_save_readme
from vaspoperator.input.run_sh import create_and_save_run_script
from vaspoperator.slurm.monitor import submit_and_monitor


@dataclass
class StepConfigREL(StepConfigBase):
    """
    Configuration for the Relaxation (REL) step.

    Inherits all base fields (material_id, incar, server_config, etc.)
    directly from StepConfigBase.
    """

    pass


@logged(name="REL Step")
class StepREL(StepBase):
    """
    Execution class for VASP Structure Relaxation calculations.

    This class orchestrates the entire lifecycle of a structural relaxation:
    1. Generating necessary VASP input files (INCAR, POSCAR, POTCAR, KPOINTS).
    2. Generating a primitive cell via SUMO if required.
    3. Submitting the job to the SLURM scheduler and monitoring its status.
    4. Parsing the outputs (`vasprun.xml`, `OUTCAR`) and saving them as Parquet files.
    """

    def __init__(self, structure: Structure, config: StepConfigREL):
        """
        Initializes the REL step.

        Args:
            structure (Structure): The initial crystal structure to be relaxed.
            config (StepConfigREL): Configuration parameters for this calculation.
        """
        self.debug(f"Initializing REL step for material: {config.material_id}")
        self.structure_initial = structure
        self.config = config

        # Consolidate path definitions
        self.step_folder = (
            self.config.calculation_dir
            / self.config.material_id
            / str(self.config.step_prefix)
        )
        self.debug(f"REL step initialized. Target folder: {self.step_folder}")

    def generate_input(self) -> None:
        """
        Generates all required VASP input files in the calculation directory.

        This method creates the standard VASP suite (INCAR, POSCAR, KPOINTS, POTCAR),
        a run script, and attempts to generate a primitive cell using SUMO. If a
        primitive cell is successfully generated, it updates the POSCAR and KPOINTS
        accordingly.
        """
        self.info(f"Starting input file generation in: {self.step_folder}")

        # Cleanly create all necessary parent directories in one go
        self.step_folder.mkdir(parents=True, exist_ok=True)
        self.debug("Calculation directory verified/created.")

        # Core VASP Inputs
        create_and_save_incar(params=self.config.incar, folder=self.step_folder)
        self.debug("INCAR generated.")

        create_and_save_poscar(
            structure=self.structure_initial, folder=self.step_folder
        )
        self.debug("Initial POSCAR generated.")

        create_and_save_kpoints(
            structure=self.structure_initial,
            kppa=self.config.kppa,
            folder=self.step_folder,
        )
        self.debug("Initial KPOINTS generated.")

        create_and_save_potcar(
            structure=self.structure_initial, folder=self.step_folder
        )
        self.debug("POTCAR generated.")

        # Metadata and Execution Scripts
        create_and_save_readme(
            structure=self.structure_initial,
            folder=self.step_folder,
            kppa=self.config.kppa,
            id=self.config.material_id,
            step=str(self.config.step_prefix),
        )

        vasp_conf = self.config.server_config["vasp"]
        create_and_save_run_script(
            id=self.config.material_id,
            n_cpus=vasp_conf["n_cpus"],
            n_nodes=vasp_conf["n_nodes"],
            folder=self.step_folder,
            step=str(self.config.step_prefix),
            max_duration=vasp_conf["max_duration"],
            cluster_part=vasp_conf["cluster_part"],
            unavailable_nodes=vasp_conf["unavailable_nodes"],
            is_exclusive=vasp_conf["is_exclusive"],
        )
        self.debug("SLURM run.sh script and README generated.")

        # Primitive cell generation via SUMO
        self.debug("Attempting primitive cell generation via SUMO kgen...")
        create_and_save_kgen(
            folder=self.step_folder,
            sumo_kgen_params=self.config.sumo_config["kgen"],
        )

        poscar_prim_path = self.step_folder / "POSCAR_prim"
        if poscar_prim_path.exists():
            self.info(
                "Primitive cell generated successfully. Updating POSCAR and KPOINTS."
            )
            self.structure_primitive = Structure.from_file(poscar_prim_path)
            create_and_save_poscar(
                structure=self.structure_primitive, folder=self.step_folder
            )
        else:
            self.warning(
                "No primitive cell generated. Falling back to the initial structure."
            )
            self.structure_primitive = self.structure_initial

        # Final KPOINTS based on primitive cell
        create_and_save_kpoints(
            structure=self.structure_primitive,
            kppa=self.config.kppa,
            folder=self.step_folder,
        )
        self.debug("Final KPOINTS created and saved.")
        self.info("All input files successfully generated.")

    def submit_and_monitor(self) -> None:
        """
        Submits the SLURM job and blocks execution until completion or timeout.
        """
        monitor_conf = self.config.server_config["monitor"]
        self.info(f"Submitting SLURM job from {self.step_folder}...")

        submit_and_monitor(
            script_path=self.step_folder / "run.sh",
            timeout_seconds=monitor_conf["timeout_seconds"],
            check_interval=monitor_conf["check_interval"],
        )
        self.info("Job monitoring concluded (completed or timed out).")

    def extract_data_dict(self) -> dict[str, Any]:
        """
        Parses `vasprun.xml` and `OUTCAR` to extract physical properties.

        Returns:
            Dict[str, Any]: A dictionary containing convergence status, energies,
            bandgap info, magnetic moments, and structural data. If parsing fails,
            returns a dictionary with `None` for all values to maintain schema integrity.
        """
        self.info("Extracting physical data from VASP outputs...")

        null_data = {
            "is_succeed": False,
            "energy": None,
            "is_spin": None,
            "e_fermi": None,
            "incar": None,
            "total_mag": None,
            "bandgap": None,
            "bandgap_direct": None,
            "bandgap_cbm": None,
            "bandgap_vbm": None,
            "is_gap_direct": None,
            "structure_final": None,
            "structure_initial": None,
            "run_stats": None,
        }

        try:
            self.debug(f"Parsing vasprun.xml and OUTCAR in {self.step_folder}")
            vasprun = Vasprun(self.step_folder / "vasprun.xml")
            outcar = Outcar(self.step_folder / "OUTCAR")

            is_converged = vasprun.converged

            if not is_converged:
                self.warning(
                    f"VASP calculation did not converge for {self.config.material_id}."
                )
                return {
                    **null_data,
                    "is_succeed": False,
                    "run_stats": outcar.run_stats,
                }

            bs = vasprun.get_band_structure(efermi="smart")

            self.info("Successfully extracted and validated VASP data.")
            return {
                "is_succeed": is_converged,
                "energy": vasprun.final_energy,
                "is_spin": vasprun.is_spin,
                "e_fermi": outcar.efermi,
                "incar": vasprun.incar.to_json(),
                "total_mag": outcar.total_mag,
                "bandgap": bs.get_band_gap()["energy"],
                "bandgap_direct": float(bs.get_direct_band_gap()),
                "bandgap_cbm": bs.get_cbm()["energy"],
                "bandgap_vbm": bs.get_vbm()["energy"],
                "is_gap_direct": bs.get_band_gap()["direct"],
                "structure_final": vasprun.final_structure.to_json(),
                "structure_initial": vasprun.initial_structure.to_json(),
                "run_stats": outcar.run_stats,
            }
        except Exception as e:
            self.error(
                f"Critical failure while extracting VASP data: {e}",
                exc_info=True,
            )
            return null_data

    def process_data(self) -> bool:
        """
        Transforms extracted calculation data into Polars DataFrames and saves
        them as Parquet files for downstream analysis.

        Returns:
            bool: True upon successful serialization.
        """
        self.info("Processing extracted data into Parquet format...")
        data_dict = self.extract_data_dict()
        timestamp = pl.lit(self.config.date).dt.datetime()

        # Isolate physical data vs run stats
        physical_data = {k: v for k, v in data_dict.items() if k != "run_stats"}

        df_data = pl.DataFrame(physical_data).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            timestamp.alias("TS"),
        )

        df_stats_data = data_dict["run_stats"] or {}
        df_stats = pl.DataFrame(df_stats_data).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            timestamp.alias("TS"),
        )

        # Setup output directories
        data_dir = self.config.results_dir / "rel_data"
        stats_dir = self.config.results_dir / "run_stats"
        data_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)

        # Construct file names
        time_str = self.config.date.strftime("%Y%m%d_%H%M%S")
        base_name = f"_{time_str}_{self.config.material_id}.parquet"
        data_file_path = data_dir / f"{self.config.step_prefix}_data{base_name}"
        stats_file_path = (
            stats_dir / f"{self.config.step_prefix}_run_stats{base_name}"
        )

        # Write to disk
        df_data.write_parquet(data_file_path)
        self.debug(f"Physical data saved to {data_file_path}")

        df_stats.write_parquet(stats_file_path)
        self.debug(f"Run statistics saved to {stats_file_path}")

        self.info("Data processing and Parquet serialization complete.")
        return True

    def get_results(self) -> dict[str, Any]:
        """
        Loads the processed Parquet data from disk and deserializes the final structure.

        Returns:
            Dict[str, Any]: A dictionary containing the boolean success flag
            and the deserialized pymatgen Structure object.
        """
        time_str = self.config.date.strftime("%Y%m%d_%H%M%S")
        file_path = (
            self.config.results_dir
            / "rel_data"
            / f"{self.config.step_prefix}_data_{time_str}_{self.config.material_id}.parquet"
        )

        self.debug(f"Retrieving serialized results from {file_path}")

        try:
            df_data = pl.read_parquet(file_path)
            final_struct_str = df_data[0, "structure_final"]

            results = {
                "is_succeed": df_data[0, "is_succeed"],
                "structure_final": (
                    Structure.from_dict(json.loads(final_struct_str))
                    if final_struct_str is not None
                    else None
                ),
            }
            self.info("Successfully retrieved and deserialized results.")
            return results

        except Exception as e:
            self.error(
                f"Failed to load results from {file_path}: {e}", exc_info=True
            )
            return {"is_succeed": False, "structure_final": None}
