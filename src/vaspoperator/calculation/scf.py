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
class StepConfigSCF(StepConfigBase):
    """Configuration for Self-Consistent Field (SCF) steps."""

    pass


@logged(name="SCF Step")
class StepSCF(StepBase):
    """
    Manages the execution of a VASP Self-Consistent Field (SCF) calculation.

    This step is critical for determining the converged charge density and
    electronic ground state, which serve as the foundation for further
    property calculations like Band Structure or DOS.
    """

    def __init__(self, structure: Structure, config: StepConfigSCF):
        """
        Initializes the SCF step.

        Args:
            structure (Structure): Initial crystal structure.
            config (StepConfigSCF): Configuration parameters.
        """
        self.debug(f"Initializing SCF for material: {config.material_id}")
        self.structure_initial = structure
        self.config = config
        self.step_folder = (
            self.config.calculation_dir
            / self.config.material_id
            / str(self.config.step_prefix)
        )
        self.debug(f"SCF initialized in: {self.step_folder}")

    def generate_input(self) -> None:
        """
        Generates VASP input files. Logic includes primitive cell
        standardization to ensure the k-point mesh is physically consistent.
        """
        self.info(f"Generating SCF inputs in {self.step_folder}")
        self.step_folder.mkdir(parents=True, exist_ok=True)

        # Basic VASP file suite
        create_and_save_incar(params=self.config.incar, folder=self.step_folder)
        create_and_save_poscar(
            structure=self.structure_initial, folder=self.step_folder
        )
        create_and_save_potcar(
            structure=self.structure_initial, folder=self.step_folder
        )

        # SLURM and Metadata
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
        create_and_save_readme(
            structure=self.structure_initial,
            folder=self.step_folder,
            kppa=self.config.kppa,
            id=self.config.material_id,
            step=str(self.config.step_prefix),
        )

        # Standardize cell (Primitive) for optimized K-Mesh
        create_and_save_kgen(
            folder=self.step_folder,
            sumo_kgen_params=self.config.sumo_config["kgen"],
        )

        poscar_prim = self.step_folder / "POSCAR_prim"
        if poscar_prim.exists():
            self.debug("Found POSCAR_prim, using for final POSCAR/KPOINTS.")
            self.structure_primitive = Structure.from_file(poscar_prim)
            create_and_save_poscar(
                structure=self.structure_primitive, folder=self.step_folder
            )
        else:
            self.structure_primitive = self.structure_initial

        create_and_save_kpoints(
            structure=self.structure_primitive,
            kppa=self.config.kppa,
            folder=self.step_folder,
        )
        self.info("SCF input generation complete.")

    def submit_and_monitor(self) -> None:
        """Submits the SCF job to SLURM and monitors completion."""
        mon = self.config.server_config["monitor"]
        self.info(f"Submitting SCF job: {self.config.material_id}")
        submit_and_monitor(
            script_path=self.step_folder / "run.sh",
            timeout_seconds=mon["timeout_seconds"],
            check_interval=mon["check_interval"],
        )

    def extract_data_dict(self) -> dict[str, Any]:
        """Parses results from vasprun.xml and OUTCAR."""
        self.debug("Extracting SCF results...")
        null_res = {
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
            vrun = Vasprun(self.step_folder / "vasprun.xml")
            ocar = Outcar(self.step_folder / "OUTCAR")

            if not vrun.converged:
                self.warning(
                    f"SCF for {self.config.material_id} did not converge."
                )
                return {**null_res, "run_stats": ocar.run_stats}

            bs = vrun.get_band_structure(efermi="smart")
            bg = bs.get_band_gap()

            return {
                "is_succeed": True,
                "energy": vrun.final_energy,
                "is_spin": vrun.is_spin,
                "e_fermi": ocar.efermi,
                "incar": vrun.incar.to_json(),
                "total_mag": ocar.total_mag,
                "bandgap": bg["energy"],
                "bandgap_direct": float(bs.get_direct_band_gap()),
                "bandgap_cbm": bs.get_cbm()["energy"],
                "bandgap_vbm": bs.get_vbm()["energy"],
                "is_gap_direct": bg["direct"],
                "structure_final": vrun.final_structure.to_json(),
                "structure_initial": vrun.initial_structure.to_json(),
                "run_stats": ocar.run_stats,
            }
        except Exception as e:
            self.error(f"Failed to parse SCF output: {e}")
            return null_res

    def process_data(self) -> bool:
        """Serializes results to Parquet files using Polars."""
        self.info("Serializing SCF results...")
        data = self.extract_data_dict()
        ts = pl.lit(self.config.date).dt.datetime()

        # Data Frames
        phys_data = {k: v for k, v in data.items() if k != "run_stats"}
        df_phys = pl.DataFrame(phys_data).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            ts.alias("TS"),
        )

        df_stats = pl.DataFrame(data["run_stats"] or {}).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            ts.alias("TS"),
        )

        # File I/O
        date_str = self.config.date.strftime("%Y%m%d_%H%M%S")
        f_suffix = f"_{date_str}_{self.config.material_id}.parquet"

        scf_dir = self.config.results_dir / "scf_data"
        stat_dir = self.config.results_dir / "run_stats"
        for d in [scf_dir, stat_dir]:
            d.mkdir(parents=True, exist_ok=True)

        df_phys.write_parquet(
            scf_dir / f"{self.config.step_prefix}_data{f_suffix}"
        )
        df_stats.write_parquet(
            stat_dir / f"{self.config.step_prefix}_run_stats{f_suffix}"
        )

        self.info(f"SCF data saved for {self.config.material_id}")
        return True

    def get_results(self) -> dict[str, Any]:
        """Loads and deserializes the final structure from disk."""
        date_str = self.config.date.strftime("%Y%m%d_%H%M%S")
        path = (
            self.config.results_dir
            / "scf_data"
            / f"{self.config.step_prefix}_data_{date_str}_{self.config.material_id}.parquet"
        )

        try:
            df = pl.read_parquet(path)
            s_final = df[0, "structure_final"]
            return {
                "is_succeed": df[0, "is_succeed"],
                "structure_final": Structure.from_dict(json.loads(s_final))
                if s_final
                else None,
            }
        except Exception as e:
            self.error(f"Could not load results from {path}: {e}")
            return {"is_succeed": False, "structure_final": None}
