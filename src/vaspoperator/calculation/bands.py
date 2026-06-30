import json
import shutil
from dataclasses import dataclass
from typing import Any

import polars as pl
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun
from sumo.cli.bandplot import bandplot

from vaspoperator.calculation.base import StepBase, StepConfigBase
from vaspoperator.globals.helpers import (
    clear_from_dat,
    copy_file_between_stages_multi,
)
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
class StepConfigBANDS(StepConfigBase):
    """Configuration for Electronic Band Structure calculations."""

    pass


@logged(name="BANDS Step")
class StepBANDS(StepBase):
    """
    Manages VASP Band Structure calculations.

    This step requires a pre-converged CHGCAR (from SCF) and a line-mode
    KPOINTS file to resolve the dispersion along high-symmetry paths.
    """

    def __init__(self, structure: Structure, config: StepConfigBANDS):
        self.debug(f"Initializing BANDS step for {config.material_id}")
        self.structure_initial = structure
        self.config = config
        self.step_folder = (
            self.config.calculation_dir
            / self.config.material_id
            / str(self.config.step_prefix)
        )
        self.structure_folder = (
            self.config.calculation_dir / self.config.material_id
        )

    def generate_input(self) -> None:
        """Generates VASP inputs, specifically handling the high-symmetry K-path."""
        self.info(f"Generating BANDS inputs in {self.step_folder}")
        self.step_folder.mkdir(parents=True, exist_ok=True)

        # Standard inputs
        create_and_save_incar(params=self.config.incar, folder=self.step_folder)
        create_and_save_poscar(
            structure=self.structure_initial, folder=self.step_folder
        )
        create_and_save_potcar(
            structure=self.structure_initial, folder=self.step_folder
        )

        v_conf = self.config.server_config["vasp"]
        create_and_save_run_script(
            id=self.config.material_id,
            n_cpus=v_conf["n_cpus"],
            n_nodes=v_conf["n_nodes"],
            folder=self.step_folder,
            step=str(self.config.step_prefix),
            max_duration=v_conf["max_duration"],
            cluster_part=v_conf["cluster_part"],
            unavailable_nodes=v_conf["unavailable_nodes"],
            is_exclusive=v_conf["is_exclusive"],
        )
        create_and_save_readme(
            structure=self.structure_initial,
            folder=self.step_folder,
            kppa=self.config.kppa,
            id=self.config.material_id,
            step=str(self.config.step_prefix),
        )

        # 1. Standard KPOINTS generation (fallback)
        create_and_save_kgen(
            folder=self.step_folder,
            sumo_kgen_params=self.config.sumo_config["kgen"],
        )

        work_struct = self.structure_initial
        if (self.step_folder / "POSCAR_prim").exists():
            work_struct = Structure.from_file(self.step_folder / "POSCAR_prim")
            create_and_save_poscar(
                structure=work_struct, folder=self.step_folder
            )

        create_and_save_kpoints(
            structure=work_struct,
            kppa=self.config.kppa,
            folder=self.step_folder,
        )

        # 2. Critical Stage: Overwrite with restart files from SCF
        scf_dir = self.structure_folder / "SCF"
        if scf_dir.exists():
            # Copy CHGCAR and POSCAR for non-self-consistent run
            for f in ["CHGCAR", "POSCAR"]:
                if (scf_dir / f).exists():
                    copy_file_between_stages_multi(
                        filename=f,
                        folder=self.structure_folder,
                        step_initial="SCF",
                        steps_to_copy=[str(self.config.step_prefix)],
                    )

            # Retrieve the specialized line-mode KPOINTS
            if (scf_dir / "KPOINTS_band").exists():
                self.debug(
                    "Found KPOINTS_band in SCF. Overwriting default KPOINTS."
                )
                copy_file_between_stages_multi(
                    filename="KPOINTS_band",
                    folder=self.structure_folder,
                    step_initial="SCF",
                    steps_to_copy=[str(self.config.step_prefix)],
                )
                shutil.move(
                    self.step_folder / "KPOINTS_band",
                    self.step_folder / "KPOINTS",
                )

    def submit_and_monitor(self) -> None:
        mon = self.config.server_config["monitor"]
        submit_and_monitor(
            script_path=self.step_folder / "run.sh",
            timeout_seconds=mon["timeout_seconds"],
            check_interval=mon["check_interval"],
        )

    def extract_data_dict(self) -> dict[str, Any]:
        """Extracts band eigenvalues and metadata."""
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
            "bands_1": None,
            "bands_2": None,
            "run_stats": None,
        }

        try:
            vrun = Vasprun(self.step_folder / "vasprun.xml")
            ocar = Outcar(self.step_folder / "OUTCAR")

            if not vrun.converged:
                return {**null_res, "run_stats": ocar.run_stats}

            # Bands are dictionaries of {Spin: eigenvalues_array}
            band_dict = vrun.get_band_structure().bands
            spin_keys = list(band_dict.keys())

            b1 = band_dict[
                spin_keys[0]
            ].tolist()  # Convert numpy to list for Polars
            b2 = (
                band_dict[spin_keys[1]].tolist() if len(band_dict) > 1 else None
            )

            bs_smart = vrun.get_band_structure(efermi="smart")
            bg = bs_smart.get_band_gap()

            return {
                "is_succeed": True,
                "energy": vrun.final_energy,
                "is_spin": vrun.is_spin,
                "e_fermi": ocar.efermi,
                "incar": vrun.incar.to_json(),
                "total_mag": ocar.total_mag,
                "bandgap": bg["energy"],
                "bandgap_direct": float(bs_smart.get_direct_band_gap()),
                "bandgap_cbm": bs_smart.get_cbm()["energy"],
                "bandgap_vbm": bs_smart.get_vbm()["energy"],
                "is_gap_direct": bg["direct"],
                "structure_final": vrun.final_structure.to_json(),
                "structure_initial": vrun.initial_structure.to_json(),
                "bands_1": b1,
                "bands_2": b2,
                "run_stats": ocar.run_stats,
            }
        except Exception as e:
            self.error(f"Band structure extraction failed: {e}")
            return null_res

    def process_data(self) -> bool:
        """Serializes band data to Parquet and generates plots."""
        data = self.extract_data_dict()
        ts = pl.lit(self.config.date).dt.datetime()

        # 1. Main Table
        df_phys = pl.DataFrame(
            {
                k: v
                for k, v in data.items()
                if k not in ["run_stats", "bands_1", "bands_2"]
            }
        ).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            ts.alias("TS"),
        )

        # 2. Runtime Statistics
        df_stats = pl.DataFrame(data["run_stats"] or {}).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            ts.alias("TS"),
        )

        # 3. Band Eigenvalues (Dependency)
        df_bands = pl.DataFrame(
            {
                "bands_1": [data["bands_1"]],
                "bands_2": [data["bands_2"]],
            }
        ).with_columns(
            pl.lit(self.config.material_id).alias("material_id"), ts.alias("TS")
        )

        # Save to Parquet
        dirs = ["bands_data", "run_stats", "bands_dependency", "bands_images"]
        for d in dirs:
            (self.config.results_dir / d).mkdir(parents=True, exist_ok=True)

        time_tag = self.config.date.strftime("%Y%m%d_%H%M%S")
        df_phys.write_parquet(
            self.config.results_dir
            / "bands_data"
            / f"{self.config.step_prefix}_data_{time_tag}_{self.config.material_id}.parquet"
        )
        df_stats.write_parquet(
            self.config.results_dir
            / "run_stats"
            / f"{self.config.step_prefix}_run_stats_{time_tag}_{self.config.material_id}.parquet"
        )
        df_bands.write_parquet(
            self.config.results_dir
            / "bands_dependency"
            / f"{self.config.step_prefix}_bands_{time_tag}_{self.config.material_id}.parquet"
        )

        # Visual Plotting via Sumo
        vxml = self.step_folder / "vasprun.xml"
        if vxml.exists():
            bandplot(
                filenames=str(vxml),
                image_format="png",
                directory=str(self.config.results_dir / "bands_images"),
                title=self.config.material_id,
                prefix=f"{time_tag}_{self.config.material_id}",
                ymin=-10,
                ymax=10,
            )
            clear_from_dat(self.config.results_dir / "bands_images")

        return True

    def get_results(self) -> dict[str, Any]:
        time_tag = self.config.date.strftime("%Y%m%d_%H%M%S")
        df = pl.read_parquet(
            self.config.results_dir
            / "bands_data"
            / f"{self.config.step_prefix}_data_{time_tag}_{self.config.material_id}.parquet"
        )
        s_final = df[0, "structure_final"]
        return {
            "is_succeed": df[0, "is_succeed"],
            "structure_final": Structure.from_dict(json.loads(s_final))
            if s_final
            else None,
        }

    @staticmethod
    def get_polars_schema() -> dict[str, pl.Schema]:
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
            "dependency": pl.Schema(
                {
                    "bands_1": pl.List(pl.Float64),
                    "bands_2": pl.List(pl.Float64),
                    "material_id": pl.String,
                    "TS": pl.Datetime(time_unit="us"),
                }
            ),
        }
