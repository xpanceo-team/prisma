import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun

from vaspoperator.calculation.base import StepBase, StepConfigBase
from vaspoperator.extract.ipa import eps_to_principal_nk, plot_nk_vs_wavelength
from vaspoperator.globals.helpers import copy_file_between_stages_multi
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
class StepConfigIPA(StepConfigBase):
    """Configuration for Independent Particle Approximation (IPA) optics steps."""

    pass


@logged(name="IPA Step")
class StepIPA(StepBase):
    """
    Manages the VASP IPA calculation to obtain frequency-dependent optical properties.

    This step typically requires a pre-converged charge density (CHGCAR) from an
    SCF step. It extracts the dielectric tensor, converts it to complex refractive
    indices (n, k), and generates diagnostic plots.
    """

    def __init__(self, structure: Structure, config: StepConfigIPA):
        """
        Initializes the IPA step.

        Args:
            structure (Structure): Initial crystal structure.
            config (StepConfigIPA): Configuration parameters.
        """
        self.debug(f"Initializing IPA for material: {config.material_id}")
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
        """
        Generates VASP inputs and copies necessary restart files (CHGCAR/POSCAR)
        from the preceding SCF directory.
        """
        self.info(f"Generating IPA inputs in {self.step_folder}")
        self.step_folder.mkdir(parents=True, exist_ok=True)

        # Standard VASP suite
        create_and_save_incar(params=self.config.incar, folder=self.step_folder)
        create_and_save_poscar(
            structure=self.structure_initial, folder=self.step_folder
        )
        create_and_save_potcar(
            structure=self.structure_initial, folder=self.step_folder
        )

        # Execution scripts and metadata
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

        # Symmetry-based K-mesh optimization
        create_and_save_kgen(
            folder=self.step_folder,
            sumo_kgen_params=self.config.sumo_config["kgen"],
        )

        poscar_prim = self.step_folder / "POSCAR_prim"
        working_structure = self.structure_initial
        if poscar_prim.exists():
            self.debug("Using primitive cell for IPA KPOINTS.")
            working_structure = Structure.from_file(poscar_prim)
            create_and_save_poscar(
                structure=working_structure, folder=self.step_folder
            )

        create_and_save_kpoints(
            structure=working_structure,
            kppa=self.config.kppa,
            folder=self.step_folder,
        )

        # Copy CHGCAR and POSCAR from SCF for a non-self-consistent run
        scf_dir = self.structure_folder / "SCF"
        if scf_dir.exists():
            for filename in ["CHGCAR", "POSCAR"]:
                if (scf_dir / filename).exists():
                    self.debug(
                        f"Restart file found: Copying {filename} from SCF."
                    )
                    copy_file_between_stages_multi(
                        filename=filename,
                        folder=self.structure_folder,
                        step_initial="SCF",
                        steps_to_copy=[str(self.config.step_prefix)],
                    )

    def submit_and_monitor(self) -> None:
        """Submits the IPA job and monitors status via SLURM."""
        mon = self.config.server_config["monitor"]
        self.info(f"Submitting IPA job for {self.config.material_id}")
        submit_and_monitor(
            script_path=self.step_folder / "run.sh",
            timeout_seconds=mon["timeout_seconds"],
            check_interval=mon["check_interval"],
        )

    def extract_data_dict(self) -> dict[str, Any]:
        """Parses dielectric function data and standard results from VASP outputs."""
        self.debug("Extracting IPA results and dielectric tensor...")
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
            "dielectric_data": None,
            "run_stats": None,
        }

        try:
            vrun = Vasprun(self.step_folder / "vasprun.xml")
            ocar = Outcar(self.step_folder / "OUTCAR")

            if not vrun.converged:
                self.warning(
                    f"IPA calculation failed to converge for {self.config.material_id}"
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
                "dielectric_data": vrun.dielectric,
                "run_stats": ocar.run_stats,
            }
        except Exception as e:
            self.error(f"Error during IPA data extraction: {e}")
            return null_res

    def process_data(self) -> bool:
        """Transforms dielectric data into optical constants and saves to Parquet/PNG."""
        self.info("Processing IPA optical data...")
        data = self.extract_data_dict()
        ts = pl.lit(self.config.date).dt.datetime()

        # 1. Process standard physical data
        phys_data = {
            k: v
            for k, v in data.items()
            if k not in ["run_stats", "dielectric_data"]
        }
        df_phys = pl.DataFrame(phys_data).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            ts.alias("TS"),
        )

        # 2. Process SLURM stats
        df_stats = pl.DataFrame(data["run_stats"] or {}).with_columns(
            pl.lit(self.config.material_id).alias("material_id"),
            pl.lit(str(self.config.step_prefix)).alias("step"),
            ts.alias("TS"),
        )

        # 3. Process Optics (The Dielectric Tensor)
        df_optics = None
        raw_eps = data["dielectric_data"]
        if raw_eps is not None:
            # raw_eps[0] = energies, [1] = real part, [2] = imag part
            real, imag = np.array(raw_eps[1]).T, np.array(raw_eps[2]).T
            df_optics = pl.DataFrame(
                {
                    "Energies": raw_eps[0],
                    "real_e_xx": real[0],
                    "real_e_yy": real[1],
                    "real_e_zz": real[2],
                    "real_e_xy": real[3],
                    "real_e_yz": real[4],
                    "real_e_xz": real[5],
                    "imag_e_xx": imag[0],
                    "imag_e_yy": imag[1],
                    "imag_e_zz": imag[2],
                    "imag_e_xy": imag[3],
                    "imag_e_yz": imag[4],
                    "imag_e_xz": imag[5],
                }
            )

            # Convert dielectric tensor to principal n and k values
            df_optics = eps_to_principal_nk(df=df_optics).with_columns(
                pl.lit(self.config.material_id).alias("material_id"),
                ts.alias("TS"),
            )

        # 4. Persistence
        time_tag = self.config.date.strftime("%Y%m%d_%H%M%S")
        base_fn = f"_{time_tag}_{self.config.material_id}.parquet"

        # Ensure directories
        paths = {
            "ipa_data": self.config.results_dir / "ipa_data",
            "run_stats": self.config.results_dir / "run_stats",
            "ipa_dependency": self.config.results_dir / "ipa_dependency",
            "plots": self.config.results_dir / "ipa_optics_images",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        df_phys.write_parquet(
            paths["ipa_data"] / f"{self.config.step_prefix}_data{base_fn}"
        )
        df_stats.write_parquet(
            paths["run_stats"] / f"{self.config.step_prefix}_run_stats{base_fn}"
        )

        if df_optics is not None:
            df_optics.write_parquet(
                paths["ipa_dependency"]
                / f"{self.config.step_prefix}_optics{base_fn}"
            )

            # Generate visual diagnostic plot
            plot_nk_vs_wavelength(
                id=self.config.material_id,
                df=df_optics,
                save_dir=paths["plots"],
                show_plot=False,
                xlim=(300, 3000),
                filename_prefix=f"{time_tag}_{self.config.material_id}_optics",
            )

        self.info(f"IPA processing complete for {self.config.material_id}")
        return True

    def get_results(self) -> dict[str, Any]:
        """Retrieves and deserializes structural results from stored Parquet."""
        time_tag = self.config.date.strftime("%Y%m%d_%H%M%S")
        path = (
            self.config.results_dir
            / "ipa_data"
            / f"{self.config.step_prefix}_data_{time_tag}_{self.config.material_id}.parquet"
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
            self.error(f"Failed to retrieve IPA results: {e}")
            return {"is_succeed": False, "structure_final": None}

    @staticmethod
    def get_polars_schema() -> dict[str, pl.Schema]:
        """
        Returns the expected Polars schemas for IPA data, run statistics,
        and frequency-dependent optical dependencies.
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
            "dependency": pl.Schema(
                {
                    "material_id": pl.String,
                    "TS": pl.Datetime(time_unit="us"),
                    "Energies": pl.Float64,
                    "wavelength_nm": pl.Float64,
                    # Real part of dielectric tensor
                    "real_e_xx": pl.Float64,
                    "real_e_yy": pl.Float64,
                    "real_e_zz": pl.Float64,
                    "real_e_xy": pl.Float64,
                    "real_e_yz": pl.Float64,
                    "real_e_xz": pl.Float64,
                    # Imaginary part of dielectric tensor
                    "imag_e_xx": pl.Float64,
                    "imag_e_yy": pl.Float64,
                    "imag_e_zz": pl.Float64,
                    "imag_e_xy": pl.Float64,
                    "imag_e_yz": pl.Float64,
                    "imag_e_xz": pl.Float64,
                    # Refractive index (n) and Extinction coefficient (k)
                    "n_xx": pl.Float64,
                    "n_yy": pl.Float64,
                    "n_zz": pl.Float64,
                    "k_xx": pl.Float64,
                    "k_yy": pl.Float64,
                    "k_zz": pl.Float64,
                }
            ),
        }
