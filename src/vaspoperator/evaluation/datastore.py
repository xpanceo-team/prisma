import datetime
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from vaspoperator.calculation.bands import StepBANDS
from vaspoperator.calculation.dos import StepDOS
from vaspoperator.calculation.ipa import StepIPA
from vaspoperator.calculation.rel import StepREL
from vaspoperator.calculation.scf import StepSCF
from vaspoperator.extract.ipa_interpolate import interpolate_ipa_to_wavelength
from vaspoperator.globals.logger import logged

logger = logging.getLogger("Datastore")


@dataclass(frozen=True)
class StructureInfo:
    """Container for lazily evaluated structure data.

    Attributes:
        df_data: Main VASP calculation results (energies, forces, etc.).
        df_run_stats: Performance metrics and SLURM execution metadata.
        df_ipa_dependency: Frequency-dependent dielectric function data.
        df_dos_dependency: Electronic density of states data.
        df_bands_dependency: Electronic band structure eigenvalues.
        df_ipa_wl: IPA data interpolated to a specific target wavelength.
    """

    df_data: pl.LazyFrame
    df_run_stats: pl.LazyFrame
    df_ipa_dependency: pl.LazyFrame
    df_dos_dependency: pl.LazyFrame
    df_bands_dependency: pl.LazyFrame
    df_ipa_wl: pl.LazyFrame


@dataclass
class DatastoreConfig:
    """Configuration for data paths and processing parameters.

    Attributes:
        target_wl: The wavelength (nm) for optical property interpolation.
        wavelength_col: Name of the column containing wavelength data.
        run_stats_path: Directory containing run performance logs.
        rel_data_path: Directory containing Relaxation results.
        scf_data_path: Directory containing SCF results.
        ipa_data_path: Directory containing IPA results.
        dos_data_path: Directory containing DOS results.
        bands_data_path: Directory containing BANDS results.
        ipa_dependency: Directory for raw IPA output files.
        dos_dependency: Directory for raw DOS output files.
        bands_dependency: Directory for raw BANDS output files.
        output_base: Root directory for the final aggregated dataset.
    """

    target_wl: float = 1064.0
    wavelength_col: str = "wavelength_nm"

    # Input Paths
    run_stats_path: Path = Path("data/results/run_stats/")
    rel_data_path: Path = Path("data/results/rel_data/")
    scf_data_path: Path = Path("data/results/scf_data/")
    ipa_data_path: Path = Path("data/results/ipa_data/")
    dos_data_path: Path = Path("data/results/dos_data/")
    bands_data_path: Path = Path("data/results/bands_data/")

    ipa_dependency: Path = Path("data/results/ipa_dependency/")
    dos_dependency: Path = Path("data/results/dos_dependency/")
    bands_dependency: Path = Path("data/results/bands_dependency/")

    # Output Paths
    output_base: Path = Path("data/final_dataset/")

    def __post_init__(self):
        """Initializes output subdirectories based on the output_base."""
        self.data_out = self.output_base / "structure_data"
        self.stats_out = self.output_base / "run_stats"
        self.ipa_dep_out = self.output_base / "ipa_dependency"
        self.ipa_wl_out = self.output_base / "ipa_dependency_wl"
        self.dos_dep_out = self.output_base / "dos_dependency"
        self.bands_dep_out = self.output_base / "bands_dependency"


@logged(name="Datastore")
class StructureDatastore:
    """Orchestrates high-throughput access to VASP calculation results.

    This class provides a unified interface to query, slice, and export
    DFT data stored across various parquet files.

    Args:
        config: A DatastoreConfig instance containing file paths and settings.
    """

    def __init__(self, config: DatastoreConfig):
        self.config = config
        self._init_catalog()

    def _init_catalog(self):
        """Initializes a light-weight catalog of all available materials and steps.

        Scans the provided directories to build an in-memory index of available
        material_ids and calculation steps without loading the full data.
        """
        search_paths = [
            (self.config.rel_data_path, StepREL),
            (self.config.scf_data_path, StepSCF),
            (self.config.ipa_data_path, StepIPA),
            (self.config.dos_data_path, StepDOS),
            (self.config.bands_data_path, StepBANDS),
        ]

        scans = []
        for path, step_cls in search_paths:
            if path.exists():
                scans.append(
                    pl.scan_parquet(
                        path / "*.parquet",
                        schema=step_cls.get_polars_schema()["data"],
                        missing_columns="insert",
                    ).select(["material_id", "TS", "step"])
                )

        if not scans:
            logger.warning("No parquet files found in provided paths.")
            self.df_catalog = pl.DataFrame(
                schema={"material_id": pl.Utf8, "TS": pl.Int64, "step": pl.Utf8}
            )
        else:
            self.df_catalog = (
                pl.concat(scans)
                .unique()
                .with_columns(
                    pl.col("TS").fill_null(
                        datetime.datetime(2026, 1, 1, 0, 0, 0)
                    )
                )
                .collect()
            )

        self.ids = (
            self.df_catalog.select("material_id").unique().sort("material_id")
        )
        self.steps = self.df_catalog.select("step").unique()

    def get_structure_info(
        self, material_id: str, step: str = "all"
    ) -> StructureInfo | None:
        """Retrieves LazyFrames for a specific material_id."""
        return self._get_cached_structure_info(material_id, step)

    def _get_cached_structure_info(
        self, material_id: str, step: str
    ) -> StructureInfo | None:
        """Retrieves LazyFrames for a specific material_id.

        Args:
            material_id: The unique identifier for the material.
            step: The specific calculation step (e.g., 'SCF', 'IPA').
                Defaults to 'all'.

        Returns:
            A StructureInfo object containing LazyFrames for the requested data,
            or None if the material_id is not found.
        """
        data_paths = [
            (self.config.rel_data_path, StepREL),
            (self.config.scf_data_path, StepSCF),
            (self.config.ipa_data_path, StepIPA),
            (self.config.dos_data_path, StepDOS),
            (self.config.bands_data_path, StepBANDS),
        ]

        df_data = pl.concat(
            [
                pl.scan_parquet(
                    p / f"*{material_id}*.parquet",
                    schema=s.get_polars_schema()["data"],
                    missing_columns="insert",
                )
                for p, s in data_paths
                if p.exists()
            ]
        ).with_columns(
            pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
        )

        df_ipa_dependency = pl.scan_parquet(
            self.config.ipa_dependency / f"*{material_id}*.parquet",
            schema=StepIPA.get_polars_schema()["dependency"],
            missing_columns="insert",
        ).with_columns(
            pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
        )

        df_ipa_wl = interpolate_ipa_to_wavelength(
            df_ipa_dependency=df_ipa_dependency, target_wl=self.config.target_wl
        )

        df_dos_dependency = pl.scan_parquet(
            self.config.dos_dependency / f"*{material_id}*.parquet",
            schema=StepDOS.get_polars_schema()["dependency"],
        ).with_columns(
            pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
        )

        df_bands_dependency = pl.scan_parquet(
            self.config.bands_dependency / f"*{material_id}*.parquet",
            schema=StepBANDS.get_polars_schema()["dependency"],
        ).with_columns(
            pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
        )

        df_run_stats = pl.scan_parquet(
            self.config.run_stats_path / f"*{material_id}*.parquet",
            schema=StepBANDS.get_polars_schema()["run_stats"],
        ).with_columns(
            pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
        )

        if step != "all":
            df_data = df_data.filter(pl.col("step") == step)
            df_run_stats = df_run_stats.filter(pl.col("step") == step)

        return StructureInfo(
            df_data=df_data,
            df_run_stats=df_run_stats,
            df_ipa_dependency=df_ipa_dependency,
            df_dos_dependency=df_dos_dependency,
            df_bands_dependency=df_bands_dependency,
            df_ipa_wl=df_ipa_wl,
        )

    def get_all_structures(self) -> StructureInfo:
        """Scan all available data across the entire catalog.

        Returns:
            A StructureInfo object encompassing all data in the datastore.
        """
        return StructureInfo(
            df_data=self._scan_all_main_data().with_columns(
                pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
            ),
            df_run_stats=pl.scan_parquet(
                self.config.run_stats_path / "*.parquet",
                schema=StepREL.get_polars_schema()["run_stats"],
                missing_columns="insert",
            ).with_columns(
                pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
            ),
            df_ipa_dependency=pl.scan_parquet(
                self.config.ipa_dependency / "*.parquet",
                schema=StepIPA.get_polars_schema()["dependency"],
                missing_columns="insert",
            ).with_columns(
                pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
            ),
            df_dos_dependency=pl.scan_parquet(
                self.config.dos_dependency / "*.parquet",
                schema=StepDOS.get_polars_schema()["dependency"],
                missing_columns="insert",
            ).with_columns(
                pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
            ),
            df_bands_dependency=pl.scan_parquet(
                self.config.bands_dependency / "*.parquet",
                schema=StepBANDS.get_polars_schema()["dependency"],
                missing_columns="insert",
            ).with_columns(
                pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
            ),
            df_ipa_wl=interpolate_ipa_to_wavelength(
                pl.scan_parquet(
                    self.config.ipa_dependency / "*.parquet",
                    schema=StepIPA.get_polars_schema()["dependency"],
                    missing_columns="insert",
                ),
                target_wl=self.config.target_wl,
            ).with_columns(
                pl.col("TS").fill_null(datetime.datetime(2026, 1, 1, 0, 0, 0))
            ),
        )

    def _scan_all_main_data(self) -> pl.LazyFrame:
        """Concatenates all primary VASP results into a single LazyFrame."""
        paths = [
            (self.config.rel_data_path, StepREL),
            (self.config.scf_data_path, StepSCF),
            (self.config.ipa_data_path, StepIPA),
            (self.config.dos_data_path, StepDOS),
            (self.config.bands_data_path, StepBANDS),
        ]
        return pl.concat(
            [
                pl.scan_parquet(
                    p / "*.parquet",
                    schema=s.get_polars_schema()["data"],
                    missing_columns="insert",
                )
                for p, s in paths
                if p.exists()
            ]
        )

    def save_dataset_parquet(self):
        """Sinks the entire dataset to the output directory using PartitionBy."""
        ds = self.get_all_structures()

        mapping = [
            (ds.df_data, self.config.data_out),
            (ds.df_run_stats, self.config.stats_out),
            (ds.df_ipa_dependency, self.config.ipa_dep_out),
            (ds.df_ipa_wl, self.config.ipa_wl_out),
            (ds.df_dos_dependency, self.config.dos_dep_out),
            (ds.df_bands_dependency, self.config.bands_dep_out),
        ]

        for lf, out_path in mapping:
            logger.info(f"Sinking dataset to {out_path}...")
            lf.sink_parquet(
                pl.PartitionBy(out_path, max_rows_per_file=5_000_000),
                mkdir=True,
            )

    def __len__(self) -> int:
        """Returns the number of unique material_ids in the catalog."""
        return self.ids.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the dimensions as (number of materials, number of steps)."""
        return (self.ids.shape[0], self.steps.shape[0])

    def __getitem__(
        self, idx: str | int | slice | tuple[str | int | slice, str]
    ) -> StructureInfo | None | list[StructureInfo]:
        """Access structure data using ID, index, or slices.

        Example:
            ds["Ag2O"] -> Single structure by ID
            ds[0:5] -> List of first 5 structures
            ds[5, "SCF"] -> SCF data for the 6th material

        Args:
            idx: The key to access. Can be a material_id string, an integer
                index, a slice object, or a tuple of (key, step).

        Returns:
            A StructureInfo object (or list of them if sliced).

        Raises:
            IndexError: If the integer index is out of bounds.
        """
        if isinstance(idx, tuple):
            key, step = idx
        else:
            key, step = idx, "all"

        # Handle slicing
        if isinstance(key, slice):
            target_ids = self.ids[key, "material_id"].to_list()
            return [self.get_structure_info(m_id, step) for m_id in target_ids]

        # Handle integer indexing
        if isinstance(key, int):
            if key < 0 or key >= len(self.ids):
                raise IndexError("Material index out of range")
            key = self.ids[key, "material_id"]

        return self.get_structure_info(material_id=key, step=step)

    def __iter__(self) -> Iterator[StructureInfo]:
        """Iterates through all materials in the catalog."""
        for mat_id in self.ids["material_id"]:
            yield self.get_structure_info(material_id=mat_id, step="all")
