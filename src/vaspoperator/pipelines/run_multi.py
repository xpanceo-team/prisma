import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
from pymatgen.core import Structure

from vaspoperator.globals.logger import setup_logging
from vaspoperator.globals.yaml import load_yaml_dict
from vaspoperator.pipelines.run_single_structure import run_vasp_calculation

logger = logging.getLogger("BatchProcessor")


def process_structure(
    row: dict[str, Any],
    steps_config: dict,
    vasp_config: dict,
    server_config: dict,
    sumo_config: dict,
    VASP_DIR: Path,
    RESULTS_DIR: Path,
    id_column: str,
    structure_column: str,
) -> tuple[bool, str]:
    """
    Worker function to execute the full DFT pipeline for a single row.
    """
    material_id = row[id_column]
    try:
        # Reconstruct pymatgen structure from JSON string
        struct_data = row[structure_column]
        structure = Structure.from_dict(json.loads(struct_data))

        results = run_vasp_calculation(
            steps_config=steps_config,
            vasp_config=vasp_config,
            server_config=server_config,
            sumo_config=sumo_config,
            material_id=material_id,
            structure=structure,
            VASP_DIR=VASP_DIR,
            RESULTS_DIR=RESULTS_DIR,
        )
        return results["SCF"]["is_succeed"], material_id
    except Exception as e:
        logger.error(f"Failed pipeline for {material_id}: {str(e)}")
        return False, material_id


def main(
    dataset_path: str = "data/raw/test_structures.parquet",
    structure_column: str = "structure_mattersim",
    id_column: str = "material_id",
    num_threads: int = 5,
    limit: int = None,
):
    """
    Entry point for batch processing.

    Args:
        dataset_path: Path to Parquet file containing structures.
        structure_column: Name of the column with JSON-serialized structures.
        id_column: Name of the column for material identification.
        num_threads: Number of concurrent pipelines (-1 for max CPU).
        limit: Optional limit on the number of structures to process.
    """
    # Initialize infrastructure
    setup_logging()
    VASP_DIR = Path("data/vasp/")
    RESULTS_DIR = Path("data/results_batch/")

    # Load configs
    vasp_config = load_yaml_dict("config/vasp.yaml")
    server_config = load_yaml_dict("config/server.yaml")
    sumo_config = load_yaml_dict("config/sumo.yaml")
    steps_config = load_yaml_dict("config/steps.yaml")

    # Load and filter data efficiently
    df = pl.read_parquet(dataset_path).select([structure_column, id_column])
    df = df.filter(pl.col(structure_column).is_not_null())

    if limit:
        df = df.head(limit)

    structures_list = df.to_dicts()

    # Resource management
    if num_threads == -1:
        num_threads = os.cpu_count() or 1

    logger.info(
        f"🚀 Batch started. Processing {len(structures_list)} materials using {num_threads} threads."
    )

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_map = {
            executor.submit(
                process_structure,
                row,
                steps_config,
                vasp_config,
                server_config,
                sumo_config,
                VASP_DIR,
                RESULTS_DIR,
                id_column,
                structure_column,
            ): row[id_column]
            for row in structures_list
        }

        for future in as_completed(future_map):
            mid = future_map[future]
            try:
                success, _ = future.result()
                results.append((success, mid))
                status = "Success" if success else "Failed"
                logger.info(f"Finished {mid}: {status}")
            except Exception as e:
                logger.error(f"Critical future failure for {mid}: {e}")
                results.append((False, mid))

    # Summary reporting
    success_count = sum(1 for s, _ in results if s)
    logger.info(
        f"Batch complete. {success_count}/{len(results)} pipelines succeeded."
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
