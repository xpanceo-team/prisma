"""
mattergen-generate
"""

import tempfile
from pathlib import Path
from typing import Any
import zipfile

from pymatgen.core import Structure
from loguru import logger

from mattergen.common.globals import GENERATED_CRYSTALS_ZIP_FILE_NAME
from mattergen.common.utils.data_classes import (
    MatterGenCheckpointInfo,
    PRETRAINED_MODEL_NAME,
)
from mattergen.generator import CrystalGenerator


class OriginalMattergenPipeline:
    def __init__(self, model_name: PRETRAINED_MODEL_NAME):
        checkpoint_info = MatterGenCheckpointInfo.from_hf_hub(model_name)

        self.generator = CrystalGenerator(
            checkpoint_info=checkpoint_info,
            record_trajectories=False,
        )

    def generate(
        self,
        batch_size: int,
        num_batches: int,
        condition: dict[str, Any],
        guidance_scale: float = 0.0,
    ) -> list[Structure]:
        """
        Generates structures into a temporary folder as .cif files,
        extracts them from the generated zip file, converts them
        into pymatgen Structures, and returns the list of structures.
        """
        self.generator.properties_to_condition_on = condition
        self.generator.diffusion_guidance_factor = guidance_scale

        # Create a temporary directory for all the processing
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            # Generate .cif files in the temporary directory
            self.generator.generate(
                output_dir=str(temp_dir),
                batch_size=batch_size,
                num_batches=num_batches,
            )

            # Extract the generated crystals zip file
            extracted_dir = self._extract_generated_crystals_zip(temp_dir)

            # Gather all .cif files from the extracted directory and build structures
            cif_files = self._gather_cif_files(extracted_dir)
            structures = [
                self._build_pymatgen_structure(cif_path) for cif_path in cif_files
            ]

        return structures

    @staticmethod
    def _extract_generated_crystals_zip(
        path: Path,
        extract_dir_name: str = "extracted",
    ) -> Path:
        """
        Extracts the generated_crystals_cif.zip from the given directory
        into a subdirectory named 'extracted'.
        """
        zip_path = path / GENERATED_CRYSTALS_ZIP_FILE_NAME
        extraction_dir = path / extract_dir_name
        extraction_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)

        logger.info(f"Extracted {zip_path} to {extraction_dir}")

        return extraction_dir

    @staticmethod
    def _gather_cif_files(path: Path) -> list[Path]:
        """
        Gather and return all .cif files in the specified directory.
        """
        return list(path.glob("*.cif"))

    @staticmethod
    def _build_pymatgen_structure(cif_path: Path) -> Structure:
        """
        Build and return a pymatgen Structure object from the specified .cif file.
        """
        return Structure.from_file(cif_path)
