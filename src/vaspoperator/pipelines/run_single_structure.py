import logging
from datetime import datetime
from pathlib import Path

from pymatgen.core import Structure

from vaspoperator.calculation.bands import StepBANDS, StepConfigBANDS
from vaspoperator.calculation.dos import StepConfigDOS, StepDOS
from vaspoperator.calculation.ipa import StepConfigIPA, StepIPA
from vaspoperator.calculation.rel import StepConfigREL, StepREL
from vaspoperator.calculation.scf import StepConfigSCF, StepSCF
from vaspoperator.globals.execution_order import get_execution_order
from vaspoperator.globals.logger import setup_logging
from vaspoperator.globals.yaml import load_yaml_dict

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = setup_logging()


def run_vasp_calculation(
    steps_config,
    vasp_config,
    server_config,
    sumo_config,
    material_id,
    structure,
    VASP_DIR,
    RESULTS_DIR,
):
    # Registry mapping types to their respective classes
    dict_steps = {
        "REL": {"config": StepConfigREL, "calc": StepREL},
        "SCF": {"config": StepConfigSCF, "calc": StepSCF},
        "IPA": {"config": StepConfigIPA, "calc": StepIPA},
        "DOS": {"config": StepConfigDOS, "calc": StepDOS},
        "BANDS": {"config": StepConfigBANDS, "calc": StepBANDS},
    }

    # Resolve DAG order
    steps_order = get_execution_order(
        steps_config["steps"], steps_config["dependencies"]
    )

    results = {}
    current_date = datetime.now()  # Fixed timestamp for the whole workflow run

    for step_id in steps_order:
        step_type = steps_config["type"][step_id]
        dependency = steps_config["dependencies"].get(step_id)

        # 1. Dependency Validation & Structure Propagation
        if dependency:
            dep_result = results.get(dependency)
            if not dep_result or not dep_result.get("is_succeed"):
                logger.error(
                    f"Step {step_id} aborted: Dependency {dependency} failed or was skipped."
                )
                break

            # Pass the relaxed/converged structure forward
            if dep_result.get("structure_final"):
                structure = dep_result["structure_final"]
                logger.info(
                    f"Step {step_id} using updated structure from {dependency}"
                )

        # 2. Configuration Setup
        logger.info(f"===> Starting Step: {step_id} (Type: {step_type})")

        config_cls = dict_steps[step_type]["config"]
        calc_cls = dict_steps[step_type]["calc"]

        config = config_cls(
            calculation_dir=VASP_DIR,
            results_dir=RESULTS_DIR,
            step_prefix=step_id,
            incar=vasp_config[step_id],
            server_config=server_config[step_id],
            sumo_config=sumo_config,
            material_id=material_id,
            kppa=vasp_config["common_params"]["kppa"],
            date=current_date,
        )

        # 3. Execution Lifecycle
        calculator = calc_cls(structure=structure, config=config)

        try:
            calculator.generate_input()

            # Uncomment for production
            calculator.submit_and_monitor()

            calculator.process_data()

            # 4. Result Retrieval
            step_res = calculator.get_results()
            results[step_id] = step_res

            if not step_res.get("is_succeed"):
                logger.warning(
                    f"Step {step_id} finished but marked as failed. Stopping pipeline."
                )
                break

        except Exception as e:
            logger.error(f"Critical error in step {step_id}: {str(e)}")
            results[step_id] = {"is_succeed": False, "structure_final": None}
            break

    return results


def main(
    structure_path: str = "data/raw/POSCAR.test",
    material_id: str = "test_structure",
):
    VASP_DIR = Path("data/vasp/")
    RESULTS_DIR = Path("data/results/")

    vasp_config = load_yaml_dict("config/vasp.yaml")
    server_config = load_yaml_dict("config/server.yaml")
    sumo_config = load_yaml_dict("config/sumo.yaml")
    steps_config = load_yaml_dict("config/steps.yaml")

    structure = Structure.from_file(structure_path)

    run_vasp_calculation(
        steps_config=steps_config,
        vasp_config=vasp_config,
        server_config=server_config,
        sumo_config=sumo_config,
        material_id=material_id,
        structure=structure,
        VASP_DIR=VASP_DIR,
        RESULTS_DIR=RESULTS_DIR,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
