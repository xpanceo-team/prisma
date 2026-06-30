import logging
from typing import Any

# Standard logging setup
logger = logging.getLogger(__name__)


def get_execution_order(
    selected_steps: list[str], dependencies: dict[str, Any]
) -> list[str]:
    """
    Determines the linear execution order for VASP workflow steps.

    Uses a Depth First Search (DFS) for topological sorting. It handles
    cases where dependencies might be None, a single string, or a list.

    Args:
        selected_steps (list[str]): Steps requested by the user.
        dependencies (dict[str, Any]): Mapping of steps to prerequisites.

    Returns:
        list[str]: A list of steps ordered by their execution priority.

    Raises:
        ValueError: If a circular dependency is detected.
    """
    logger.info(
        f"Resolving execution order for selected steps: {selected_steps}"
    )

    ordered: list[str] = []
    visited: set[str] = set()
    temp_stack: set[str] = set()

    def resolve(step: str):
        # Detect infinite loops (A -> B -> A)
        if step in temp_stack:
            logger.error(
                f"Circular dependency detected involving step: '{step}'"
            )
            raise ValueError(f"Circular dependency detected involving: {step}")

        if step in visited:
            return

        logger.debug(f"Resolving prerequisites for: '{step}'")
        temp_stack.add(step)

        # Normalize dependencies: handle None, strings, or lists
        raw_deps = dependencies.get(step, [])
        if raw_deps is None:
            prerequisites = []
        elif isinstance(raw_deps, str):
            prerequisites = [raw_deps]
        else:
            prerequisites = raw_deps

        # Recursively visit all prerequisites
        for prereq in prerequisites:
            logger.debug(f"  - '{step}' depends on '{prereq}'")
            resolve(prereq)

        temp_stack.remove(step)
        visited.add(step)
        ordered.append(step)
        logger.info(f"Step '{step}' added to execution queue.")

    # Process each requested step
    for step in selected_steps:
        if step not in dependencies and step not in visited:
            logger.warning(
                f"Step '{step}' requested but not found in dependency map."
            )
        resolve(step)

    logger.info(f"Final execution sequence: {' -> '.join(ordered)}")
    return ordered
