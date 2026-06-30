from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import combinations

from tqdm.auto import tqdm
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core import Composition, Structure

from crystal_diffusers.utils.logging import logger
from crystal_diffusers.validation.sun import _require_columns

_ds_rows_by_system = None
_ref_entries_by_elementset = None


def _all_nonempty_subsets(elementset: frozenset[str]) -> list[frozenset[str]]:
    elements = sorted(elementset)
    out = []
    for r in range(1, len(elements)):
        for c in combinations(elements, r):
            out.append(frozenset(c))
    return out


def _ehull_worker(
    systems_chunk: list[str],
    energy_key: str,
    build_hull_from_reference_only: bool,
    return_phase_diagrams: bool,
) -> dict[int, float] | tuple[dict[int, float], dict[str, PhaseDiagram]]:
    ds_rows_by_system = _ds_rows_by_system
    ref_entries_by_elementset = _ref_entries_by_elementset

    out = {}
    phase_diagrams_by_system = {}

    for system in systems_chunk:
        system_rows = ds_rows_by_system[system]

        target_elementset = frozenset(system.split("-"))

        missing_terminals = [
            el for el in target_elementset
            if not ref_entries_by_elementset.get(frozenset([el]))
        ]
        if missing_terminals:
            logger.warning(
                f"Skipping system {system} due to missing terminal "
                f"entries for {sorted(missing_terminals)}."
            )
            continue

        target_subsets = _all_nonempty_subsets(target_elementset)

        ref_entries = []
        ref_entries_with_all_elements = ref_entries_by_elementset.get(target_elementset, [])
        ref_entries.extend(ref_entries_with_all_elements)

        for subset in target_subsets:
            entries = ref_entries_by_elementset.get(subset, [])
            ref_entries.extend(entries)

        if build_hull_from_reference_only:
            hull_entries = ref_entries
        else:
            hull_entries = ref_entries + [
                PDEntry(Composition(row["formula"]), row[energy_key])
                for row in system_rows
            ]

        # Need at least 2 total entries for a phase diagram to exist
        if len(hull_entries) < 2:
            logger.warning(f"Skipping system {system} as it doesn't have enough entries for builing convex hull.")
            continue

        try:
            pd = PhaseDiagram(hull_entries)
        except Exception as e:
            print(f"Skipping system {system} due to error: {e}")
            continue

        for row in system_rows:
            entry = PDEntry(
                composition=Composition(row["formula"]),
                energy=row[energy_key],
            )
            e_hull = pd.get_e_above_hull(entry, allow_negative=True)
            out[row["idx"]] = float(e_hull)

        if return_phase_diagrams:
            phase_diagrams_by_system[system] = pd

    if return_phase_diagrams:
        return out, phase_diagrams_by_system
    else:
        return out


def add_system_info(row, structure_key: str = "structure"):
    s = Structure.from_str(row[structure_key], fmt="json")

    return {
        "chemical_system": s.chemical_system,
        "unique_elements": s.chemical_system.split("-"),
        "formula": s.formula,
    }


def _init_ehull_worker(ds_rows_by_system, ref_entries_by_elementset):
    global _ds_rows_by_system, _ref_entries_by_elementset
    _ds_rows_by_system = ds_rows_by_system
    _ref_entries_by_elementset = ref_entries_by_elementset


def _iter_chunks(items: list[str], chunk_size: int):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def compute_dataset_ehull(
    ds,
    ref_ds,
    energy_key="relaxed_energy",
    structure_key: str = "structure",
    num_proc=16,
    filter_system_size: int| None = None,
    build_hull_from_reference_only: bool = True,
    chunk_size_systems: int = 500,
    return_phase_diagrams: bool = False,
):
    if not build_hull_from_reference_only:
        raise ValueError(
            "build_hull_from_reference_only=False currently is not supported"
        )
    start_method = multiprocessing.get_start_method(allow_none=True)
    if start_method != "fork":
        logger.warning(
            "compute_dataset_ehull is fastest with multiprocessing start method 'fork'."
            f" Current start method: {start_method!r}. This may be slower due to extra "
            "serialization/startup cost."
        )

    _require_columns(ds, {energy_key, structure_key}, name="Dataset")
    _require_columns(ref_ds, {energy_key, structure_key}, name="Reference dataset")

    if "idx" in ds.column_names:
        ds = ds.remove_columns("idx")
    ds = ds.add_column("idx", list(range(len(ds))))

    if "energy_above_hull" not in ds.column_names:
        ds = ds.add_column("energy_above_hull", [None] * len(ds))

    # Work only on rows that have both the structure and energy needed for hull computation.
    valid_ds = ds.filter(
        lambda r: r[structure_key] is not None and r[energy_key] is not None,
        num_proc=num_proc,
    )
    valid_ref_ds = ref_ds.filter(
        lambda r: r[structure_key] is not None and r[energy_key] is not None,
        num_proc=num_proc,
    )

    valid_ds = valid_ds.map(
        partial(add_system_info, structure_key=structure_key),
        num_proc=num_proc,
    )
    valid_ref_ds = valid_ref_ds.map(
        partial(add_system_info, structure_key=structure_key),
        num_proc=num_proc,
    )

    # FIXME: filter with energy_above_hull is none is confusing
    systems = list(
        set(
            valid_ds.filter(lambda x: x["energy_above_hull"] is None, num_proc=num_proc)[
                "chemical_system"
            ]
        )
    )
    # TODO: remove magic variable
    no_terminal_entries = [
        "Ac", "He", "U", "Ar", "Pm", "Kr",
        "Th", "Tc", "Np", "At", "Pa", "Pu",
    ]

    # filter huge element number systems and systems without terminal entries
    if filter_system_size:
        systems = [
            s for s in systems
            if (len(s.split("-")) <= filter_system_size)
            and not any(el in no_terminal_entries for el in s.split("-"))
        ]

    # Build generated rows grouped by exact system (minimal payload)
    ds_rows_by_system: dict[str, list[dict]] = {}
    for r in tqdm(
        valid_ds.select_columns(["idx", "chemical_system", "formula", energy_key]),
        desc="grouping dataset by system",
    ):
        sys = r["chemical_system"]
        ds_rows_by_system.setdefault(sys, []).append(
            {
                "idx": int(r["idx"]),
                "formula": r["formula"],
                energy_key: r[energy_key],
            }
        )

    # Build reference entries grouped by element-set (PDEntry objects built once)
    ref_entries_by_elementset: dict[frozenset[str], list[PDEntry]] = {}
    for r in tqdm(
        valid_ref_ds.select_columns(["unique_elements", "formula", energy_key]),
        desc="grouping reference dataset by system",
    ):
        ref_element_set = frozenset(r["unique_elements"])
        ref_entries_by_elementset.setdefault(ref_element_set, []).append(
            PDEntry(Composition(r["formula"]), r[energy_key])
        )

    # Parallel compute in chunks to reduce executor overhead
    idx_to_ehull: dict[int, float] = {}
    systems_chunks = list(_iter_chunks(systems, chunk_size_systems))

    phase_diagrams_by_system: dict[str, PhaseDiagram] = {}

    worker = partial(
        _ehull_worker,
        energy_key=energy_key,
        build_hull_from_reference_only=build_hull_from_reference_only,
        return_phase_diagrams=return_phase_diagrams,
    )
    initargs = (ds_rows_by_system, ref_entries_by_elementset)

    with ProcessPoolExecutor(
        max_workers=min(num_proc, len(systems_chunks) or 1),
        initializer=_init_ehull_worker,
        initargs=initargs,
    ) as pool:
        futures = [pool.submit(worker, ch) for ch in systems_chunks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Phase diagrams",
            unit="system chunk",
        ):
            result = future.result()

            if return_phase_diagrams:
                idx_to_ehull_upd, phase_diagrams = result
                for sys_name, pd in phase_diagrams.items():
                    phase_diagrams_by_system[sys_name] = pd
            else:
                idx_to_ehull_upd = result

            idx_to_ehull.update(idx_to_ehull_upd)

    # Merge into dataset column
    ehull_col = list(ds["energy_above_hull"])

    for idx, val in idx_to_ehull.items():
        ehull_col[idx] = val

    if "energy_above_hull" in ds.column_names:
        ds = ds.remove_columns("energy_above_hull")
    ds = ds.add_column("energy_above_hull", ehull_col)

    if return_phase_diagrams:
        return ds, phase_diagrams_by_system
    else:
        return ds
