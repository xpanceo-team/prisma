from functools import partial

import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
from tqdm.auto import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from datasets import Sequence, Value, Dataset, List

from prisma.utils.logging import logger

_NOVELTY_SYS_GROUPS: dict[str, list["NoveltySysItem"]] | None = None
_NOVELTY_REF_GROUPS: dict[str, list["NoveltyRefItem"]] | None = None
_UNIQUENESS_GROUP_ITEMS: dict[str, list["UniqueItem"]] | None = None
_BIG_UNIQUENESS_GROUP_ITEMS: dict[str, list["UniqueItem"]] | None = None
_STRUCTURE_PARSE_MISS = object()


@dataclass(frozen=True, slots=True)
class NoveltySysItem:
    """A single system-side structure entry used for novelty checks."""
    idx: int
    structure_json: str | None


@dataclass(frozen=True, slots=True)
class NoveltyRefItem:
    """A single reference-side structure entry used for novelty checks."""
    material_id: str
    structure_json: str | None


@dataclass(frozen=True, slots=True)
class UniqueItem:
    """A single entry used for uniqueness checks within a formula group."""
    idx: int
    structure_json: str | None
    energy: float | None


def _iter_chunks(items: list[str], chunk_size: int) -> Iterator[list[str]]:
    """Yield consecutive chunks from a list."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _iter_chunks_by_cost(
    sys_groups: dict[str, list[NoveltySysItem]],
    ref_groups: dict[str, list[NoveltyRefItem]],
    target_cost: int = 200_000,
) -> Iterator[list[str]]:
    """Chunk systems to balance work: cost ~ |sys| * |ref| comparisons."""
    def est_cost(system: str) -> int:
        return len(sys_groups.get(system, [])) * len(ref_groups.get(system, []))

    systems_to_match = [s for s in sys_groups.keys() if s in ref_groups]

    if not systems_to_match:
        return

    systems_to_match.sort(key=est_cost, reverse=True)

    cur_chunk: list[str] = []
    cur_cost = 0

    for system in systems_to_match:
        c = est_cost(system)

        # If a single system is huge, isolate it.
        if c >= target_cost:
            if cur_chunk:
                yield cur_chunk
                cur_chunk = []
                cur_cost = 0
            yield [system]
            continue

        if cur_chunk and (cur_cost + c) > target_cost:
            yield cur_chunk
            cur_chunk = []
            cur_cost = 0

        cur_chunk.append(system)
        cur_cost += c

    if cur_chunk:
        yield cur_chunk


def _ensure_sequential_idx(ds: Dataset) -> Dataset:
    """Ensure an 'idx' column matching row positions [0..len(ds)-1]."""
    n = len(ds)
    if "idx" in ds.column_names:
        ds = ds.remove_columns("idx")
    return ds.add_column("idx", list(range(n)))


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _is_missing_value(x: object) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    # avoid pandas call for common fast-paths (strings)
    if isinstance(x, str):
        return False
    return pd.isna(x)


def _parse_structure_json(
    structure_json: str | None,
    cache: dict[str, Structure | None],
) -> Structure | None:
    """Parse pymatgen Structure JSON with a small per-task cache."""
    if _is_missing_value(structure_json):
        return None

    if not isinstance(structure_json, str):
        return None

    # None is a valid cached result for unparseable/missing structures
    cached = cache.get(structure_json, _STRUCTURE_PARSE_MISS)
    if cached is not _STRUCTURE_PARSE_MISS:
        return cached

    try:
        s = Structure.from_str(structure_json, fmt="json")
    except Exception as e:
        # Debug only: stack traces here can dominate runtime at scale.
        logger.debug("Failed to parse structure JSON: %s", e, exc_info=True)
        s = None

    cache[structure_json] = s
    return s


def _require_columns(ds: Dataset, required: set[str], *, name: str) -> None:
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _best_mp_start_method_hint() -> None:
    method = multiprocessing.get_start_method(allow_none=True)
    # On Linux, "fork" enables true COW for globals => big speedup.
    if method is not None and method != "fork":
        logger.warning(
            "Multiprocessing start method is %r. For best performance on Linux/HPC, "
            "prefer 'fork' to reduce serialization and enable copy-on-write.",
            method,
        )


def _is_fork_available() -> bool:
    if sys.platform == "win32":
        return False
    try:
        ctx = multiprocessing.get_context("fork")
        return ctx.get_start_method() == "fork"
    except ValueError:
        return False


def _mp_context_fork_if_available():
    if _is_fork_available():
        return multiprocessing.get_context("fork")
    return None


def _init_matches_worker(
    sys_groups: dict[str, list[NoveltySysItem]],
    ref_groups: dict[str, list[NoveltyRefItem]],
) -> None:
    global _NOVELTY_SYS_GROUPS, _NOVELTY_REF_GROUPS
    _NOVELTY_SYS_GROUPS = sys_groups
    _NOVELTY_REF_GROUPS = ref_groups


def _matches_worker(
    systems_chunk: list[str],
    *,
    max_matches: int | None = None,
) -> dict[int, list[str] | None]:
    """Compute structure matches vs reference for a batch of systems.

    Args:
        systems_chunk: Group of systems being processed.
        max_matches: If set, stop after collecting this many matches per system item.

    Returns:
        Mapping idx -> list of matching reference material_ids, or None if structure missing/unparseable.
    """
    if _NOVELTY_SYS_GROUPS is None or _NOVELTY_REF_GROUPS is None:
        raise RuntimeError("Novelty worker is not initialized.")

    sys_groups = _NOVELTY_SYS_GROUPS
    ref_groups = _NOVELTY_REF_GROUPS

    matcher = StructureMatcher(attempt_supercell=True)

    out: dict[int, list[str]| None] = {}

    for system in systems_chunk:
        sys_items = sys_groups.get(system, [])
        ref_items = ref_groups.get(system, [])
        if not sys_items:
            continue

        structures_cache: dict[str, Structure | None] = {}
        parsed_ref: list[tuple[str, Structure]] = []
        for ref in ref_items:
            ref_s = _parse_structure_json(ref.structure_json, structures_cache)
            if ref_s is not None:
                parsed_ref.append((ref.material_id, ref_s))

        for item in sys_items:
            sys_s = _parse_structure_json(item.structure_json, structures_cache)
            if sys_s is None:
                out[item.idx] = None
                continue

            try:
                matches: list[str] = []
                for material_id, ref_s in parsed_ref:
                    if matcher.fit(sys_s, ref_s):
                        matches.append(material_id)
                        if max_matches is not None and len(matches) >= max_matches:
                            break

                out[item.idx] = matches
            except Exception as e:
                logger.exception(
                    "Novelty matching failed for system=%s, idx=%s due to error: %s",
                    system,
                    item.idx,
                    e
                )
                out[item.idx] = None

    return out


def _init_uniqueness_worker_small_group(group_items: dict[str, list[UniqueItem]]) -> None:
    global _UNIQUENESS_GROUP_ITEMS
    _UNIQUENESS_GROUP_ITEMS = group_items


def _uniqueness_worker_small_group(
    systems_chunk: list[str],
    double_check: bool = False,
) -> tuple[dict[int, bool | None], dict[int, list[int] | None]]:
    """Compute uniqueness for a batch of systems (small groups only).

    For each equivalence component, choose canonical representative:
      - lowest energy (energy=None treated as +inf),
      - tie-break by lowest idx.

    Returns:
        Mapping idx -> is_unique (True/False), or None if structure missing/unparseable.
    """
    if _UNIQUENESS_GROUP_ITEMS is None:
        raise RuntimeError("Uniqueness worker is not initialized.")

    group_items = _UNIQUENESS_GROUP_ITEMS

    out: dict[int, bool | None] = {}
    matches: dict[int, list[int] | None] = {}
    for sys_name in systems_chunk:
        sys_out, sys_matches = _compute_uniqueness_for_group(
            group_items[sys_name],
            double_check=double_check,
        )
        out.update(sys_out)
        matches.update(sys_matches)

    return out, matches


def _compute_uniqueness_for_group(
    items: list[UniqueItem],
    double_check: bool = False,
) -> tuple[dict[int, bool | None], dict[int, list[int] | None]]:
    """Compute uniqueness flags for one group using StructureMatcher.group_structures().

    Returns:
        out: idx -> True/False, or None if structure missing/unparseable.
        matches: idx -> [] for winners, [winner_idx] for non-winners, or None if unknown.
    """
    matcher = StructureMatcher(attempt_supercell=True)

    cache: dict[str, Structure | None] = {}
    parsed: list[tuple[int, float | None, Structure]] = []
    out: dict[int, bool | None] = {}
    matches: dict[int, list[int] | None] = {}

    for it in items:
        s = _parse_structure_json(it.structure_json, cache)
        if s is None:
            out[it.idx] = None
            matches[it.idx] = None  # keep matches aligned with out for missing/unparseable
            continue
        parsed.append((it.idx, it.energy, s))

    if not parsed:
        return out, matches

    structures = [s for _, _, s in parsed]
    idx_by_obj_id = {id(s): idx for idx, _, s in parsed}
    energy_by_idx = {idx: e for idx, e, _ in parsed}

    def energy_key(m: int) -> tuple[float, int]:
        e = energy_by_idx.get(m)
        return (float("inf") if e is None else float(e), m)

    try:
        groups = matcher.group_structures(structures)
    except Exception as exc:
        # Worst-case fallback: if grouping fails, mark all parseable as None to avoid lying.
        logger.exception("group_structures failed: %s", exc)
        for idx, _, _ in parsed:
            out[idx] = None
            matches[idx] = None
        return out, matches

    for grp in groups:
        member_idxs = [idx_by_obj_id[id(s)] for s in grp]

        winner = min(member_idxs, key=energy_key)
        for m in member_idxs:
            out[m] = (m == winner)
            matches[m] = [] if m == winner else [winner]

    if not double_check:
        return out, matches

    # Second pass only over current winners: cheap way to catch leftovers from non-transitivity.
    unique_idxs = [idx for idx, flag in out.items() if flag is True]
    if len(unique_idxs) < 2:
        return out, matches

    structure_by_idx = {idx: s for idx, _, s in parsed}
    unique_structures: list[Structure] = [structure_by_idx[idx] for idx in unique_idxs]
    idx_by_obj_id_2 = {id(s): idx for idx, s in zip(unique_idxs, unique_structures)}

    try:
        groups2 = matcher.group_structures(unique_structures)
    except Exception as exc:
        # If the double-check fails, keep first-pass results (better than nuking correctness).
        logger.exception("group_structures failed on double_check: %s", exc)
        return out, matches

    # Map: first-pass winner -> final winner after second pass (partitioned, so no chains expected).
    final_winner_by_first_winner = {idx: idx for idx in unique_idxs}

    for grp in groups2:
        member_idxs = [idx_by_obj_id_2[id(s)] for s in grp]
        winner = min(member_idxs, key=energy_key)
        for m in member_idxs:
            final_winner_by_first_winner[m] = winner

    # Update winners themselves, then "repoint" everyone else who referenced an old winner.
    for idx in unique_idxs:
        new_winner = final_winner_by_first_winner.get(idx, idx)
        out[idx] = (idx == new_winner)
        matches[idx] = [] if out[idx] else [new_winner]

    for idx, flag in list(out.items()):
        if flag is None:
            continue
        m = matches.get(idx)
        if not m:
            # winners are handled above ([]) or missing/unparseable (None); keep as-is.
            continue

        old_winner = m[0]
        new_winner = final_winner_by_first_winner.get(old_winner, old_winner)
        matches[idx] = [new_winner]
        # Non-winners can't become winners in this scheme; keep out[idx] as False.
        out[idx] = False

    return out, matches


def _init_uniqueness_worker_big_group(group_items: dict[str, list[UniqueItem]]) -> None:
    global _BIG_UNIQUENESS_GROUP_ITEMS
    _BIG_UNIQUENESS_GROUP_ITEMS = group_items


def _uniqueness_worker_big_group(
    sys_name: str,
    double_check: bool = False
) -> tuple[dict[int, bool | None], dict[int, list[int] | None]]:
    if _BIG_UNIQUENESS_GROUP_ITEMS is None:
        raise RuntimeError("Big-group uniqueness worker is not initialized.")
    return _compute_uniqueness_for_group(
        _BIG_UNIQUENESS_GROUP_ITEMS[sys_name],
        double_check=double_check,
    )


def get_uniqueness(
    ds: Dataset,
    *,
    group_key: str = "reduced_formula",
    structure_key: str = "relaxed_structure",
    energy_key: str = "energy_above_hull",
    self_matches_key: str = "self_matches",
    reuse_self_matches: bool = False,
    num_proc: int = 16,
    big_group_threshold: int = 500,
    chunk_size_systems: int = 500,
    double_check: bool = False,
) -> tuple[list[bool | None], list[list[int] | None]]:
    """Compute matches inside one dataset within each formula group.

    Two structures are considered equivalent if `StructureMatcher().fit(a, b)` is True.
    For each equivalence component, we pick a single canonical representative:
      - lowest energy (energy=None treated as +inf), then lowest idx.

    Args:
        ds: Dataset with `group_key`, `structure_key`, and `energy_key`.
        group_key: Column used to define “uniqueness groups”.
        structure_key: Column containing pymatgen Structure JSON.
        energy_key: Column containing energy above hull (float).
        self_matches_key: Column containing cached within-dataset structure matches.
        reuse_self_matches: Whether to reuse existing cached `self_matches_key` and
            `is_unique` values. The safe default is False because `self_matches`
            can become stale after energy changes.
        num_proc: Number of processes to parallelize across formulas.
        big_group_threshold: Formulas with >= this many rows are treated as "big groups"
            and processed with chunked intra-formula parallelism.
        chunk_size_systems: Number of systems to be processed by a worker.
        double_check: If True, run a second pass on the first-pass winners to reduce
            missed duplicates from non-transitive StructureMatcher grouping (heuristic).
    Returns:
        Tuple of columns aligned to ds row positions:
          - `is_unique`: bool if computed, None if missing/unparseable/ungroupable.
          - `self_matches`: [] for winners, [winner_idx] for non-winners, or None.
    """
    _best_mp_start_method_hint()
    _require_columns(ds, {group_key, structure_key, energy_key}, name="Dataset")

    # Reusing cached uniqueness is unsafe after energy changes, so it is opt-in only.
    existing_is_unique: list[bool | None] | None = None
    existing_self_matches: list[list[int] | None] | None = None
    if reuse_self_matches and "is_unique" in ds.column_names:
        existing_is_unique = list(ds["is_unique"])
    if reuse_self_matches and self_matches_key in ds.column_names:
        existing_self_matches = list(ds[self_matches_key])

    ds = ds.select_columns([group_key, energy_key, structure_key])
    ds = _ensure_sequential_idx(ds)

    df = ds.to_pandas()

    # Missing group => cannot decide uniqueness.
    missing_group_mask = df[group_key].isna()
    missing_group_idxs = df.loc[missing_group_mask, "idx"].astype(int).tolist()
    df = df.loc[~missing_group_mask]

    if existing_is_unique is not None:
        non_unique_idxs = {i for i, v in enumerate(existing_is_unique) if v is False}
        if non_unique_idxs:
            df = df.loc[~df["idx"].isin(non_unique_idxs)]

    if df.empty:
        if existing_is_unique is None:
            out  = [None] * len(ds)
        else:
            out =  [None if v is None else bool(v) for v in existing_is_unique]

        if existing_self_matches is None:
            self_matches_out: list[list[int] | None] = [None] * len(ds)
        else:
            self_matches_out = existing_self_matches.copy()

        # Ensure missing-group rows are None
        for idx in missing_group_idxs:
            out[idx] = None
            self_matches_out[idx] = None
        return out, self_matches_out

    group_counts = df[group_key].astype(str).value_counts()
    big_groups = set(group_counts[group_counts >= big_group_threshold].index.tolist())

    # Build items grouped by formula
    grouped: dict[str, list[UniqueItem]] = {}
    for system, group in df.groupby(group_key, sort=False):
        sys_name = str(system)
        items: list[UniqueItem] = []
        for row in group.itertuples(index=False):
            energy = getattr(row, energy_key)
            if energy is None or _is_nan(energy) or pd.isna(energy):
                energy = None
            else:
                energy = float(energy)

            items.append(
                UniqueItem(
                    idx=int(row.idx),
                    structure_json=getattr(row, structure_key),
                    energy=energy,
                )
            )
        grouped[sys_name] = items

    # Split small vs big groups
    small_groups_items: dict[str, list[UniqueItem]] = {}
    big_groups_items: dict[str, list[UniqueItem]] = {}

    for sys_name, items in grouped.items():
        if sys_name in big_groups:
            big_groups_items[sys_name] = items
        else:
            small_groups_items[sys_name] = items

    # biggest systems first
    big_sys_names = sorted(
        big_groups_items.keys(),
        key=lambda k: len(big_groups_items[k]),
        reverse=True,
    )

    idx_to_unique: dict[int, bool | None] = {}
    idx_to_matches: dict[int, list[int] | None] = {}

    mp_context = _mp_context_fork_if_available()

    if small_groups_items:
        systems_chunks = list(
            _iter_chunks(
                list(small_groups_items.keys()),
                chunk_size_systems
            )
        )

        with ProcessPoolExecutor(
            max_workers=min(num_proc, len(systems_chunks)),
            mp_context=mp_context,
            initializer=_init_uniqueness_worker_small_group,
            initargs=(small_groups_items,),
        ) as pool:
            worker = partial(
                _uniqueness_worker_small_group,
                double_check=double_check,
            )
            futures = [
                pool.submit(
                    worker,
                    chunk
                ) for chunk in systems_chunks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Self matches calculation",
                unit=f"chunk ({chunk_size_systems} systems)",
            ):
                out, matches = future.result()
                idx_to_unique.update(out)
                idx_to_matches.update(matches)

    if big_groups_items:
        with ProcessPoolExecutor(
            max_workers=min(num_proc, len(big_groups_items)),
            mp_context=mp_context,
            initializer=_init_uniqueness_worker_big_group,
            initargs=(big_groups_items,),
        ) as pool:
            worker = partial(
                _uniqueness_worker_big_group,
                double_check=double_check,
            )
            futures = [
                pool.submit(worker, sys_name) for sys_name in big_sys_names
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Uniqueness calculation (big groups)",
                unit="system",
            ):
                out, matches = future.result()
                idx_to_unique.update(out)
                idx_to_matches.update(matches)

    if existing_is_unique is None:
        is_unique_col: list[bool | None] = [None] * len(ds)
    else:
        is_unique_col = [None if v is None else bool(v) for v in existing_is_unique]

    if existing_self_matches is None:
        self_matches_col: list[list[int] | None] = [None] * len(ds)
    else:
        self_matches_col = existing_self_matches.copy()

    for idx, v in idx_to_matches.items():
        self_matches_col[idx] = v

    for idx, flag in idx_to_unique.items():
        is_unique_col[idx] = flag

    for idx in missing_group_idxs:
        is_unique_col[idx] = None
        self_matches_col[idx] = None

    return is_unique_col, self_matches_col


def get_matches_to_reference(
    ds: Dataset,
    ref_ds: Dataset,
    *,
    group_key: str = "reduced_formula",
    structure_key: str = "relaxed_structure",
    matches_key: str = "matches",
    num_proc: int = 16,
    max_matches: int | None = 1,
    reuse_matches: bool = True,
) -> list[list[str] | None]:
    """Compute `matches` for ds by comparing structures to ref_ds within same system.

   Semantics:
      - matches == []  -> novel (no match in ref for that formula)
      - matches == [..] -> non-novel
      - matches is None -> missing/unparseable structure JSON (system-side)

    Important:
      - If a formula has no entries in ref_ds, all rows for that formula are treated as novel ([]).

    Args:
        ds: Target dataset. Must contain `group_key` and `structure_key`.
        ref_ds: Reference dataset. Must contain `group_key`, `structure_key`, and "material_id".
        group_key: Column used to group systems before matching (e.g., reduced formula).
        structure_key: Column containing pymatgen Structure JSON.
        matches_key: Column containing cached matches to the reference dataset.
        num_proc: Number of processes for per-system parallelism.
        max_matches: Early-stop after N matches per row. Use 1 for S.U.N. speed.
            Set to None to collect all matches.

    Returns:
        List of matches aligned to ds row positions (Sequence[string]) where:
          - [] -> novel (no match),
          - [..] -> non-novel,
          - None -> cannot evaluate (missing/unparseable structure OR missing group).
    """
    _best_mp_start_method_hint()
    _require_columns(ds, {group_key, structure_key}, name="Dataset")
    _require_columns(ref_ds, {group_key, structure_key, "material_id"}, name="Reference dataset")

    if matches_key not in ds.column_names:
        ds = ds.add_column(
            matches_key,
            [None] * len(ds),
            feature=Sequence(Value("string")),
        )
    else:
        if not reuse_matches:
            ds = ds.remove_columns([matches_key])
            ds = ds.add_column(
                matches_key,
                [None] * len(ds),
                feature=Sequence(Value("string")),
            )

    # Load only necessary columns to reduce memory.
    ds = ds.select_columns([group_key, matches_key, structure_key])
    ds = _ensure_sequential_idx(ds)

    sys_df = ds.to_pandas()
    sys_df = sys_df[sys_df[matches_key].isna()]

    if sys_df.empty:
        return list(ds[matches_key])

    # Missing group => cannot match.
    missing_group_mask = sys_df[group_key].isna()
    idx_to_matches: dict[int, list[str] | None] = {}
    for idx in sys_df.loc[missing_group_mask, "idx"].astype(int).tolist():
        idx_to_matches[idx] = None
    sys_df = sys_df.loc[~missing_group_mask]

    if sys_df.empty:
        matches_col = list(ds[matches_key])
        for idx, matches in idx_to_matches.items():
            matches_col[idx] = matches
        return matches_col

    needed_systems = set(sys_df[group_key].astype(str).unique().tolist())

    # Group into compact payloads per system.
    sys_groups: dict[str, list[NoveltySysItem]] = {}
    for system, g in sys_df.groupby(group_key, sort=False):
        sys_name = str(system)
        items: list[NoveltySysItem] = []
        for row in g.itertuples(index=False):
            items.append(
                NoveltySysItem(
                    int(row.idx),
                    getattr(row, structure_key),
                ),
            )
        sys_groups[sys_name] = items

    ref_df = ref_ds.select_columns(["material_id", group_key, structure_key]).to_pandas()
    ref_df = ref_df[~ref_df[group_key].isna()]
    ref_df[group_key] = ref_df[group_key].astype(str)
    ref_df = ref_df[ref_df[group_key].isin(needed_systems)]

    ref_groups: dict[str, list[NoveltyRefItem]] = {}
    for system, g in ref_df.groupby(group_key, sort=False):
        items: list[NoveltyRefItem] = []
        for row in g.itertuples(index=False):
            items.append(
                NoveltyRefItem(
                    material_id=str(row.material_id),
                    structure_json=getattr(row, structure_key),
                )
            )
        ref_groups[str(system)] = items

    # Formulas with no reference rows are "novel by definition".
    no_ref_systems = [sys_name for sys_name in sys_groups.keys() if sys_name not in ref_groups]
    for sys_name in no_ref_systems:
        for item in sys_groups[sys_name]:
            # Still honor system-side missing structure => None
            idx_to_matches[item.idx] = [] if not _is_missing_value(item.structure_json) else None

    # Remaining formulas need actual matching
    chunks = list(
        _iter_chunks_by_cost(sys_groups, ref_groups)
    )
    if chunks:
        chunk_sizes = [
            sum(len(sys_groups[s]) for s in chunk) for chunk in chunks
        ]
        total_structures = sum(chunk_sizes)

        mp_context = _mp_context_fork_if_available()
        with ProcessPoolExecutor(
            max_workers=min(num_proc, len(chunks)),
            mp_context=mp_context,
            initializer=_init_matches_worker,
            initargs=(sys_groups, ref_groups),
        ) as pool:
            n_structures_by_future = {}
            for chunk, n_structures in zip(chunks, chunk_sizes):
                future = pool.submit(_matches_worker, chunk, max_matches=max_matches)
                n_structures_by_future[future] = n_structures

            with tqdm(
                    total=total_structures,
                    desc="Reference matches calculation",
                    unit="structure",
            ) as pbar:
                for future in as_completed(n_structures_by_future):
                    result = future.result()
                    idx_to_matches.update(result)
                    pbar.update(n_structures_by_future[future])

    matches_col = list(ds[matches_key])
    for idx, matches in idx_to_matches.items():
        matches_col[idx] = matches

    return matches_col


def compute_dataset_sun(
    ds: Dataset,
    ref_ds: Dataset,
    *,
    group_key: str = "reduced_formula",
    structure_key: str = "relaxed_structure",
    energy_key: str = "energy_above_hull",
    matches_key: str = "matches",
    self_matches_key: str = "self_matches",
    reuse_matches: bool = True,
    reuse_self_matches: bool = False,
    num_proc: int = 16,
    novelty_max_matches: int | None = 1,
    chunk_size_systems: int = 500,
    stability_threshold: float = 0.1,
) -> Dataset:
    """Compute S.U.N. (stable, unique, novel) metric columns for a dataset.

    This implementation:
      - ensures stable idx,
      - computes novelty matches (early stopping by default),
      - computes deterministic uniqueness via DSU,
      - computes is_stable/is_novel/S.U.N. without ds random access inside multiprocessing.

    Args:
        ds: Target dataset.
        ref_ds: Reference dataset.
        group_key: Column used to group examples for novelty and uniqueness computations.
        structure_key: Column containing pymatgen Structure JSON.
        energy_key: Column with energy above hull.
        matches_key: Column containing cached matches to the reference dataset.
        self_matches_key: Column containing cached within-dataset structure matches.
        reuse_matches: Whether to reuse existing cached `matches_key` values.
        reuse_self_matches: Whether to reuse existing cached `self_matches_key`
            and `is_unique` values. The safe default is False because
            `self_matches` can become stale after energy changes.
        num_proc: Number of processes.
        novelty_max_matches: For novelty, stop after N matches per row. Use 1 for SUN.
        chunk_size_systems: Number of systems to be processed by a worker.
        stability_threshold: Threshold for determining stability.


    Returns:
        Dataset with columns: is_stable, is_novel, is_unique, S.U.N.
    """
    matches_col = get_matches_to_reference(
        ds,
        ref_ds,
        group_key=group_key,
        structure_key=structure_key,
        matches_key=matches_key,
        num_proc=num_proc,
        max_matches=novelty_max_matches,
        reuse_matches=reuse_matches,
    )

    is_unique_col, self_matches_col = get_uniqueness(
        ds,
        group_key=group_key,
        structure_key=structure_key,
        energy_key=energy_key,
        self_matches_key=self_matches_key,
        reuse_self_matches=reuse_self_matches,
        num_proc=num_proc,
        chunk_size_systems=chunk_size_systems,
        double_check=True,
    )

    energies = list(ds[energy_key]) if energy_key in ds.column_names else [None] * len(ds)

    is_stable_col: list[bool | None] = []
    is_novel_col: list[bool | None] = []
    sun_col: list[bool | None] = []

    for energy, matches, is_unique in tqdm(
        zip(energies, matches_col, is_unique_col),
        total=len(energies),
        desc="Calculating S.U.N.",
    ):
        if energy is None or _is_nan(energy) or pd.isna(energy):
            is_stable = None
        else:
            is_stable = float(energy) < stability_threshold

        if matches is None:
            is_novel = None
        else:
            is_novel = len(matches) == 0

        if is_stable is None or is_novel is None or is_unique is None:
            sun = None
        else:
            sun = is_stable and is_novel and is_unique

        is_stable_col.append(is_stable)
        is_novel_col.append(is_novel)
        sun_col.append(sun)

    for col_name, col_values, col_type in (
        ("is_stable", is_stable_col, Value("bool")),
        ("is_unique", is_unique_col, Value("bool")),
        ("is_novel", is_novel_col, Value("bool")),
        ("S.U.N.", sun_col, Value("bool")),
        (matches_key, matches_col, List(Value("string"))),
        (self_matches_key, self_matches_col, List(Value("int64"))),
    ):
        if col_name in ds.column_names:
            ds = ds.remove_columns(col_name)
        ds = ds.add_column(col_name, col_values, feature=col_type)

    return ds
