from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from datasets import Dataset, DatasetDict

from prisma.data.loading import load_dataset_source
from prisma.data.persistence import (
    load_dataset_metadata,
    load_saved_dataset,
    save_dataset,
)
from prisma.data.preparation import materialize_training_splits, prepare_dataset


def build_parser(prog: str = "prisma data") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Prepare, inspect, and publish PRISMA datasets.",
    )
    commands = parser.add_subparsers(dest="data_command", required=True)
    _add_prepare_parser(commands)
    _add_inspect_parser(commands)
    _add_push_parser(commands)
    _add_embed_parser(commands)
    return parser


def _add_embed_parser(commands) -> None:
    parser = commands.add_parser(
        "embed", help="Add optional PET or MACE structure embeddings."
    )
    parser.add_argument(
        "source", help="Prepared dataset directory or Hub dataset name."
    )
    parser.add_argument("--source-type", choices=("auto", "hub"), default="auto")
    parser.add_argument("--config-name")
    parser.add_argument("--revision")
    parser.add_argument("--extractor", choices=("pet", "mace"), required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Local extractor checkpoint file."
    )
    parser.add_argument(
        "--column", default="embedding", help="Output column (default: embedding)."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default="cpu", help="Extractor device, e.g. cpu or cuda:0."
    )
    parser.add_argument(
        "--max-num-atoms",
        type=int,
        help="Keep structures up to this atom count, as training would.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Materialize this fraction of train as a valid split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--num-proc", type=int, help="Worker processes.")
    parser.add_argument("--output", required=True, help="New local dataset directory.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace the output directory."
    )


def _add_prepare_parser(commands) -> None:
    parser = commands.add_parser(
        "prepare",
        help="Normalize and save a training dataset.",
    )
    parser.add_argument("source", help="Local source path or Hub dataset name.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--source-type",
        choices=("auto", "hub"),
        default="auto",
        help="Use 'hub' for a remote dataset (default: local auto-detection).",
    )
    parser.add_argument("--config-name", help="Hub dataset configuration name.")
    parser.add_argument("--revision", help="Hub dataset revision.")
    parser.add_argument(
        "--structure-column",
        default="structure",
        help="Column containing structure data.",
    )
    parser.add_argument(
        "--structure-format",
        choices=("auto", "pymatgen-json", "cif", "poscar"),
        default="auto",
        help="Format used by the structure column (default: auto).",
    )
    parser.add_argument(
        "--id-column",
        default="material_id",
        help="Column to normalize as material_id.",
    )
    parser.add_argument(
        "--split-column",
        help="Column containing train, valid, or test split names.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Fraction of train data to store as the valid split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--num-proc", type=int, help="Worker processes.")
    parser.add_argument(
        "--max-shard-size",
        help="Maximum saved Arrow shard size, for example 500MB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset.",
    )


def _add_inspect_parser(commands) -> None:
    parser = commands.add_parser("inspect", help="Show dataset splits and columns.")
    parser.add_argument("source", help="Saved dataset path or Hub dataset name.")
    parser.add_argument(
        "--source-type",
        choices=("auto", "hub"),
        default="auto",
    )
    parser.add_argument("--config-name", help="Hub dataset configuration name.")
    parser.add_argument("--revision", help="Hub dataset revision.")


def _add_push_parser(commands) -> None:
    parser = commands.add_parser("push", help="Publish a saved dataset to the Hub.")
    parser.add_argument("source", help="Dataset directory created by prepare.")
    parser.add_argument("repo_id", help="Destination Hub dataset repository.")
    parser.add_argument("--config-name", default="default")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--revision", help="Destination repository revision.")
    parser.add_argument("--create-pr", action="store_true")
    parser.add_argument("--commit-message")
    parser.add_argument("--commit-description")
    parser.add_argument("--max-shard-size")
    parser.add_argument("--num-proc", type=int)


def main(argv: Sequence[str] | None = None, *, prog: str = "prisma data") -> None:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        if args.data_command == "prepare":
            _prepare(args)
        elif args.data_command == "inspect":
            _inspect(args)
        elif args.data_command == "embed":
            _embed(args)
        else:
            _push(args)
    except (OSError, TypeError, ValueError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


def _embed(args) -> None:
    from prisma.data.embeddings import embed_dataset
    from prisma.embeddings import create_embedder

    output = Path(args.output).expanduser().resolve()
    if (
        args.source_type == "auto"
        and output == Path(args.source).expanduser().resolve()
    ):
        raise ValueError(
            "Choose a different output directory to preserve the source dataset."
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --overwrite to replace it."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    dataset = load_dataset_source(
        args.source,
        source_type=args.source_type,
        config_name=args.config_name,
        revision=args.revision,
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict(train=dataset)
    embedder = create_embedder(args.extractor, args.checkpoint, args.device)
    skip_report = {}
    result = embed_dataset(
        dataset,
        embedder,
        column=args.column,
        batch_size=args.batch_size,
        skip_report=skip_report,
    )
    materialized_report = None
    if args.max_num_atoms is not None or args.validation_fraction is not None:
        result, materialized_report = materialize_training_splits(
            result,
            max_num_atoms=args.max_num_atoms,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            num_proc=args.num_proc,
        )
    metadata = load_dataset_metadata(args.source) if args.source_type == "auto" else {}
    metadata.setdefault("embeddings", {})[args.column] = embedder.metadata()
    metadata.setdefault("embedding_filter", {})[args.column] = skip_report
    if materialized_report is not None:
        metadata["materialized_training_splits"] = materialized_report
    save_dataset(
        result,
        output,
        overwrite=args.overwrite,
        metadata=metadata,
        num_proc=args.num_proc,
    )
    print(f"Saved dataset to {output}")
    print(f"Embedding column: {args.column}; dimension: {embedder.dimension}")
    for split, counts in skip_report.items():
        print(
            f"  {split}: {len(result[split])} kept, "
            f"{counts['count']} skipped (unsupported atomic numbers: "
            f"{counts['by_atomic_number']})"
        )
    print("\nUse this in your training YAML:")
    print(
        yaml.safe_dump(
            {
                "conditions": {
                    args.column: {"type": "vector", "input_dim": embedder.dimension}
                }
            },
            sort_keys=False,
        )
    )


def _prepare(args) -> None:
    dataset = prepare_dataset(
        args.source,
        source_type=args.source_type,
        config_name=args.config_name,
        revision=args.revision,
        structure_column=args.structure_column,
        structure_format=args.structure_format,
        id_column=args.id_column,
        split_column=args.split_column,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        num_proc=args.num_proc,
    )
    output_path = save_dataset(
        dataset,
        args.output,
        overwrite=args.overwrite,
        max_shard_size=args.max_shard_size,
        num_proc=args.num_proc,
    )
    print(f"Saved dataset to {output_path}\n")
    print("Structure conversion:")
    print(f"  input column: {args.structure_column}")
    print(f"  input format: {args.structure_format}")
    print("  stored format: pymatgen JSON\n")
    _print_dataset_summary(dataset)


def _inspect(args) -> None:
    dataset = load_dataset_source(
        args.source,
        source_type=args.source_type,
        config_name=args.config_name,
        revision=args.revision,
    )
    _print_dataset_summary(dataset)


def _push(args) -> None:
    dataset = load_saved_dataset(args.source)
    commit = dataset.push_to_hub(
        args.repo_id,
        config_name=args.config_name,
        private=True if args.private else None,
        revision=args.revision,
        create_pr=args.create_pr,
        commit_message=args.commit_message,
        commit_description=args.commit_description,
        max_shard_size=args.max_shard_size,
        num_proc=args.num_proc,
    )
    print(f"Published dataset to {args.repo_id}")
    if commit is not None and getattr(commit, "oid", None):
        print(f"Revision: {commit.oid}")


def _print_dataset_summary(dataset: Dataset | DatasetDict) -> None:
    datasets = dataset if isinstance(dataset, DatasetDict) else {"train": dataset}
    print("Splits:")
    for split, split_dataset in datasets.items():
        print(f"  {split}: {len(split_dataset):,} rows")

    print("\nColumns:")
    first_dataset = next(iter(datasets.values()))
    for name, feature in first_dataset.features.items():
        print(f"  {name}: {feature}")
