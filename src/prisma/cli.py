from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from prisma.data.cli import main as data_main
from prisma.training.cli import main as training_main


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="prisma")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("data", help="Prepare, inspect, and publish datasets.")
    commands.add_parser("train", help="Train a crystal diffusion model.")

    if argv and argv[0] == "data":
        data_main(argv[1:], prog="prisma data")
        return
    if argv and argv[0] == "train":
        training_main(argv[1:], prog="prisma train")
        return
    parser.parse_args(argv)
