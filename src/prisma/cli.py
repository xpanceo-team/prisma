from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from prisma.data.cli import main as data_main


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="prisma")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("data", help="Prepare, inspect, and publish datasets.")

    if argv and argv[0] == "data":
        data_main(argv[1:], prog="prisma data")
        return
    parser.parse_args(argv)
