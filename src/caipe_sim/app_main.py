"""Command-line entry point for the pulse-enabled Yb-171 simulator."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="Export a small CPU simulation without opening windows"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="Demo output directory (default: outputs)"
    )
    args = parser.parse_args()
    if args.demo:
        from .demo import export_demo

        export_demo(args.output)
    else:
        from .yb171_simulator import interactive_populations

        interactive_populations()


if __name__ == "__main__":
    main()
