"""
Entry point for running the Yb171+ interactive simulation.

This keeps the `examples/rabi_flopping_Yb171.py` script intact and simply
loads it to launch the UI. A thin wrapper is provided to allow for future
atom selection; at present only Yb171 is supported.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow running as a script or as a module
if __package__ is None:
    src_root = Path(__file__).resolve().parents[1]  # .../src
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from caipe_sim.interactive_ui import interactive_populations
else:
    from .interactive_ui import interactive_populations


def main(atom: str = "Yb171") -> None:
    """Launch the interactive populations UI."""
    _ = atom  # placeholder for future atom selection, eventually want to make it so we can have a choice between what aroms to run 
    interactive_populations()


if __name__ == "__main__":
    main()
