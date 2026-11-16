"""Keep archived implementation imports outside the installed application."""

import ast
from pathlib import Path

from caipe_sim import yb171_simulator


def test_active_package_does_not_import_historical_helpers():
    package = Path(yb171_simulator.__file__).parent
    historical = {"hamiltonian_utils", "interactive_ui", "batch_runner",
                  "batch_viewer", "three_level_model", "visualization_utils", "archive"}
    for source in package.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *[item.name for item in node.names]]
            else:
                continue
            for name in names:
                assert not historical.intersection(name.split(".")), (source.name, name)
                assert name != "_clebsch_gordan", (source.name, name)
