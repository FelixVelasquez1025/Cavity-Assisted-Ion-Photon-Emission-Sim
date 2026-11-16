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


def test_simulator_file_loads_without_package_context(tmp_path):
    """An editor's file runner supplies no __package__, unlike the CLI."""
    import subprocess
    import sys

    source = Path(yb171_simulator.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         "import runpy, sys; "
         "module = runpy.run_path(sys.argv[1]); "
         "assert callable(module['interactive_populations'])", str(source)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
