"""Compare the unchanged demo pulse sequence with two, three, four and five Fock states.

Run from the repository root: uv run python scripts/check_fock_truncation.py
Results are printed as JSON; no files or default settings are changed.
"""

import json

import numpy as np
import qutip

from caipe_sim.demo import simulate_demo
from caipe_sim.yb171_simulator import solver_options


def compare_fock_truncations() -> dict:
    results = {n: simulate_demo(fock_dimension=n) for n in (2, 3, 4, 5)}
    comparisons = {}
    for small, large in ((2, 3), (3, 4), (4, 5)):
        np.testing.assert_array_equal(results[small]["time_us"], results[large]["time_us"])
        comparisons[f"{small}_vs_{large}"] = {
            key: float(np.max(np.abs(results[small][key] - results[large][key])))
            for key in ("P_S", "P_P", "P_D", "N_left", "N_right")
        }
    three_vs_four = max(comparisons["3_vs_4"].values())
    return {
        "fock_dimensions_per_mode": [2, 3, 4, 5],
        "time_window_us": [float(results[2]["time_us"][0]), float(results[2]["time_us"][-1])],
        "samples": len(results[2]["time_us"]),
        "max_absolute_differences": comparisons,
        "threshold": 1e-3,
        "dimension_3_passes_threshold": three_vs_four < 1e-3,
        "dimension_4_passes_threshold": max(comparisons["4_vs_5"].values()) < 1e-3,
        "qutip_version": qutip.__version__,
        "solver_options": solver_options(),
    }



if __name__ == "__main__":
    print(json.dumps(compare_fock_truncations(), indent=2))
