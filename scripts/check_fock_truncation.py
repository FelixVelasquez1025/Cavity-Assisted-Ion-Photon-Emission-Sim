"""Compare the unchanged demo pulse sequence with two and three Fock states.

Run from the repository root: uv run python scripts/check_fock_truncation.py
Results are printed as JSON; no files or default settings are changed.
"""

import json

import numpy as np
import qutip

from caipe_sim.demo import simulate_demo
from caipe_sim.yb171_simulator import solver_options


def compare_fock_truncations() -> dict:
    small = simulate_demo(fock_dimension=2)
    large = simulate_demo(fock_dimension=3)
    np.testing.assert_array_equal(small["time_us"], large["time_us"])
    differences = {
        key: float(np.max(np.abs(small[key] - large[key])))
        for key in ("P_S", "P_P", "P_D", "N_left", "N_right")
    }
    return {
        "fock_dimensions_per_mode": [2, 3],
        "time_window_us": [float(small["time_us"][0]), float(small["time_us"][-1])],
        "samples": len(small["time_us"]),
        "max_absolute_difference": differences,
        "max_manifold_population_difference": max(differences[k] for k in ("P_S", "P_P", "P_D")),
        "qutip_version": qutip.__version__,
        "solver_options": solver_options(),
    }


if __name__ == "__main__":
    print(json.dumps(compare_fock_truncations(), indent=2))
