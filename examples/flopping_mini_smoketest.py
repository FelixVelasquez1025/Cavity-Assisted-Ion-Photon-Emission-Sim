"""Check populations in the pulse-enabled 16-state model on CPU."""

import numpy as np

from caipe_sim.demo import simulate_demo


if __name__ == "__main__":
    data = simulate_demo()
    populations = np.stack([data[key] for key in ("P_S", "P_P", "P_D")])
    np.testing.assert_allclose(populations.sum(axis=0), 1.0, atol=1e-6)
    assert populations.min() > -1e-6
    assert populations.max() < 1.0 + 1e-6
    assert data["P_P"].max() > 0.05, "Drive did not excite the atom"
    print("CPU hyperfine pulse smoke test passed")
