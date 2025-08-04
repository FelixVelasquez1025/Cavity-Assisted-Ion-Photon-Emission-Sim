import unittest
import numpy as np
from caipe_sim.three_level_model import sweep_kappa, sweep_gamma


class TestSweepFunctions(unittest.TestCase):
    def test_sweep_kappa_output_shapes(self):
        # Run the sweep
        kappa_vals, pop_g, pop_e, pop_s, n_photon = sweep_kappa()

        # Assert length consistency
        self.assertEqual(len(kappa_vals), len(pop_g))
        self.assertEqual(len(kappa_vals), len(pop_e))
        self.assertEqual(len(kappa_vals), len(pop_s))
        self.assertEqual(len(kappa_vals), len(n_photon))

        # Check monotonicity of sweep parameter
        self.assertTrue(np.all(np.diff(kappa_vals) > 0), "kappa values should increase")

        # Check all populations are in [0, 1]
        for pop_array in [pop_g, pop_e, pop_s]:
            self.assertTrue(
                np.all((0 <= np.array(pop_array)) & (np.array(pop_array) <= 1))
            )

    def test_sweep_gamma_output_shapes(self):
        gamma_vals, pop_g, pop_e, pop_s, n_photon = sweep_gamma()

        self.assertEqual(len(gamma_vals), len(pop_g))
        self.assertEqual(len(gamma_vals), len(pop_e))
        self.assertEqual(len(gamma_vals), len(pop_s))
        self.assertEqual(len(gamma_vals), len(n_photon))

        self.assertTrue(np.all(np.diff(gamma_vals) > 0), "gamma values should increase")

        for pop_array in [pop_g, pop_e, pop_s]:
            self.assertTrue(
                np.all((0 <= np.array(pop_array)) & (np.array(pop_array) <= 1))
            )


if __name__ == "__main__":
    unittest.main()
