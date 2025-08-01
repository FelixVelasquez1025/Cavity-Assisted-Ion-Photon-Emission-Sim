import unittest
from caipe_sim.three_level_model import run_sim


class TestThreeLevelSimulation(unittest.TestCase):
    def test_output_structure(self):
        results = run_simulation()
        self.assertIsInstance(results, dict)
        self.assertIn("pop_g", results)
        self.assertIn("pop_e", results)
        self.assertIn("pop_s", results)
        self.assertIn("n_photon", results)
