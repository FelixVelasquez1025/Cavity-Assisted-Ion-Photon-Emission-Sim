import unittest
import numpy as np
from caipe_sim.three_level_model import run_sim
from caipe_sim.three_level_model import sweep_parameters


class TestThreeLevelSimulation(unittest.TestCase):
    def test_output_structure(self):
        results = run_sim()
        self.assertIsInstance(results, dict)
        self.assertIn("pop_g", results)
        self.assertIn("pop_e", results)
        self.assertIn("pop_s", results)
        self.assertIn("n_photon", results)

class TestSweepParameters(unittest.TestCase):
    def setup(self):
        self.constants = {
            "gamma": 0.05,
            "kappa": 0.2,
            "rabi": 2.0,
            "gval": 1.0,
            "detune_c": 1.0,
            "detune_l": 1.0,
            "Ncav": 4
        }
        self.range = np.linspace (0.1,1,5)
        self.params = ["gval","rabi","kappa","gamma"]

    def test_sweep_lengths(self):
        for param in self.params:
            with self.subTest(param=param):
                param_vals,pg,pe,ps,photons = sweep_parameters(param,self.range,self.constants)
                self.assertEqual(len(param_vals),5)
                self.assertEqual(len(pg),5)
                self.assertEqual(len(pe),5)
                self.assertEqual(len(ps),5)
                self.assertEqual(len(photons),5)

    def test_non_negative_expectation_value(self):
        for param in self.params:
            with self.subTest(param=param):
                _,_,_,_,photons= sweep_parameters(param,self.range,self.constants)
                for i in photons:
                    self.assertGreaterEqual(i,0)

    def test_sweep_types(self):
        for param in self.params:
            with self.subTest(param=param):
                param_vals,pg,pe,ps,photons = sweep_parameters(param, self.range, self.constants)
                self.assertIsInstance(param_vals, np.ndarray)
                


    






