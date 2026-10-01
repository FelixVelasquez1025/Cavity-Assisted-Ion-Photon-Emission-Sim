"""Physical invariants for the pulse-enabled portfolio application."""

import numpy as np
import pytest
from qutip import basis, mesolve, qeye
from scipy.special import erf

from caipe_sim import yb171_simulator as model
from caipe_sim.demo import simulate_demo


def test_hyperfine_projectors_partition_the_basis():
    assert model.HF_BASIS_DIM == 16
    assert len(set(model.HF_BASIS_ORDER)) == 16
    projectors = [model.projector_for_level_hf(level) for level in ("S", "P", "D")]
    assert [int(p.tr()) for p in projectors] == [4, 4, 8]
    assert (sum(projectors) - qeye(16)).norm() < 1e-12
    for p in projectors:
        assert (p * p - p).norm() < 1e-12
    for i, state in enumerate(model.HF_BASIS_ORDER):
        assert model.hf_state_index(*state) == i


@pytest.mark.parametrize("pair", ["SP", "DP"])
def test_hyperfine_selection_rules_and_laser_hermiticity(pair):
    assert any(model.E1_DATA_HF[pair].values())
    for q, transitions in model.E1_DATA_HF[pair].items():
        for lower, upper, coeff in transitions:
            assert np.isfinite(coeff)
            assert upper[2] - lower[2] == q
            assert abs(upper[1] - lower[1]) <= 1
            assert (upper[1], lower[1]) != (0, 0)
    for h in model._build_systems_hf(pair, 1.0, 16).values():
        assert h.isherm


def test_cavity_coupling_is_hermitian():
    for h in model.H_cavity_DP_modes_hf(1.0, 2, 2).values():
        assert h.isherm
        assert h.dims == [[2, 2, 16], [2, 2, 16]]
        assert h.norm() > 0


def test_gaussian_pi_pulse_matches_analytic_rabi_solution():
    # With pi polarization and no decay/repump, S(F=0,m=0) couples only to
    # P(F=1,m=0). This is an isolated resonant two-level subsystem.
    ground = model.hf_state_index("S", 0.0, 0.0)
    excited = model.hf_state_index("P", 1.0, 0.0)
    h = model._build_systems_hf("SP", 1.0, 16)["pi"]
    times = np.linspace(0.0, 2e-6, 201)
    args = dict(drive_amp=2 * np.pi * 1e6, drive_t0=1e-6, drive_sigma=0.2e-6)
    p_e = basis(16, excited).proj()
    result = mesolve(
        [h * 0, [h, model.omega_drive]], basis(16, ground), times,
        e_ops=[p_e], args=args, options=model.solver_options(),
    )
    sigma, t0 = args["drive_sigma"], args["drive_t0"]
    area = args["drive_amp"] * sigma * np.sqrt(np.pi / 2) * (
        erf((times - t0) / (np.sqrt(2) * sigma)) - erf(-t0 / (np.sqrt(2) * sigma))
    )
    expected = np.sin(abs(h[excited, ground]) * area) ** 2
    assert expected.max() > 0.1
    np.testing.assert_allclose(result.expect[0], expected, atol=2e-5)


@pytest.fixture(scope="module")
def demo_data():
    return simulate_demo()


def test_driven_open_system_conserves_population(demo_data):
    populations = np.stack([demo_data[k] for k in ("P_S", "P_P", "P_D")])
    assert np.isfinite(populations).all()
    assert populations.min() > -1e-6
    assert populations.max() < 1 + 1e-6
    np.testing.assert_allclose(populations.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(populations[:, 0], [1, 0, 0], atol=1e-12)
    assert demo_data["P_P"].max() > 0.05
    assert demo_data["P_D"].max() > 0.01


def test_cavity_observables_are_nonnegative_and_not_trivial(demo_data):
    for key in ("N_left", "N_right"):
        assert np.isfinite(demo_data[key]).all()
        assert demo_data[key].min() > -1e-6
        assert demo_data[key].max() <= 1 + 1e-6
    assert demo_data["N_left"].max() > 1e-4


def test_zero_drive_preserves_ground_state():
    ground = basis(16, model.hf_state_index("S", 0.0, 0.0))
    h = model.eq_2_4_5_H_S_hf(0.0, 1e-4) + model.eq_2_4_6_H_D_hf(0.0, 1e-4)
    result = mesolve(h, ground, np.linspace(0, 1e-6, 20), e_ops=[ground.proj()])
    np.testing.assert_allclose(result.expect[0], 1.0, atol=1e-12)
