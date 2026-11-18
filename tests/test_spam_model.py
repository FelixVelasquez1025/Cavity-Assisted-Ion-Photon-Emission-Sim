import numpy as np
import pytest
from caipe_sim.spam import atomic as a
from caipe_sim.spam.model import Beam, Tone, build_model, linear_polarization


def test_linear_polarization_normalization_and_limits():
    for theta in (0, .4, np.pi/2):
        eps = linear_polarization(theta)
        assert sum(abs(e)**2 for e in eps.values()) == pytest.approx(1)
        assert eps[1] == -eps[-1]
    assert linear_polarization(0)[0] == 1
    assert linear_polarization(np.pi/2)[1] == pytest.approx(-1/np.sqrt(2))


def test_single_tone_static_and_off_resonant_leakage_retained():
    model = build_model([Beam("369", 2*np.pi*1e6, .7)])
    assert model.hamiltonian.isherm
    assert model.omitted_tones == []
    assert model.full_time_dependent_hamiltonian().isconstant
    for l,u in [(('S',1,0),('P',1,1)), (('S',0,0),('P',1,0)), (('S',1,0),('P',0,0))]:
        assert abs(model.hamiltonian[a.INDEX[u],a.INDEX[l]]) > 0
    h = model.hamiltonian
    # In one carrier frame the diagonal differences retain BOTH GHz detunings.
    assert (h[a.INDEX['P',1,0],a.INDEX['P',1,0]]-h[a.INDEX['P',0,0],a.INDEX['P',0,0]]).real/(2*np.pi) == pytest.approx(2.105e9)


def test_secular_two_tone_against_full_time_dependent_reference():
    from caipe_sim.spam.validation import two_tone_comparison
    report = two_tone_comparison()
    assert report['max_population_difference'] < 1e-4
    assert report['max_density_frobenius_difference'] < 1e-3


def test_trace_hermiticity_and_positive_density_over_long_run():
    beams = [Beam('369', 2*np.pi*10e6, .96, sidebands=(Tone(14.748e9,2*np.pi*3e6),)),
             Beam('935', 2*np.pi*10e6, .96, sidebands=(Tone(-3.07e9,2*np.pi*3e6),))]
    model = build_model(beams, B_gauss=5)
    initial = (a.ket('S',0,0)+1j*a.ket('S',1,1)).unit()
    for rho in model.evolve(initial, [0, 1e-7, 1e-5, 1e-3]):
        assert rho.tr() == pytest.approx(1, abs=2e-7)
        assert np.max(abs(rho.full()-rho.full().conj().T)) < 1e-7
        assert np.linalg.eigvalsh(rho.full()).min() > -1e-7


def test_laser_dephasing_off_by_default_and_convention():
    base = build_model([Beam('369',1,.5)])
    extra = build_model([Beam('369',1,.5,linewidth_dephasing_s=100)])
    assert len(extra.collapses) == len(base.collapses)+1
    c = extra.collapses[-1]
    assert c[a.INDEX['P',1,0], a.INDEX['P',1,0]] == pytest.approx(np.sqrt(200))


def test_weak_probe_hyperfine_spectrum():
    from caipe_sim.spam.validation import hyperfine_probe_check
    report = hyperfine_probe_check()
    assert report['S_F1_peak_detunings_hz'] == pytest.approx([0,2.105e9],abs=1e5)
    assert report['P_peak_separation_hz'] == pytest.approx(2.105e9,abs=1e5)
    assert report['S_F0_peak_detuning_hz'] == pytest.approx(14.747812118466e9,abs=1e5)


def test_dark_subspace_and_zero_field_fluorescence_suppression():
    from caipe_sim.spam.validation import dark_state_check
    # Near-resonant S F1->P F0 row has rank one: two independent dark states.
    model = build_model([Beam('369',2*np.pi*20e6,np.deg2rad(55))])
    row = model.hamiltonian.full()[a.INDEX['P',0,0], [a.INDEX['S',1,m] for m in (-1,0,1)]]
    _, singular, vh = np.linalg.svd(row.reshape(1,3))
    assert singular[0] > 0
    assert abs(row@vh[1].conj()) < 1e-8
    report = dark_state_check()
    # Full model's off-resonant P F1 excitation weakly lifts the darkness.
    assert 0 < report['fluorescence_369_per_s']['0'] < 1e-3*report['fluorescence_369_per_s']['5']
    assert max(report['stationary_residual_norm_per_s'].values()) < 1e-5
    assert min(report['minimum_density_eigenvalues'].values()) > -1e-9


def test_finite_time_channel_complete_positivity():
    from caipe_sim.spam.validation import long_run_check
    report = long_run_check()
    assert report['minimum_choi_eigenvalue'] > -1e-8
    assert report['choi_hermiticity_error'] < 1e-8


def test_probe_resonances_also_appear_in_full_master_equation():
    from caipe_sim.spam.validation import finite_weak_probe_check
    for feature in finite_weak_probe_check():
        left, peak, right = feature['P_population_after_100ns']
        assert peak > 20*max(left,right)
        assert peak < 1e-5  # Weak probe: ground depletion/pumping stays negligible.
