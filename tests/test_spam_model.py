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
    beam = Beam("369", 2*np.pi*5e6, .7,
                sidebands=(Tone(14.748e9,2*np.pi*5e6),))
    model = build_model([beam], B_gauss=5)
    times = np.linspace(0, 3e-9, 61)
    initial = (a.ket("S",0,0)+a.ket("S",1,0)).unit()
    secular = model.evolve(initial, times)
    full = model.full_evolution(initial, times, max_step_s=1/(50*14.748e9))
    difference = max(np.max(abs(x.diag()-y.diag())) for x,y in zip(secular,full))
    assert difference < 1e-4


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
