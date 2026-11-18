import numpy as np
import pytest
from caipe_sim import yb171_simulator as existing
from caipe_sim.spam import atomic as a


def test_basis_spectra_and_ordering():
    assert len(a.STATES) == len(a.INDEX) == 20
    for name, manifold in a.MANIFOLDS.items():
        lo, hi = manifold.Fs
        assert a.hyperfine_hz(name, hi)-a.hyperfine_hz(name, lo) == pytest.approx(manifold.splitting_hz)
        assert sum((2*F+1)*a.hyperfine_hz(name, F) for F in manifold.Fs) == pytest.approx(0, abs=1e-5)
    assert a.hyperfine_hz("A", 0) > a.hyperfine_hz("A", 1)
    assert a.internal_hamiltonian(5).isherm


@pytest.mark.parametrize("upper_level", ["P", "A", "D"])
def test_every_decay_sublevel_normalizes(upper_level):
    for upper in [s for s in a.STATES if s[0] == upper_level]:
        channels = [(l, r) for u, l, r in a.decay_channels() if u == upper]
        total = sum(r for _, r in channels)
        assert total == pytest.approx(1/a.MANIFOLDS[upper_level].lifetime_s, rel=1e-10)
        for lower, fraction in a.BRANCHES[upper_level].items():
            assert sum(r for l, r in channels if l[0] == lower)/total == pytest.approx(fraction, rel=1e-10)
    cs = a.collapse_operators()
    out = sum((c.dag()*c for c in cs))
    for upper in [s for s in a.STATES if s[0] == upper_level]:
        assert out[a.INDEX[upper], a.INDEX[upper]].real == pytest.approx(1/a.MANIFOLDS[upper_level].lifetime_s, rel=1e-10)


@pytest.mark.parametrize("level,F,expected", [("S",1,1.4),("P",1,.47),("D",1,1.4),("D",2,.84), ("A",1,.923402946317)])
def test_all_zeeman_slopes(level, F, expected):
    # Foot §6.3, with NIST ASD gJ=1.320 for A; do NOT apply LS to A.
    h = a.internal_hamiltonian(1).full()-a.internal_hamiltonian(0).full()
    i, j = a.INDEX[level,F,1], a.INDEX[level,F,0]
    assert (h[i,i]-h[j,j]).real/(2*np.pi*1e6) == pytest.approx(expected, rel=.02)
    assert a.g_f(level, 0) == 0


def test_e1_coefficients_reuse_validated_convention():
    for lower in a.STATES:
        if lower[0] not in ("S", "D"):
            continue
        for upper in [s for s in a.STATES if s[0] == "P"]:
            if abs(upper[2]-lower[2]) > 1:
                continue
            key = "2S12_to_2P12" if lower[0] == "S" else "2D32_to_2P12"
            expected = existing.e1_matrix_element(key, upper[2]-lower[2], False, .5, a.MANIFOLDS[lower[0]].J,
                Fp=upper[1], F=lower[1], mFp=upper[2], mF=lower[2])/existing.REDUCED_E1_ELEMENTS[key]
            assert a.coefficient(lower, upper) == pytest.approx(expected)


def test_optional_clock_quadratic_shift():
    delta = (a.internal_hamiltonian(5, clock_quadratic=True)-a.internal_hamiltonian(5)).diag()
    assert (delta[a.INDEX["S",1,0]]-delta[a.INDEX["S",0,0]])/(2*np.pi) == pytest.approx(310.8*25)
