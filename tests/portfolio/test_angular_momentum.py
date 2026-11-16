"""Signed reference checks, independent of the implementation's Wigner wrappers."""

import numpy as np
import pytest

from caipe_sim import yb171_simulator as model


# Racah-normalized absorption matrix element <Fp,mp|d_q|F,m>/ <Jp||d||J>:
# (-1)^(Fp-mp) (Fp 1 F; -mp q m)
# * (-1)^(Jp+I+F+1) sqrt((2Fp+1)(2F+1)) {Jp Fp I; F J 1}.
# Convention: Steck, Rubidium 87 D Line Data, Sec. 3.3, Eqs. (35)-(36),
# translated to absorption order and the Racah normalization of Eq. (39).
# https://steck.us/alkalidata/rubidium87numbers.pdf
# With I=Jp=1/2, the following 6j/3j values give the listed signed
# hyperfine reduced factors and full coefficients by direct substitution:
# J,F,Fp,m,mp      6j          3j           hyperfine factor   coefficient
# 1/2,0,1,0,0     1/sqrt(6)  -1/sqrt(3)   1/sqrt(2)         1/sqrt(6)
# 1/2,1,0,0,0     1/sqrt(6)  -1/sqrt(3)  -1/sqrt(2)         1/sqrt(6)
# 1/2,1,1,0,1    -1/3        -1/sqrt(6)   1                -1/sqrt(6)
# 3/2,1,0,0,0    -1/sqrt(6)  -1/sqrt(3)   1/sqrt(2)        -1/sqrt(6)
# 3/2,1,1,0,1    -1/6        -1/sqrt(6)   1/2              -1/sqrt(24)
# 3/2,2,1,0,0     1/sqrt(12)  sqrt(2/15)  sqrt(5)/2        -1/sqrt(6)
# 3/2,2,1,2,1     1/sqrt(12)  1/sqrt(5)   sqrt(5)/2         1/2
# The F=1 rows have odd phase exponent 3; F=0,2 have even exponent 2,4.
@pytest.mark.parametrize(
    "pair,J,F,Fp,m,mp,hf_expected,coefficient_expected",
    [
        ("SP", .5, 0, 1, 0, 0, 1/np.sqrt(2), 1/np.sqrt(6)),
        ("SP", .5, 1, 0, 0, 0, -1/np.sqrt(2), 1/np.sqrt(6)),
        ("SP", .5, 1, 1, 0, 1, 1, -1/np.sqrt(6)),
        ("DP", 1.5, 1, 0, 0, 0, 1/np.sqrt(2), -1/np.sqrt(6)),
        ("DP", 1.5, 1, 1, 0, 1, .5, -1/np.sqrt(24)),
        ("DP", 1.5, 2, 1, 0, 0, np.sqrt(5)/2, -1/np.sqrt(6)),
        ("DP", 1.5, 2, 1, 2, 1, np.sqrt(5)/2, .5),
    ],
)
def test_hyperfine_coefficients_match_signed_reference(
    pair, J, F, Fp, m, mp, hf_expected, coefficient_expected
):
    assert model.hyperfine_angular_factor(.5, Fp, J, F) == pytest.approx(hf_expected)
    key = "2S12_to_2P12" if pair == "SP" else "2D32_to_2P12"
    coefficient = model.e1_matrix_element(
        key, mp-m, False, .5, J, Fp=Fp, F=F, mFp=mp, mF=m
    ) / model.REDUCED_E1_ELEMENTS[key]
    assert coefficient == pytest.approx(coefficient_expected)


@pytest.mark.parametrize(
    "level,F,reference_mhz_per_gauss",
    [("S12", 1, 1.4), ("P12", 1, .47), ("D32", 1, 1.4), ("D32", 2, .84)],
)
def test_zeeman_matches_reference(level, F, reference_mhz_per_gauss):
    # First-order per-mF weak-field values requested for the Yb+ model.
    # Landé formula and Bohr-magneton convention: arXiv:2509.04416, Eqs. 16-23.
    # That source treats neutral Yb; this test evaluates the formula for the
    # ion's S/P/D quantum numbers, not its neutral-atom tabulated g_J values.
    atom_level = getattr(model.Yb171, level)
    _, g_f = model.lande_g_factors(
        F, model.I_NUCLEAR, atom_level.J, atom_level.L, atom_level.S
    )
    coefficient = g_f * model.MU_B * 1e-4 / (2*np.pi*1e6)
    assert coefficient == pytest.approx(reference_mhz_per_gauss, rel=.02)
    if level != "P12":
        builder = model.eq_2_4_5_H_S_hf if level == "S12" else model.eq_2_4_6_H_D_hf
        h = builder(0.0, 1e-4)
        i0 = model.hf_state_index(level[0], F, 0)
        i1 = model.hf_state_index(level[0], F, 1)
        shift = float(np.real(h[i1, i1]-h[i0, i0])) / (2*np.pi*1e6)
        assert shift == pytest.approx(reference_mhz_per_gauss, rel=.02)
    # P currently has no diagonal Zeeman term: only its g_F formula is checked.


def test_hyperfine_metadata_reproduces_adopted_splittings():
    # E_F=A*K/2, I=1/2: splitting is A for S/P and 2*A for D.
    # Sources: physics-references.md; D splitting: PRA 76, 052314 (2007).
    from scipy.constants import h

    expected = {"S": 12.642812118466e9, "P": 2.105e9, "D": .86375e9}
    for name, atom_level in [("S", model.Yb171.S12), ("P", model.Yb171.P12), ("D", model.Yb171.D32)]:
        data = next(d for d in model.Yb171.level_data if d.level is atom_level)
        f_low, f_high = abs(atom_level.J-.5), atom_level.J+.5
        delta_k = f_high*(f_high+1)-f_low*(f_low+1)
        assert (data.Ahfs/h)*delta_k/2 == pytest.approx(expected[name])
        assert model.HFS_A_GHZ[name]*1e9 == pytest.approx(data.Ahfs/h)
