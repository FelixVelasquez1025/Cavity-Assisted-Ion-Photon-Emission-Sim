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
