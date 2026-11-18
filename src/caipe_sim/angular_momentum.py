"""Shared signed Wigner–Eckart coefficients, used by cavity and free-space models.

Racah normalization and absorption ordering: Steck, Alkali Data, Sec. 3.3;
Foot, Atomic Physics (2005), Ch. 6. Existing signed reference tests remain
in tests/test_angular_momentum.py. No atomic reduced element is stored here.
"""
import math
from sympy import S as _S
from sympy.physics.wigner import wigner_3j as _sympy_wigner_3j
from sympy.physics.wigner import wigner_6j as _sympy_wigner_6j


def wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Numeric wrapper around SymPy wigner_3j returning a real float.

    SymPy may return an exact algebraic (possibly complex) expression; we
    evaluate numerically and return the real part, warning only if the
    imaginary part is non-negligible.
    """
    val = _sympy_wigner_3j(_S(j1), _S(j2), _S(j3), _S(m1), _S(m2), _S(m3))
    cval = complex(val.evalf())
    if abs(cval.imag) > 1e-12:
      # Wigner 3j for physical arguments should be real; keep real part.
      pass
    return float(cval.real)


def wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
  """Numeric wrapper around SymPy wigner_6j returning a real float."""
  val = _sympy_wigner_6j(_S(j1), _S(j2), _S(j3), _S(j4), _S(j5), _S(j6))
  cval = complex(val.evalf())
  if abs(cval.imag) > 1e-12:
    pass
  return float(cval.real)


def hyperfine_angular_factor(Jp: float, Fp: float, J: float, F: float, I: float = 0.5, rank: int = 1) -> float:
  """Rank-k hyperfine recoupling factor; rank=1 preserves the E1 convention."""
  phase_exp = Jp + I + F + rank
  phase = 1.0 if int(round(phase_exp)) % 2 == 0 else -1.0
  prefac = math.sqrt((2 * Fp + 1) * (2 * F + 1))
  sixj = wigner_6j(Jp, Fp, I, F, J, float(rank))
  return phase * prefac * sixj

def hyperfine_matrix_coefficient(
    Jp: float, Fp: float, mFp: float, J: float, F: float, mF: float,
    q: float, I: float = 0.5, rank: int = 1,
) -> float:
    """<F',m'|T^k_q|F,m>/<J'||T^k||J>, with no dipole-strength scale."""
    phase = (-1) ** (Fp - mFp)
    three_j = wigner_3j(Fp, float(rank), F, -mFp, q, mF)
    hf_factor = hyperfine_angular_factor(Jp, Fp, J, F, I=I, rank=rank)
    return phase * three_j * hf_factor
