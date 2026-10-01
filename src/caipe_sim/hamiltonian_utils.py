"""
Utility functions and constants for constructing the Yb171⁺ Hamiltonian.

This module collects the physics-layer helpers that were previously embedded
in ``examples/rabi_flopping_Yb171.py``
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np
from qutip import Qobj, basis, destroy, qeye, tensor
from scipy import constants as consts
from sympy import S as _S
from sympy.physics.wigner import wigner_3j as _sympy_wigner_3j
from sympy.physics.wigner import wigner_6j as _sympy_wigner_6j

from .Yb171 import Yb171

# ---------------------------------------------------------------------------
# Constants and level metadata
# ---------------------------------------------------------------------------

HBAR = consts.hbar
MU_B = consts.physical_constants["Bohr magneton"][0]
I_NUCLEAR = 0.5  # Yb-171 nuclear spin

# Hyperfine magnetic-dipole constants (A) in GHz.
HFS_A_GHZ = {
    "S": 12.642812118466,  # plus quadratic Zeeman term noted below
    "P": 2.105,
    "D": 0.86372,
}
# Quadratic Zeeman (Hz · G^-2) for S1/2
QUAD_ZEEMAN_S_HZ_PER_G2 = 310.6

# Canonical 16-state hyperfine basis ordering (increasing manifold, then F, then mF):
# S1/2: F=0 (mF=0), F=1 (mF=-1,0,+1)
# P1/2: F=0 (mF=0), F=1 (mF=-1,0,+1)
# D3/2: F=1 (mF=-1,0,+1), F=2 (mF=-2,-1,0,+1,+2)
HF_BASIS_ORDER: List[Tuple[str, float, float]] = [
    ("S", 0.0, 0.0),
    ("S", 1.0, -1.0),
    ("S", 1.0, 0.0),
    ("S", 1.0, +1.0),
    ("P", 0.0, 0.0),
    ("P", 1.0, -1.0),
    ("P", 1.0, 0.0),
    ("P", 1.0, +1.0),
    ("D", 1.0, -1.0),
    ("D", 1.0, 0.0),
    ("D", 1.0, +1.0),
    ("D", 2.0, -2.0),
    ("D", 2.0, -1.0),
    ("D", 2.0, 0.0),
    ("D", 2.0, +1.0),
    ("D", 2.0, +2.0),
]
HF_BASIS_DIM = len(HF_BASIS_ORDER)
HF_INDEX: Dict[Tuple[str, float, float], int] = {state: i for i, state in enumerate(HF_BASIS_ORDER)}


def _g_factor_for_level(level) -> float:
    """Fetch the Landé g-factor for a given Yb171 level."""
    for level_data in Yb171.level_data:
        if level_data.level == level:
            if level_data.g_J is None:
                raise ValueError(f"No Landé g-factor defined for {level}.")
            return float(level_data.g_J)
    raise ValueError(f"Level {level} not found in Yb171 data.")


g_s = _g_factor_for_level(Yb171.S12)
g_d = _g_factor_for_level(Yb171.D32)
g_p = _g_factor_for_level(Yb171.P12)

# Reference linewidths (2π × MHz or mHz) converted to rad/s.
LINEWIDTH_SP = 2 * np.pi * 21.57e6  # S1/2 ↔ P1/2
LINEWIDTH_DP = 2 * np.pi * 1.482e6  # D3/2 ↔ P1/2 (852 nm)
LINEWIDTH_SD = 2 * np.pi * 136e3  # S1/2 ↔ D3/2 (136 mHz → 136 kHz)

# ---------------------------------------------------------------------------
# Angular momentum helpers
# ---------------------------------------------------------------------------


def wigner_3j(j1: float, j2: float, j3: float, m1: float, m2: float, m3: float) -> float:
    """Numeric wrapper around SymPy wigner_3j returning a real float."""
    val = _sympy_wigner_3j(_S(j1), _S(j2), _S(j3), _S(m1), _S(m2), _S(m3))
    cval = complex(val.evalf())
    return float(cval.real)


def _clebsch_gordan(j1: float, m1: float, j2: float, m2: float, J: float, M: float) -> float:
    """Clebsch-Gordan coefficient using Wigner-3j."""
    exp2 = int(round(2.0 * (-j1 - +j2 - M)))
    phase = 1.0 if (exp2 % 2 == 0) else -1.0
    w3 = wigner_3j(j1, j2, J, m1, m2, -M)
    w3_real = float(complex(w3).real)
    val = phase * math.sqrt(2.0 * J + 1.0) * w3_real
    return float(val)


def wigner_6j(j1: float, j2: float, j3: float, j4: float, j5: float, j6: float) -> float:
    """Numeric wrapper around SymPy wigner_6j returning a real float."""
    val = _sympy_wigner_6j(_S(j1), _S(j2), _S(j3), _S(j4), _S(j5), _S(j6))
    cval = complex(val.evalf())
    return float(cval.real)


def hyperfine_angular_factor(Jp: float, Fp: float, J: float, F: float, I: float = 0.5) -> float:
    """Angular factor for hyperfine-coupled fermionic isotopes (Eq. 2 in notes)."""
    phase_exp = Jp + I + F + 1
    phase = 1.0 if int(round(2 * phase_exp)) % 2 == 0 else -1.0
    prefac = math.sqrt((2 * Fp + 1) * (2 * F + 1))
    sixj = wigner_6j(Jp, Fp, I, F, J, 1.0)
    return phase * prefac * sixj


def hyperfine_energy(
    J: float,
    F: float,
    A_GHz: float,
    B_field_G: float = 5.0,
    quadratic_Hz_per_G2: float = 0.0,
) -> float:
    """Hyperfine shift (rad/s) using ΔE = 0.5 * A * K, plus optional quadratic term."""
    I = I_NUCLEAR
    K = F * (F + 1) - J * (J + 1) - I * (I + 1)
    A_rad = A_GHz * 1e9 * 2 * np.pi  # GHz → Hz → rad/s
    delta_E = 0.5 * A_rad * K
    quad = (quadratic_Hz_per_G2 * (B_field_G**2)) * 2 * np.pi if quadratic_Hz_per_G2 != 0.0 else 0.0
    return delta_E + quad


def e1_matrix_element(
    transition: str,
    q: float,
    bosonic: bool,
    Jp: float,
    J: float,
    Mp: float | None = None,
    M: float | None = None,
    Fp: float | None = None,
    F: float | None = None,
    mFp: float | None = None,
    mF: float | None = None,
    I: float = 0.5,
) -> float:
    """Full E1 matrix element for Yb."""
    reduced = REDUCED_E1_ELEMENTS[transition]
    if bosonic:
        if Mp is None or M is None:
            raise ValueError("Bosonic case requires Mp and M.")
        phase = (-1) ** (Jp - Mp)
        three_j = wigner_3j(Jp, 1.0, J, -Mp, q, M)
        return phase * three_j * reduced
    else:
        if None in (Fp, F, mFp, mF):
            raise ValueError("Fermionic case requires Fp, F, mFp, mF.")
        phase = (-1) ** (Fp - mFp)
        three_j = wigner_3j(Fp, 1.0, F, -mFp, q, mF)
        hf_factor = hyperfine_angular_factor(Jp, Fp, J, F, I=I)
        return phase * three_j * hf_factor * reduced


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

Level = str  # 'S', 'P', or 'D'


def hf_state_index(level: str, F: float, mF: float) -> int:
    """Return index in the 16-state hyperfine basis."""
    try:
        return HF_INDEX[(level, F, mF)]
    except KeyError as exc:
        raise ValueError(f"Unknown hyperfine state ({level}, F={F}, mF={mF})") from exc


def hf_states_for_level(level: str) -> List[Tuple[str, float, float]]:
    """List all (level, F, mF) states for a given level label."""
    return [st for st in HF_BASIS_ORDER if st[0] == level]


def projector_for_level_hf(level: str, dim: int = HF_BASIS_DIM) -> Qobj:
    """Projector onto all hyperfine states of a given fine-structure level."""
    P = Qobj(np.zeros((dim, dim), dtype=complex))
    for (_, F, mF) in hf_states_for_level(level):
        idx = hf_state_index(level, F, mF)
        ket = basis(dim, idx)
        P += ket * ket.dag()
    return P


# Local 8-state basis ordering (only used for the constant-Ω fallback):
# 0: S1/2, m=-1/2; 1: S1/2, m=+1/2; 2: P1/2, m=-1/2; 3: P1/2, m=+1/2;
# 4: D3/2, m=-3/2; 5: D3/2, m=-1/2; 6: D3/2, m=+1/2; 7: D3/2, m=+3/2
def state_index(level: Level, m: float) -> int:
    """Map a (level, m) pair to its index in the 8-state local basis."""
    if level == "S":
        return 0 if m == -0.5 else 1
    if level == "P":
        return 2 if m == -0.5 else 3
    if level == "D":
        return {-1.5: 4, -0.5: 5, 0.5: 6, 1.5: 7}[m]
    raise ValueError(f"Unknown state ({level}, {m})")


def _mj_values(J: float) -> List[float]:
    """Return the list of m_j values for a given total angular momentum J."""
    twoJ = int(round(2 * J))
    return [(-twoJ + k) / 2.0 for k in range(0, 2 * twoJ + 1)]


def _e1_cgc(j_lower: float, m_lower: float, j_upper: float, m_upper: float) -> float:
    """Electric-dipole Clebsch-Gordan coefficient for a single transition."""
    q = m_upper - m_lower
    if q not in (-1.0, 0.0, 1.0):
        return 0.0
    return _clebsch_gordan(j_lower, m_lower, 1.0, q, j_upper, m_upper)


# ---------------------------------------------------------------------------
# Reduced matrix elements (Eq. 54)
# ---------------------------------------------------------------------------

_EA0 = consts.e * consts.physical_constants["Bohr radius"][0]

# Diagram-derived lifetimes, branching ratios, and angular frequencies
TAU_P12 = 8.07e-9  # 6p 2P1/2
TAU_D32 = 52.7e-3  # 5d 2D3/2
BR_P12_TO_S12 = 0.995
BR_P12_TO_D32 = 0.005
OMEGA_369 = 2 * np.pi * consts.c / 369.5262e-9  # P1/2 <-> S1/2
OMEGA_2400 = 2 * np.pi * consts.c / 2.400e-6  # P1/2 <-> D3/2
OMEGA_435 = 2 * np.pi * consts.c / 435.5e-9  # D3/2 <-> S1/2


def _rme_eq54(beta: float, omega: float, tau: float, Je: float) -> float:
    """Reduced matrix element via Eq. (54)."""
    pref = (2 * Je + 1)
    num = 3 * np.pi * consts.epsilon_0 * consts.hbar * (consts.c**3) * beta * pref
    denom = (omega**3) * tau
    return math.sqrt(num / denom)


REDUCED_E1_ELEMENTS = {
    "2S12_to_2P12": _rme_eq54(BR_P12_TO_S12, OMEGA_369, TAU_P12, Je=0.5) / _EA0,
    "2D32_to_2P12": _rme_eq54(BR_P12_TO_D32, OMEGA_2400, TAU_P12, Je=0.5) / _EA0,
    "2S12_to_2D32": _rme_eq54(1.0, OMEGA_435, TAU_D32, Je=1.5) / _EA0,
}

# ---------------------------------------------------------------------------
# Clebsch-Gordan tables and E1 coefficients
# ---------------------------------------------------------------------------


def cgc_lists_general() -> Dict[str, Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]]]:
    """Build polarization-partitioned CGC tables for SP and DP manifolds."""
    J_S = float(Yb171.S12.J)
    J_P = float(Yb171.P12.J)
    J_D = float(Yb171.D32.J)

    def build_pair(lower_label: Level, j_lower: float, upper_label: Level, j_upper: float):
        out: Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]] = {+1: [], 0: [], -1: []}
        for m_l in _mj_values(j_lower):
            for q in (-1, 0, +1):
                m_u = m_l + q
                if m_u < -j_upper or m_u > j_upper:
                    continue
                cgc = _e1_cgc(j_lower, m_l, j_upper, m_u)
                if cgc == 0.0:
                    continue
                out[q].append(((lower_label, m_l), (upper_label, m_u), float(cgc)))
        return out

    SP = build_pair("S", J_S, "P", J_P)
    DP = build_pair("D", J_D, "P", J_P)
    return {"SP": SP, "DP": DP}


CGC_DATA = cgc_lists_general()


def precompute_e1_coeffs(is_bosonic: bool) -> Dict[str, Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]]]:
    """Precompute E1 angular factors for either bosonic (m_J) or fermionic (m_F) use."""
    coeffs: Dict[str, Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]]] = {
        "SP": {+1: [], 0: [], -1: []},
        "DP": {+1: [], 0: [], -1: []},
    }
    reduced_map = {"SP": "2S12_to_2P12", "DP": "2D32_to_2P12"}
    for pair, pol_map in CGC_DATA.items():
        trans_key = reduced_map.get(pair)
        if trans_key is None:
            continue
        for pol, entries in pol_map.items():
            for (lower, upper, _cgc) in entries:
                J_lower = float(Yb171.S12.J if lower[0] == "S" else Yb171.D32.J)
                J_upper = float(Yb171.P12.J)
                if is_bosonic:
                    g_e1 = e1_matrix_element(trans_key, pol, True, J_upper, J_lower, Mp=upper[1], M=lower[1])
                else:
                    I_val = 0.5
                    F_lower = J_lower + I_val
                    F_upper = J_upper + I_val
                    mI = I_val
                    g_e1 = e1_matrix_element(
                        trans_key,
                        pol,
                        False,
                        J_upper,
                        J_lower,
                        Fp=F_upper,
                        F=F_lower,
                        mFp=upper[1] + mI,
                        mF=lower[1] + mI,
                        I=I_val,
                    )
                coeffs[pair][pol].append((lower, upper, g_e1))
    return coeffs


def precompute_e1_coeffs_hf() -> Dict[str, Dict[int, List[Tuple[Tuple[str, float, float], Tuple[str, float, float], float]]]]:
    """Hyperfine E1 coefficients across full F,mF manifold."""
    coeffs: Dict[str, Dict[int, List[Tuple[Tuple[str, float, float], Tuple[str, float, float], float]]]] = {
        "SP": {+1: [], 0: [], -1: []},
        "DP": {+1: [], 0: [], -1: []},
    }
    reduced_map = {"SP": "2S12_to_2P12", "DP": "2D32_to_2P12"}
    F_levels = {"S": [0.0, 1.0], "P": [0.0, 1.0], "D": [1.0, 2.0]}
    J_map = {"S": float(Yb171.S12.J), "P": float(Yb171.P12.J), "D": float(Yb171.D32.J)}
    for pair, trans_key in reduced_map.items():
        lower_level, upper_level = ("S", "P") if pair == "SP" else ("D", "P")
        J_lower = J_map[lower_level]
        J_upper = J_map[upper_level]
        for F_lower in F_levels[lower_level]:
            for mF_lower in [m for m in np.arange(-F_lower, F_lower + 1, 1.0)]:
                for q in (-1, 0, +1):
                    mF_upper = mF_lower + q
                    for F_upper in F_levels[upper_level]:
                        if mF_upper < -F_upper or mF_upper > F_upper:
                            continue
                        try:
                            coeff = e1_matrix_element(
                                trans_key,
                                q,
                                bosonic=False,  # hyperfine (fermionic) path
                                Jp=J_upper,
                                J=J_lower,
                                Fp=F_upper,
                                F=F_lower,
                                mFp=mF_upper,
                                mF=mF_lower,
                                I=I_NUCLEAR,
                            )
                        except Exception:
                            coeff = 0.0
                        # Keep zero-valued elements for complete bookkeeping.
                        coeffs[pair][q].append(
                            ((lower_level, F_lower, mF_lower), (upper_level, F_upper, mF_upper), coeff)
                        )
    return coeffs


E1_DATA_HF = precompute_e1_coeffs_hf()

# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------

_q_to_label = {+1: "left", 0: "pi", -1: "right"}


def delta_mj(m_lower: float, m_upper: float) -> int:
    """Return Δm_j = m_upper - m_lower in {-1,0,+1}."""
    return int(round(m_upper - m_lower))


def _index_from_local(level: Level, m: float) -> int:
    return state_index(level, m)


def _build_systems(
    pair: str,
    Omega_value: float,
    index_of: Callable[[Level, float], int],
    dim: int,
    e1_coeffs: Dict[str, Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]]] | None = None,
) -> Dict[str, Qobj]:
    """Accumulate three Hamiltonians (left, pi, right) for a pair (SP/DP)."""
    systems = {name: Qobj(np.zeros((dim, dim), dtype=complex)) for name in ("left", "pi", "right")}
    data = e1_coeffs[pair] if e1_coeffs is not None else CGC_DATA[pair]
    for q_group in (-1, 0, +1):
        for (lower, upper, coeff) in data.get(q_group, []):
            m_l, m_u = lower[1], upper[1]
            q = delta_mj(m_l, m_u)
            if q not in (-1, 0, +1):
                continue
            Omega = float(Omega_value)
            li = index_of(lower[0], m_l)
            ui = index_of(upper[0], m_u)
            ket_l = basis(dim, li)
            ket_u = basis(dim, ui)
            sigma = ket_u * ket_l.dag()
            label = _q_to_label[q]
            systems[label] = systems[label] + 0.5 * HBAR * Omega * coeff * (sigma + sigma.dag())
    return systems


def _build_systems_hf(
    pair: str,
    Omega_value: float,
    dim: int,
    e1_coeffs_hf: Dict[str, Dict[int, List[Tuple[Tuple[str, float, float], Tuple[str, float, float], float]]]] | None = None,
) -> Dict[str, Qobj]:
    """Hyperfine version: accumulate three Hamiltonians (left, pi, right) over F,mF."""
    systems = {name: Qobj(np.zeros((dim, dim), dtype=complex)) for name in ("left", "pi", "right")}
    data = e1_coeffs_hf[pair] if e1_coeffs_hf is not None else E1_DATA_HF[pair]
    for q_group in (-1, 0, +1):
        for (lower, upper, coeff) in data.get(q_group, []):
            m_l, m_u = lower[2], upper[2]
            q = int(round(m_u - m_l))
            if q not in (-1, 0, +1):
                continue
            Omega = float(Omega_value)
            li = hf_state_index(*lower)
            ui = hf_state_index(*upper)
            ket_l = basis(dim, li)
            ket_u = basis(dim, ui)
            sigma = ket_u * ket_l.dag()
            label = _q_to_label[q]
            systems[label] = systems[label] + 0.5 * HBAR * Omega * coeff * (sigma + sigma.dag())
    return systems


def transition_sigma_components(pair: str, dim: int = 8) -> Dict[int, Qobj]:
    """Non-Hermitian atomic operators S_q = Σ_{Δm=q} CGC |upper><lower|."""
    zero = Qobj(np.zeros((dim, dim), dtype=complex))
    comps: Dict[int, Qobj] = {-1: zero, 0: zero, +1: zero}
    data = CGC_DATA[pair]
    for q_group in (-1, 0, +1):
        for (lower, upper, cgc) in data.get(q_group, []):
            q = delta_mj(lower[1], upper[1])
            if q not in (-1, 0, +1):
                continue
            li = state_index(*lower)
            ui = state_index(*upper)
            ket_l = basis(dim, li)
            ket_u = basis(dim, ui)
            term = cgc * (ket_u * ket_l.dag())
            comps[q] = comps[q] + term if comps[q] is not zero else term
    for k in (-1, 0, +1):
        if not isinstance(comps[k], Qobj):
            comps[k] = Qobj(np.zeros((dim, dim), dtype=complex))
    return comps


def transition_sigma_components_hf(pair: str, dim: int = HF_BASIS_DIM) -> Dict[int, Qobj]:
    """Hyperfine non-Hermitian operators S_q = Σ CGC |upper><lower| over F,mF."""
    zero = Qobj(np.zeros((dim, dim), dtype=complex))
    comps: Dict[int, Qobj] = {-1: zero, 0: zero, +1: zero}
    data = E1_DATA_HF[pair]
    for q_group in (-1, 0, +1):
        for (lower, upper, coeff) in data.get(q_group, []):
            q = int(round(upper[2] - lower[2]))
            if q not in (-1, 0, +1):
                continue
            li = hf_state_index(*lower)
            ui = hf_state_index(*upper)
            ket_l = basis(dim, li)
            ket_u = basis(dim, ui)
            term = coeff * (ket_u * ket_l.dag())
            comps[q] = comps[q] + term if comps[q] is not zero else term
    for k in (-1, 0, +1):
        if not isinstance(comps[k], Qobj):
            comps[k] = Qobj(np.zeros((dim, dim), dtype=complex))
    return comps


def cavity_sigma_DP_components(dim: int = 8) -> Dict[int, Qobj]:
    """Non-Hermitian atomic operators for the cavity D↔P coupling."""
    S = transition_sigma_components("DP", dim=dim)
    return {+1: S.get(+1, Qobj(np.zeros((dim, dim), dtype=complex))), -1: S.get(-1, Qobj(np.zeros((dim, dim), dtype=complex)))}


def precompute_lowering_tensors(N_a: int, N_b: int, dim_atom: int) -> Dict[str, List[Tuple[Qobj, float]]]:
    """Precompute and tensor atomic lowering operators into cavity⊗atom space."""
    Ia = qeye(N_a)
    Ib = qeye(N_b)
    tensors: Dict[str, List[Tuple[Qobj, float]]] = {}
    for pair in ("SP", "DP"):
        ops: List[Tuple[Qobj, float]] = []
        for pol_entries in CGC_DATA[pair].values():
            for (lower, upper, cgc) in pol_entries:
                li = state_index(*lower)
                ui = state_index(*upper)
                ket_l = basis(dim_atom, li)
                ket_u = basis(dim_atom, ui)
                atom_op = ket_l * ket_u.dag()
                ops.append((tensor(Ia, Ib, atom_op), abs(cgc)))
        tensors[pair] = ops
    return tensors


def precompute_lowering_tensors_hf(N_a: int, N_b: int, dim_atom: int = HF_BASIS_DIM) -> Dict[str, List[Tuple[Qobj, float]]]:
    Ia = qeye(N_a)
    Ib = qeye(N_b)
    tensors: Dict[str, List[Tuple[Qobj, float]]] = {}
    for pair in ("SP", "DP"):
        ops: List[Tuple[Qobj, float]] = []
        for pol_entries in E1_DATA_HF[pair].values():
            for (lower, upper, coeff) in pol_entries:
                li = hf_state_index(*lower)
                ui = hf_state_index(*upper)
                ket_l = basis(dim_atom, li)
                ket_u = basis(dim_atom, ui)
                atom_op = ket_l * ket_u.dag()
                ops.append((tensor(Ia, Ib, atom_op), abs(coeff)))
        tensors[pair] = ops
    return tensors


def H_cavity_DP_modes(g: float, N_a: int, N_b: int, dim_atom: int = 8) -> Dict[str, Qobj]:
    """Eq. 2.4.14-2.4.15: build H_a and H_b for two cavity polarizations."""
    I_a = qeye(N_a)
    I_b = qeye(N_b)
    a = destroy(N_a)
    b = destroy(N_b)

    comps = cavity_sigma_DP_components(dim=dim_atom)
    S_p = comps[+1]
    S_m = comps[-1]

    H_a = HBAR * g * (tensor(a, I_b, S_p) + tensor(a.dag(), I_b, S_p.dag()))
    H_b = HBAR * g * (tensor(I_a, b, S_m) + tensor(I_a, b.dag(), S_m.dag()))
    return {"left": H_a, "right": H_b}


def H_cavity_DP_modes_hf(g: float, N_a: int, N_b: int, dim_atom: int = HF_BASIS_DIM) -> Dict[str, Qobj]:
    """Hyperfine version of Eq. 2.4.14-2.4.15."""
    I_a = qeye(N_a)
    I_b = qeye(N_b)
    a = destroy(N_a)
    b = destroy(N_b)

    comps = transition_sigma_components_hf("DP", dim=dim_atom)
    S_p = comps[+1]
    S_m = comps[-1]

    H_a = HBAR * g * (tensor(a, I_b, S_p) + tensor(a.dag(), I_b, S_p.dag()))
    H_b = HBAR * g * (tensor(I_a, b, S_m) + tensor(I_a, b.dag(), S_m.dag()))
    return {"left": H_a, "right": H_b}


def rabi_frequencies(E0_driv: float, E0_rep: float) -> Tuple[float, float]:
    """Compute Ω_driv and Ω_rep (rad/s) for Eqs. 2.4.12 and 2.4.13."""
    return float(E0_driv) / (2 * HBAR), float(E0_rep) / (2 * HBAR)


def H_laser_systems_from_E0(E0_driv: float, E0_rep: float, dim: int = 8, e1_coeffs=None) -> Dict[str, Qobj]:
    """Laser Hamiltonians using field amplitudes E0 (rad/s scale)."""
    Omega_driv, Omega_rep = rabi_frequencies(E0_driv, E0_rep)
    Hsp = _build_systems("SP", float(Omega_driv), _index_from_local, dim, e1_coeffs=e1_coeffs)
    Hdp = _build_systems("DP", float(Omega_rep), _index_from_local, dim, e1_coeffs=e1_coeffs)
    return {"left": Hsp.get("left") + Hdp.get("left"), "pi": Hsp.get("pi") + Hdp.get("pi"), "right": Hsp.get("right") + Hdp.get("right")}


def H_laser_systems_from_E0_hf(E0_driv: float, E0_rep: float, dim: int = HF_BASIS_DIM, e1_coeffs_hf=None) -> Dict[str, Qobj]:
    """Hyperfine version of the laser Hamiltonian builder."""
    Omega_driv, Omega_rep = rabi_frequencies(E0_driv, E0_rep)
    Hsp = _build_systems_hf("SP", float(Omega_driv), dim, e1_coeffs_hf=e1_coeffs_hf)
    Hdp = _build_systems_hf("DP", float(Omega_rep), dim, e1_coeffs_hf=e1_coeffs_hf)
    return {"left": Hsp.get("left") + Hdp.get("left"), "pi": Hsp.get("pi") + Hdp.get("pi"), "right": Hsp.get("right") + Hdp.get("right")}


def H_interaction_with_cavity(H_laser_atom: Dict[str, Qobj], g: float, N_a: int, N_b: int) -> Dict[str, Qobj]:
    """Lift atomic laser Hamiltonians to 2-mode cavity ⊗ atom space and add H_a/H_b."""
    sample = next(iter(H_laser_atom.values()))
    dim_atom = sample.shape[0]
    I_a = qeye(N_a)
    I_b = qeye(N_b)

    Hc = H_cavity_DP_modes(g=g, N_a=N_a, N_b=N_b, dim_atom=dim_atom)

    H_left = tensor(I_a, I_b, H_laser_atom.get("left")) + Hc["left"]
    H_right = tensor(I_a, I_b, H_laser_atom.get("right")) + Hc["right"]
    H_pi = tensor(I_a, I_b, H_laser_atom.get("pi"))

    return {"left": H_left, "pi": H_pi, "right": H_right}


def H_interaction_with_cavity_hf(H_laser_atom: Dict[str, Qobj], g: float, N_a: int, N_b: int) -> Dict[str, Qobj]:
    """Hyperfine version of the cavity interaction builder."""
    sample = next(iter(H_laser_atom.values()))
    dim_atom = sample.shape[0]
    I_a = qeye(N_a)
    I_b = qeye(N_b)

    Hc = H_cavity_DP_modes_hf(g=g, N_a=N_a, N_b=N_b, dim_atom=dim_atom)

    H_left = tensor(I_a, I_b, H_laser_atom.get("left")) + Hc["left"]
    H_right = tensor(I_a, I_b, H_laser_atom.get("right")) + Hc["right"]
    H_pi = tensor(I_a, I_b, H_laser_atom.get("pi"))

    return {"left": H_left, "pi": H_pi, "right": H_right}


def H0_cavity(delta_c: float, delta_rep: float, N_a: int, N_b: int, dim_atom: int) -> Qobj:
    """Bare two-mode cavity Hamiltonian (Eq. 2.4.7)."""
    coeff = (delta_c - delta_rep)
    I_atom = qeye(dim_atom)
    a = destroy(N_a)
    b = destroy(N_b)
    Hc = coeff * HBAR * (tensor(a.dag() * a, qeye(N_b), I_atom) + tensor(qeye(N_a), b.dag() * b, I_atom))
    return Hc


def eq_2_4_5_H_S(delta_drive: float, B_T: float, g_S: float = g_s, dim: int = 8) -> Qobj:
    """S-manifold diagonal Hamiltonian in the rotating frame."""
    H = Qobj(np.zeros((dim, dim), dtype=complex))
    for m in (-0.5, +0.5):
        idx = state_index("S", m)
        ket = basis(dim, idx)
        E = delta_drive + g_S * MU_B * B_T * m
        H += E * (ket * ket.dag())
    return H


def eq_2_4_6_H_D(delta_rep: float, B_T: float, g_D: float = g_d, dim: int = 8) -> Qobj:
    """D-manifold diagonal Hamiltonian in the rotating frame."""
    H = Qobj(np.zeros((dim, dim), dtype=complex))
    for m in (-1.5, -0.5, +0.5, +1.5):
        idx = state_index("D", m)
        ket = basis(dim, idx)
        E = delta_rep + g_D * MU_B * B_T * m
        H += E * (ket * ket.dag())
    return H


def eq_2_4_5_H_S_hf(delta_drive: float, B_T: float, g_S: float = g_s, dim: int = HF_BASIS_DIM) -> Qobj:
    """Hyperfine S-manifold diagonal Hamiltonian."""
    H = Qobj(np.zeros((dim, dim), dtype=complex))
    J = float(Yb171.S12.J)
    A = HFS_A_GHZ["S"]
    for (lvl, F, mF) in HF_BASIS_ORDER:
        if lvl != "S":
            continue
        idx = hf_state_index(lvl, F, mF)
        ket = basis(dim, idx)
        E_hf = hyperfine_energy(J, F, A, B_field_G=B_T * 1e4, quadratic_Hz_per_G2=QUAD_ZEEMAN_S_HZ_PER_G2)
        E = delta_drive + g_S * MU_B * B_T * mF + E_hf
        H += E * (ket * ket.dag())
    return H


def eq_2_4_6_H_D_hf(delta_rep: float, B_T: float, g_D: float = g_d, dim: int = HF_BASIS_DIM) -> Qobj:
    """Hyperfine D-manifold diagonal Hamiltonian."""
    H = Qobj(np.zeros((dim, dim), dtype=complex))
    J = float(Yb171.D32.J)
    A = HFS_A_GHZ["D"]
    for (lvl, F, mF) in HF_BASIS_ORDER:
        if lvl != "D":
            continue
        idx = hf_state_index(lvl, F, mF)
        ket = basis(dim, idx)
        E_hf = hyperfine_energy(J, F, A, B_field_G=B_T * 1e4, quadratic_Hz_per_G2=0.0)
        E = delta_rep + g_D * MU_B * B_T * mF + E_hf
        H += E * (ket * ket.dag())
    return H


def add_bare_terms_to_systems(
    H_systems: Dict[str, Qobj],
    H_S: Qobj,
    H_D: Qobj,
    delta_c: float | None = None,
    delta_rep: float | None = None,
    N_a: int | None = None,
    N_b: int | None = None,
) -> Dict[str, Qobj]:
    """Add bare-ion terms and bare-cavity terms to each system."""
    out: Dict[str, Qobj] = {}
    add_cavity = (None not in (delta_c, delta_rep, N_a, N_b))
    if add_cavity:
        dim_atom = H_S.shape[0]
        Ia = qeye(int(N_a))
        Ib = qeye(int(N_b))
        H_bare_atom = tensor(Ia, Ib, H_S + H_D)
        H_cav = H0_cavity(float(delta_c), float(delta_rep), int(N_a), int(N_b), dim_atom)
        for key, H in H_systems.items():
            out[key] = H + H_bare_atom + H_cav
    else:
        for key, H in H_systems.items():
            out[key] = H + H_S + H_D
    return out


def projector_for_indices(indices: List[int], dim: int = 8) -> Qobj:
    """Projector onto a set of bare |level,m> indices in the 8-state basis."""
    P = Qobj(np.zeros((dim, dim), dtype=complex))
    for i in indices:
        ket = basis(dim, i)
        P += ket * ket.dag()
    return P


__all__ = [
    "HBAR",
    "MU_B",
    "I_NUCLEAR",
    "HFS_A_GHZ",
    "QUAD_ZEEMAN_S_HZ_PER_G2",
    "HF_BASIS_ORDER",
    "HF_BASIS_DIM",
    "HF_INDEX",
    "g_s",
    "g_d",
    "g_p",
    "LINEWIDTH_SP",
    "LINEWIDTH_DP",
    "LINEWIDTH_SD",
    "wigner_3j",
    "wigner_6j",
    "_clebsch_gordan",
    "hyperfine_angular_factor",
    "hyperfine_energy",
    "e1_matrix_element",
    "hf_state_index",
    "hf_states_for_level",
    "projector_for_level_hf",
    "state_index",
    "cgc_lists_general",
    "precompute_e1_coeffs",
    "precompute_e1_coeffs_hf",
    "delta_mj",
    "_build_systems",
    "_build_systems_hf",
    "transition_sigma_components",
    "transition_sigma_components_hf",
    "cavity_sigma_DP_components",
    "precompute_lowering_tensors",
    "precompute_lowering_tensors_hf",
    "H_cavity_DP_modes",
    "H_cavity_DP_modes_hf",
    "rabi_frequencies",
    "H_laser_systems_from_E0",
    "H_laser_systems_from_E0_hf",
    "H_interaction_with_cavity",
    "H_interaction_with_cavity_hf",
    "H0_cavity",
    "eq_2_4_5_H_S",
    "eq_2_4_6_H_D",
    "eq_2_4_5_H_S_hf",
    "eq_2_4_6_H_D_hf",
    "add_bare_terms_to_systems",
    "projector_for_indices",
    "REDUCED_E1_ELEMENTS",
    "E1_DATA_HF",
    "CGC_DATA",
]
