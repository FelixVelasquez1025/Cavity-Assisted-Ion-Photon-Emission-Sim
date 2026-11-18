"""Free-space atomic inputs; complete provenance: docs/spam-physics-notes.md.

A denotes 3D[3/2]1/2, not a hyperfine constant. Frequencies here are Hz.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qutip import Qobj, basis
from scipy.constants import hbar, physical_constants

from caipe_sim import yb171_simulator as angular

DIMENSION = 20
I = 0.5  # Olmschenk et al., PRA 76, 052314 (2007), Sec. I.
MU_B_OVER_HBAR = physical_constants["Bohr magneton"][0] / hbar
# Signed, in Bohr-magneton units: arXiv:2509.04416, Eqs. 16–19.
G_I = angular.G_I
SOURCES = {
    "S_hfs": "https://doi.org/10.1109/58.585119",
    "P_hfs": "https://doi.org/10.1103/PhysRevA.49.3351",
    "D_A_hfs_branch": "https://doi.org/10.1103/PhysRevA.76.052314",
    "P_lifetime": "https://doi.org/10.1103/PhysRevA.80.022502",
    "D_lifetime": "https://doi.org/10.1103/PhysRevA.61.022507",
    "A_lifetime": "https://doi.org/10.1088/0953-4075/26/20/003",
    "A_branch": "https://doi.org/10.1103/PhysRevA.110.043105",  # Sec. II.
    "A_gJ": "https://physics.nist.gov/asd",  # ASD v5.12, accessed 2026-11-18.
}
# UNVERIFIED: extra precision of the adopted D splitting. Source gives 0.86(2) GHz.
UNVERIFIED = ("D splitting precision: 0.86375 GHz adopted; source reports 0.86(2) GHz",)


@dataclass(frozen=True)
class Manifold:
    J: float
    Fs: tuple[int, ...]
    # Signed E(F_high)-E(F_low), not A/h; A is negative for auxiliary A.
    splitting_hz: float
    lifetime_s: float | None
    g_j: float


# S/P/D electronic g factors reuse the validated weak-field LS approximation
# (Foot §§5.5, 6.3; magnetic constants arXiv:2509.04416). Not measured g_J.
def _g_j(L, J):
    return angular.lande_g_factors(1, I, J, L, .5)[0]


MANIFOLDS = {
    "S": Manifold(.5, (0, 1), 12.642812118466e9, None, _g_j(0, .5)),
    "P": Manifold(.5, (0, 1), 2.105e9, 8.12e-9, _g_j(1, .5)),
    "D": Manifold(1.5, (1, 2), .86375e9, 52.7e-3, _g_j(2, 1.5)),
    # Ordering: Olmschenk Fig. 5; gJ is NIST's tabulated 1.320, NOT LS.
    "A": Manifold(.5, (0, 1), -2.2095e9, 37.7e-9, 1.320),
}
BRANCHES = {"P": {"S": 1-.00501, "D": .00501},
            "A": {"S": .982, "D": 1-.982}, "D": {"S": 1.0}}
# A branching is PRA 110, 043105, Sec. II; D->S assumes E2 exhausts decay.
STATES = tuple((name, F, m) for name, a in MANIFOLDS.items()
               for F in a.Fs for m in range(-F, F+1))
INDEX = {state: n for n, state in enumerate(STATES)}


def ket(level: str, F: int, m: int):
    return basis(DIMENSION, INDEX[level, F, m])


def g_f(level: str, F: int) -> float:
    """Foot §6.3 projection formula; nuclear g is in Bohr-magneton units."""
    if F == 0:
        return 0.0
    a = MANIFOLDS[level]
    ff, jj, ii = F*(F+1), a.J*(a.J+1), I*(I+1)
    return (a.g_j*(ff+jj-ii)+G_I*(ff+ii-jj))/(2*ff)


def hyperfine_hz(level: str, F: int) -> float:
    """Centroid-referenced hyperfine energies, E=A*K/2 (Foot Ch. 6)."""
    a = MANIFOLDS[level]
    f_low, f_high = a.Fs
    delta_k = f_high*(f_high+1)-f_low*(f_low+1)
    a_hz = 2*a.splitting_hz/delta_k
    return a_hz*(F*(F+1)-I*(I+1)-a.J*(a.J+1))/2


def internal_hamiltonian(B_gauss: float = 0.0, *, clock_quadratic=False) -> Qobj:
    """H/hbar in rad/s; independent electronic centroids are omitted.

    Optional clock-only correction shifts the two S,m=0 states by ±q B²/2.
    q=310.8 Hz/G²: Fisk 1997, also Olmschenk 2007 Sec. IV. This is not a
    complete second-order Zeeman Hamiltonian for all magnetic sublevels.
    """
    if not np.isfinite(B_gauss):
        raise ValueError("B_gauss must be finite")
    diag = []
    for level, F, m in STATES:
        value = 2*np.pi*hyperfine_hz(level, F) + g_f(level, F)*MU_B_OVER_HBAR*B_gauss*1e-4*m
        if clock_quadratic and level == "S" and m == 0:
            value += 2*np.pi*310.8*B_gauss**2*(.5 if F == 1 else -.5)
        diag.append(value)
    return Qobj(np.diag(diag))


@lru_cache(maxsize=None)
def coefficient(lower: tuple, upper: tuple, rank: int = 1) -> float:
    """Racah-normalized matrix element; reuse validated E1 recoupling.

    rank=2 is needed only for spontaneous D->S E2 decay. No E2 reduced
    element is inferred; lifetimes set total rates independently.
    """
    l, F, m = lower
    u, Fp, mp = upper
    J, Jp = MANIFOLDS[l].J, MANIFOLDS[u].J
    q = mp-m
    if abs(q) > rank or not abs(F-Fp) <= rank <= F+Fp:
        return 0.0
    if rank == 1:
        hf = angular.hyperfine_angular_factor(Jp, Fp, J, F, I)
    elif rank == 2:
        # General rank-k recoupling, Foot Ch. 6 / Steck Sec. 3.3.
        hf = (-1)**int(Jp+I+F+rank)*np.sqrt((2*Fp+1)*(2*F+1))*angular.wigner_6j(Jp, Fp, I, F, J, rank)
    else:
        raise ValueError("Only E1 and E2 are supported")
    return (-1)**(Fp-mp)*angular.wigner_3j(Fp, rank, F, -mp, q, m)*hf


@lru_cache(maxsize=None)
def decay_channels() -> tuple:
    """(upper, lower, physical rate s^-1); normalize each upper/branch.

    Independent resolved-sublevel jumps implement the secular vacuum bath.
    No cavity, phenomenological linewidth, or legacy sqrt(2 gamma) factor.
    """
    channels = []
    for upper in STATES:
        u = upper[0]
        if u not in BRANCHES:
            continue
        for lower_level, branching in BRANCHES[u].items():
            rank = 2 if u == "D" else 1
            allowed = [(lower, abs(coefficient(lower, upper, rank))**2)
                       for lower in STATES if lower[0] == lower_level]
            norm = sum(weight for _, weight in allowed)
            if norm <= 0:
                raise ValueError(f"No allowed decay for {upper} -> {lower_level}")
            for lower, weight in allowed:
                if weight:
                    channels.append((upper, lower, branching*weight/norm/MANIFOLDS[u].lifetime_s))
    return tuple(channels)


def collapse_operators():
    return [np.sqrt(rate)*ket(*lower)*ket(*upper).dag()
            for upper, lower, rate in decay_channels()]
