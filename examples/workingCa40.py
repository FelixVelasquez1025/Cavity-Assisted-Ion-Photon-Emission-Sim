from __future__ import annotations

import math
from typing import Dict, List, Tuple, Callable

import numpy as np
from qutip import Qobj, basis, destroy, qeye, tensor, mesolve
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy import constants as consts

from atomic_physics.ions.ca40 import Ca40  # type: ignore



# Local 8-state basis ordering (only used for the constant-Ω fallback):
# 0: S1/2, m=-1/2; 1: S1/2, m=+1/2; 2: P1/2, m=-1/2; 3: P1/2, m=+1/2;
# 4: D3/2, m=-3/2; 5: D3/2, m=-1/2; 6: D3/2, m=+1/2; 7: D3/2, m=+3/2

Level = str  # 'S', 'P', or 'D'

# Physical constants (SI) from SciPy
HBAR = consts.hbar
MU_B = consts.physical_constants["Bohr magneton"][0]

def state_index(level: Level, m: float) -> int:
    if level == 'S':
        return 0 if m == -0.5 else 1
    if level == 'P':
        return 2 if m == -0.5 else 3
    if level == 'D':
        return { -1.5: 4, -0.5: 5, 0.5: 6, 1.5: 7 }[m]
    raise ValueError(f"Unknown state ({level}, {m})")


def cgc_lists() -> Dict[str, Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]]]:
    """Clebsch-Gordan coefficients for S↔P and D↔P transitions grouped by Δm=q."""
    SP = {
        +1: [(('S', -0.5), ('P', +0.5), math.sqrt(2.0 / 3.0))],
        0: [
            (('S', -0.5), ('P', -0.5), -math.sqrt(1.0 / 3.0)),
            (('S', +0.5), ('P', +0.5), +math.sqrt(1.0 / 3.0)),
        ],
        -1: [(('S', +0.5), ('P', -0.5), -math.sqrt(2.0 / 3.0))],
    }
    DP = {
        +1: [
            (('D', -1.5), ('P', -0.5), math.sqrt(1.0 / 2.0)),
            (('D', -0.5), ('P', +0.5), math.sqrt(1.0 / 6.0)),
        ],
        0: [
            (('D', -0.5), ('P', -0.5), -math.sqrt(1.0 / 6.0)),
            (('D', +0.5), ('P', +0.5), +math.sqrt(1.0 / 6.0)),
        ],
        -1: [
            (('D', +1.5), ('P', +0.5), -math.sqrt(1.0 / 2.0)),
            (('D', +0.5), ('P', -0.5), -math.sqrt(1.0 / 6.0)),
        ],
    }
    return {'SP': SP, 'DP': DP}


def delta_mj(m_lower: float, m_upper: float) -> int:
    """Return Δm_j = m_upper − m_lower in {-1,0,+1} (rounded to nearest int)."""
    return int(round(m_upper - m_lower))


_q_to_label = {+1: 'left', 0: 'pi', -1: 'right'}





def _index_from_local(level: Level, m: float) -> int:
    return state_index(level, m)


def _build_systems(
    pair: str,
    get_omega: Callable[[Level, float, Level, float, int], float],
    index_of: Callable[[Level, float], int],
    dim: int,
) -> Dict[str, Qobj]:
    """Core builder: accumulate three Hamiltonians (left, pi, right).

    For each CGC-listed transition A,m_A → B,m_B, compute q=Δm, fetch Ω,
    and add ½ Ω · CGC (|u><l| + h.c.) into the appropriate system.
    """
    systems = {name: Qobj(np.zeros((dim, dim), dtype=complex)) for name in ('left', 'pi', 'right')}
    data = cgc_lists()[pair]
    for q_group in (-1, 0, +1):
        for (lower, upper, cgc) in data.get(q_group, []):
            m_l, m_u = lower[1], upper[1]
            q = delta_mj(m_l, m_u)
            if q not in (-1, 0, +1):
                continue
            Omega = float(get_omega(lower[0], m_l, upper[0], m_u, q))
            if Omega == 0.0:
                continue
            li = index_of(lower[0], m_l)
            ui = index_of(upper[0], m_u)
            ket_l = basis(dim, li)
            ket_u = basis(dim, ui)
            sigma = ket_u * ket_l.dag()
            label = _q_to_label[q]
            systems[label] = systems[label] + 0.5 * Omega * cgc * (sigma + sigma.dag())
    return systems





def H_SP_systems_constant(Omega_by_q: Dict[int, float], dim: int = 8) -> Dict[str, Qobj]:
    """Fallback: constant Ω per spherical component in the local 8-state basis."""

    def get_omega(_L: Level, _mL: float, _U: Level, _mU: float, q: int) -> float:
        return float(Omega_by_q.get(q, 0.0))

    return _build_systems('SP', get_omega, _index_from_local, dim)


def H_DP_systems_constant(Omega_by_q: Dict[int, float], dim: int = 8) -> Dict[str, Qobj]:
    """Fallback: constant Ω per spherical component in the local 8-state basis."""

    def get_omega(_L: Level, _mL: float, _U: Level, _mU: float, q: int) -> float:
        return float(Omega_by_q.get(q, 0.0))

    return _build_systems('DP', get_omega, _index_from_local, dim)


def transition_sigma_components(pair: str, dim: int = 8) -> Dict[int, Qobj]:
    """Build non-Hermitian atomic operators S_q = Σ_{Δm=q} CGC |upper><lower|.

    This is a local-basis helper (uses state_index) for composing Eq. 2.4.12–2.4.15
    pieces. It does not rely on any external ion object.
    """
    zero = Qobj(np.zeros((dim, dim), dtype=complex))
    comps: Dict[int, Qobj] = {-1: zero, 0: zero, +1: zero}
    data = cgc_lists()[pair]
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


def cavity_sigma_DP_components(dim: int = 8) -> Dict[int, Qobj]:
    """Non-Hermitian atomic operators for the cavity D↔P coupling.

    Returns S_q = Σ_{Δm=q} CGC |P><D| for q ∈ {+1, -1}. (No π component.)
    """
    S = transition_sigma_components('DP', dim=dim)
    return {+1: S.get(+1, Qobj(np.zeros((dim, dim), dtype=complex))),
            -1: S.get(-1, Qobj(np.zeros((dim, dim), dtype=complex)))}


def H_cavity_DP_modes(g: float, N_a: int, N_b: int, dim_atom: int = 8) -> Dict[str, Qobj]:
    """Eq. 2.4.14-2.4.15: build H_a and H_b for two cavity polarizations.

    - a: left-circular (σ+), accumulates q=+1 terms.
    - b: right-circular (σ−), accumulates q=−1 terms.

    H_a = g ( a ⊗ S_{+1} + a† ⊗ S_{+1}† ),
    H_b = g ( b ⊗ S_{-1} + b† ⊗ S_{-1}† ).
    The total cavity coupling is H_c = H_a + H_b.
    """
    I_a = qeye(N_a)
    I_b = qeye(N_b)
    I_atom = qeye(dim_atom)
    a = destroy(N_a)
    b = destroy(N_b)

    comps = cavity_sigma_DP_components(dim=dim_atom)
    S_p = comps[+1]
    S_m = comps[-1]

    H_a = g * (tensor(a, I_b, S_p) + tensor(a.dag(), I_b, S_p.dag()))
    H_b = g * (tensor(I_a, b, S_m) + tensor(I_a, b.dag(), S_m.dag()))
    return {'left': H_a, 'right': H_b}




def H_laser_systems_constant(Om_SP_by_q: Dict[int, float], Om_DP_by_q: Dict[int, float], dim: int = 8) -> Dict[str, Qobj]:
    """Sum H_SP and H_DP for each polarization system with constant Ω per q."""
    Hsp = H_SP_systems_constant(Om_SP_by_q, dim=dim)
    Hdp = H_DP_systems_constant(Om_DP_by_q, dim=dim)
    return {
        'left': Hsp.get('left') + Hdp.get('left'),
        'pi':   Hsp.get('pi') + Hdp.get('pi'),
        'right':Hsp.get('right') + Hdp.get('right'),
    }


def H_interaction_with_cavity(
    H_laser_atom: Dict[str, Qobj],
    g: float,
    N_a: int,
    N_b: int,
) -> Dict[str, Qobj]:
    """Lift atomic laser Hamiltonians to 2-mode cavity ⊗ atom space and add H_a/H_b.

    - left/right get: tensor(I_a, I_b, H_atom_left/right) + H_a/H_b
    - pi gets:        tensor(I_a, I_b, H_atom_pi)  (no cavity coupling)
    """
    # Detect atomic dimension from any provided operator
    sample = next(iter(H_laser_atom.values()))
    dim_atom = sample.shape[0]
    I_a = qeye(N_a)
    I_b = qeye(N_b)

    Hc = H_cavity_DP_modes(g=g, N_a=N_a, N_b=N_b, dim_atom=dim_atom)

    H_left  = tensor(I_a, I_b, H_laser_atom.get('left'))  + Hc['left']
    H_right = tensor(I_a, I_b, H_laser_atom.get('right')) + Hc['right']
    H_pi    = tensor(I_a, I_b, H_laser_atom.get('pi'))

    return {'left': H_left, 'pi': H_pi, 'right': H_right}


# Bare two-mode cavity Hamiltonian (Eq. 2.4.7): ħ(Δ_c − Δ_rep)(a†a + b†b)
def H0_cavity(delta_c: float, delta_rep: float, N_a: int, N_b: int, dim_atom: int) -> Qobj:
    coeff = (delta_c - delta_rep)
    I_atom = qeye(dim_atom)
    a = destroy(N_a)
    b = destroy(N_b)
    Hc = coeff * (tensor(a.dag() * a, qeye(N_b), I_atom) + tensor(qeye(N_a), b.dag() * b, I_atom))
    return Hc





def add_bare_to_full_systems(
    H_systems: Dict[str, Qobj],
    H_S: Qobj,
    H_D: Qobj,
    delta_c: float,
    delta_rep: float,
    N_a: int,
    N_b: int,
) -> Dict[str, Qobj]:
    """Add bare ion + bare cavity Hamiltonians to each polarization system."""
    dim_atom = H_S.shape[0]
    Ia = qeye(N_a)
    Ib = qeye(N_b)
    H_bare_atom = tensor(Ia, Ib, H_S + H_D)
    H_cav = H0_cavity(delta_c, delta_rep, N_a, N_b, dim_atom)
    out = {}
    for key, H in H_systems.items():
        out[key] = H + H_bare_atom + H_cav
    return out


# Bare-ion Hamiltonians (Eqs. 2.4.5 and 2.4.6)
def eq_2_4_5_H_S(delta_drive: float, B_T: float, g_S: float = 2.0022664, dim: int = 8) -> Qobj:
    """S-manifold diagonal Hamiltonian in the rotating frame.

    H_S = Σ_{m=±1/2} (Δ_driv + g_S μ_B B m / ħ) |S,m><S,m|
    """
    H = Qobj(np.zeros((dim, dim), dtype=complex))
    for m in (-0.5, +0.5):
        idx = state_index('S', m)
        ket = basis(dim, idx)
        E = delta_drive + g_S * MU_B * B_T * m
        H += E * (ket * ket.dag())
    return H


def eq_2_4_6_H_D(delta_rep: float, B_T: float, g_D: float, dim: int = 8) -> Qobj:
    """D-manifold diagonal Hamiltonian in the rotating frame.

    H_D = Σ_{m∈{-3/2,-1/2,+1/2,+3/2}} (Δ_rep + g_D μ_B B m / ħ) |D,m><D,m|
    """
    H = Qobj(np.zeros((dim, dim), dtype=complex))
    for m in (-1.5, -0.5, +0.5, +1.5):
        idx = state_index('D', m)
        ket = basis(dim, idx)
        E = delta_rep + g_D * MU_B * B_T * m
        H += E * (ket * ket.dag())
    return H


# Add bare-ion Hamiltonians to interactive Hamiltonian for each system 
def add_bare_ion_to_systems(
    systems: Dict[str, Qobj],
    H_S: Qobj,
    H_D: Qobj,
) -> Dict[str, Qobj]:
    """Add the polarization-independent bare Hamiltonians to each system."""
    out: Dict[str, Qobj] = {}
    for key, H in systems.items():
        out[key] = H + H_S + H_D
    return out


# Convenience: projectors onto manifolds in the local 8-state basis
def projector_for_indices(indices: List[int], dim: int = 8) -> Qobj:
    P = Qobj(np.zeros((dim, dim), dtype=complex))
    for i in indices:
        ket = basis(dim, i)
        P += ket * ket.dag()
    return P




# Interactive UI to explore Ca40+ populations
def interactive_populations():
    """Interactive UI: sliders for Δ_c, Δ_rep, Δ_driv (MHz) and B (G),
    plus a radio selector for polarization system (left, pi, right).

    Uses atomic-only interaction + bare ion H0. Δ_c slider is included for
    completeness but does not affect the atomic-only Hamiltonian.
    """
    # Default parameters
    params = dict(
        B_G=5.0,
        delta_drive_MHz=0.0,
        delta_rep_MHz=0.0,
        delta_c_MHz=0.0,
        g_S=2.0022664,
        g_D=0.8,
        tmax_us=50.0,
        density=8.0,  # samples per microsecond
        system='pi',
    )

    # Fixed Rabi frequencies for demo (rad/s)
    Om_SP = {+1: 0.0, 0: 2 * np.pi * 5e6, -1: 0.0}
    Om_DP = {+1: 0.0, 0: 2 * np.pi * 2e6, -1: 2 * np.pi * 2e6}

    # Initial atomic state |S, +1/2>
    psi0_atom = basis(8, state_index('S', +0.5))
    P_S = projector_for_indices([state_index('S', -0.5), state_index('S', +0.5)], dim=8)
    P_P = projector_for_indices([state_index('P', -0.5), state_index('P', +0.5)], dim=8)
    P_D = projector_for_indices([state_index('D', -1.5), state_index('D', -0.5), state_index('D', +0.5), state_index('D', +1.5)], dim=8)

    def build_and_solve():
        B_T = params['B_G'] * 1e-4
        ddrv = 2 * np.pi * params['delta_drive_MHz'] * 1e6
        drep = 2 * np.pi * params['delta_rep_MHz'] * 1e6

        H_spdp = H_laser_systems_constant(Om_SP, Om_DP, dim=8)
        H_S = eq_2_4_5_H_S(ddrv, B_T, params['g_S'], dim=8)
        H_D = eq_2_4_6_H_D(drep, B_T, params['g_D'], dim=8)

        steps = max(100, int(round(params['tmax_us'] * params['density'])))
        tlist = np.linspace(0, params['tmax_us'] * 1e-6, steps)

        # Always include cavity with fixed Fock dimensions
        N_a, N_b = 3, 3
        g = 2 * np.pi * s_gcav.val * 1e6
        dc = 2 * np.pi * s_dcav.val * 1e6
        H_lift = H_interaction_with_cavity(H_spdp, g=g, N_a=N_a, N_b=N_b)
        H_full = add_bare_to_full_systems(H_lift, H_S, H_D, dc, drep, N_a, N_b)
        H = H_full[rb_sys.value_selected]
        Ia, Ib = qeye(N_a), qeye(N_b)
        psi0 = tensor(basis(N_a, 0), basis(N_b, 0), psi0_atom)
        res = mesolve(H, psi0, tlist, c_ops=[], e_ops=[tensor(Ia, Ib, P_S), tensor(Ia, Ib, P_P), tensor(Ia, Ib, P_D)])
        return tlist, res

    fig, ax = plt.subplots(figsize=(9, 6))
    # Reserve extra bottom margin for added cavity controls
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.60)

    # Controls
    axcolor = 'lightgoldenrodyellow'
    row_h, gap = 0.035, 0.007
    y0 = 0.46
    xL, wC = 0.10, 0.35
    xR = 0.55

    def row_y(i):
        return y0 - i * (row_h + gap)

    ax_BG     = plt.axes([xL, row_y(0), wC, row_h], facecolor=axcolor)
    ax_tmax   = plt.axes([xR, row_y(0), wC, row_h], facecolor=axcolor)
    ax_ddrv   = plt.axes([xL, row_y(1), wC, row_h], facecolor=axcolor)
    ax_drep   = plt.axes([xR, row_y(1), wC, row_h], facecolor=axcolor)
    ax_dcav   = plt.axes([xL, row_y(2), wC, row_h], facecolor=axcolor)
    ax_dense  = plt.axes([xR, row_y(2), wC, row_h], facecolor=axcolor)
    ax_gcav   = plt.axes([xL, row_y(3), wC, row_h], facecolor=axcolor)

    s_BG   = Slider(ax_BG,   'B (G)',       0.0, 50.0, valinit=params['B_G'])
    s_tmax = Slider(ax_tmax, 't_max (us)',  5.0, 200.0, valinit=params['tmax_us'])
    s_ddrv = Slider(ax_ddrv, 'Δ_driv (MHz)', -50.0, 50.0, valinit=params['delta_drive_MHz'])
    s_drep = Slider(ax_drep, 'Δ_rep (MHz)', -50.0, 50.0, valinit=params['delta_rep_MHz'])
    s_dcav = Slider(ax_dcav, 'Δ_c (unused)',   -200.0, 200.0, valinit=params['delta_c_MHz'])
    # Control temporal resolution via samples per microsecond to avoid aliasing
    s_dense= Slider(ax_dense,'samples/us',  1.0, 200.0, valinit=params['density'])
    s_gcav = Slider(ax_gcav, 'g_cav (MHz)', 0.0, 50.0, valinit=0.0)

    ax_sys = plt.axes([0.82, 0.08, 0.15, 0.12], facecolor=axcolor)
    rb_sys = RadioButtons(ax_sys, ('left', 'pi', 'right'), active=1)

    # Initial compute and line setup (after controls exist)
    t, res = build_and_solve()
    Pg, Pe, Ps = (np.array(res.expect[i]) for i in range(3))
    ln_e, = ax.plot(t * 1e6, Pe, label='P1/2 population')
    ln_g, = ax.plot(t * 1e6, Pg, label='S1/2 population')
    ln_s, = ax.plot(t * 1e6, Ps, label='D3/2 population')
    ax.set_xlabel('Time (us)')
    ax.set_ylabel('Population')
    ax.set_ylim(0, 1.05)
    ax.set_title('Ca40+ populations')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    def recompute(_=None):
        params['B_G'] = s_BG.val
        params['tmax_us'] = s_tmax.val
        params['density'] = float(s_dense.val)
        params['delta_drive_MHz'] = s_ddrv.val
        params['delta_rep_MHz'] = s_drep.val
        params['delta_c_MHz'] = s_dcav.val
        params['system'] = rb_sys.value_selected
        t2, res2 = build_and_solve()
        ln_e.set_data(t2 * 1e6, np.array(res2.expect[1]))
        ln_g.set_data(t2 * 1e6, np.array(res2.expect[0]))
        ln_s.set_data(t2 * 1e6, np.array(res2.expect[2]))
        ax.set_xlim(t2[0] * 1e6, t2[-1] * 1e6)
        fig.canvas.draw_idle()

    for s in (s_BG, s_tmax, s_dense, s_ddrv, s_drep, s_dcav, s_gcav):
        s.on_changed(recompute)
    rb_sys.on_clicked(recompute)

    plt.show()

if __name__ == "__main__":
    interactive_populations()





