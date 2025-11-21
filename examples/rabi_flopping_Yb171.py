from __future__ import annotations
import math
from typing import Dict, List, Tuple, Callable
import time
from functools import wraps
import numpy as np
from qutip import Qobj, basis, destroy, qeye, tensor, mesolve, sesolve, expect
from qutip.solver import Options
from qutip import correlation_2op_1t
from qutip import correlation_3op_1t
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy import constants as consts
from atomic_physics.ions.Yb171 import Yb171 
from sympy import S as _S
from sympy.physics.wigner import wigner_3j as _sympy_wigner_3j



FUNCTION_TIMINGS: Dict[str, Dict[str, float | int]] = {}

# Physical constants
# We work in angular-frequency units (rad/s) and set ħ = 1.
# Use μB/ħ so g_J μB B has units of rad/s when B is in Tesla.

HBAR = 1.0

MU_B = consts.physical_constants["Bohr magneton"][0] 


def _g_factor_for_level(level) -> float:
  """Fetch the Landé g-factor for a given Yb171 level."""
  for level_data in Yb171.level_data:
    if level_data.level == level:
      if level_data.g_J is None:
        raise ValueError(f"No Landé g-factor defined for {level}.")
      return float(level_data.g_J)
  raise ValueError(f"Level {level} not found in Yb171 data.")


# Landé g-factors drawn from atomic_physics metadata
g_s = _g_factor_for_level(Yb171.S12)
g_d = _g_factor_for_level(Yb171.D32)
# Reference linewidths (2π × MHz or mHz) converted to rad/s.
LINEWIDTH_SP = 2 * np.pi * 21.57e6   # S1/2 ↔ P1/2
LINEWIDTH_DP = 2 * np.pi * 1.482e6   # D3/2 ↔ P1/2 (852 nm)
LINEWIDTH_SD = 2 * np.pi * 136e3     # S1/2 ↔ D3/2 (136 mHz → 136 kHz)
 


def solver_options(store_states: bool = False) -> Options:
  """Solver options tuned for faster convergence without progress bar noise."""
  return Options(
    atol=1e-8,
    rtol=1e-6,
    nsteps=10000,
    store_states=store_states,
    progress_bar=None,
  )


def _set_y_limits(ax, data: np.ndarray | None, default: tuple[float, float], padding: float = 0.1) -> None:
  """Adjust y-axis bounds with padding while handling empty or degenerate data."""
  if data is None or data.size == 0:
    ax.set_ylim(*default)
    return
  low = float(np.nanmin(data))
  high = float(np.nanmax(data))
  if not np.isfinite(low) or not np.isfinite(high):
    ax.set_ylim(*default)
    return
  span = high - low
  pad = max(abs(high) * padding, abs(low) * padding, span * padding, 1e-6)
  if span < 1e-12:
    ax.set_ylim(low - pad, high + pad)
  else:
    ax.set_ylim(low - pad, high + pad)


def _record_timing(name: str, elapsed: float) -> None:
  """Accumulate timing statistics for the timed decorator."""
  stats = FUNCTION_TIMINGS.setdefault(name, {'count': 0, 'total': 0.0, 'max': 0.0})
  stats['count'] += 1
  stats['total'] += elapsed
  if elapsed > stats['max']:
    stats['max'] = elapsed


def timed(func=None, *, label: str | None = None):
  """Decorator to time a function and log its elapsed duration."""
  def decorator(fn):
    name = label or fn.__name__

    @wraps(fn)
    def wrapper(*args, **kwargs):
      start = time.perf_counter()
      try:
        return fn(*args, **kwargs)
      finally:
        elapsed = time.perf_counter() - start
        _record_timing(name, elapsed)

    return wrapper

  if func is None:
    return decorator
  return decorator(func)


def print_timing_summary() -> None:
  if not FUNCTION_TIMINGS:
    print("Timing summary: no calls recorded.")
    return
  print("Timing summary (seconds):")
  for name in sorted(FUNCTION_TIMINGS):
    stats = FUNCTION_TIMINGS[name]
    avg = stats['total'] / stats['count']
    print(
      f" {name}: count={stats['count']}, total={stats['total']:.6f}, "
      f"avg={avg:.6f}, max={stats['max']:.6f}"
    )



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


def _clebsch_gordan(j1: float, m1: float, j2: float, m2: float, J: float, M: float) -> float:
  
  # Use integer parity for the phase to avoid complex results from float powers
  exp2 = int(round(2.0 * (-j1 -+j2 - M))) # 2*(j1 - j2 + M) should be integer
  phase = 1.0 if (exp2 % 2 == 0) else -1.0
  w3 = wigner_3j(j1, j2, J, m1, m2, -M)
  
  # Ensure numeric real scalar even if the backend returns an exact/complex type
  w3_real = float(complex(w3).real)
  val = phase * math.sqrt(2.0 * J + 1.0) * w3_real
  return float(val)


# Local 8-state basis ordering (only used for the constant-Ω fallback):
# 0: S1/2, m=-1/2; 1: S1/2, m=+1/2; 2: P1/2, m=-1/2; 3: P1/2, m=+1/2;
# 4: D3/2, m=-3/2; 5: D3/2, m=-1/2; 6: D3/2, m=+1/2; 7: D3/2, m=+3/2

Level = str # 'S', 'P', or 'D'
CGC_DATA: Dict[str, Dict[int, List[Tuple[Tuple[Level, float], Tuple[Level, float], float]]]] = {}



def rabi_frequencies(E0_driv: float, E0_rep: float) -> Tuple[float, float]:
  """Compute Ω_driv and Ω_rep (rad/s) for Eqs. 2.4.12 and 2.4.13.

  Uses Eq. 2.1.8 with ħ = 1. The polarization geometry factor (ε·d)
  is handled by the Clebsch-Gordan structure inside the sigma operators
  built in this module, so here Ω = E0 in rad/s.

  Parameters
  - E0_driv: driving laser field amplitude (rad/s scale)
  - E0_rep: repump laser field amplitude (rad/s scale)
  """
  return float(E0_driv)/(2*HBAR), float(E0_rep)/(2*HBAR)

#This maps the m_j states to integer indicies so we can build loops with correct indexing later on 
def state_index(level: Level, m: float) -> int:
  """Map a (level, m) pair to its index in the 8-state local basis."""
  if level == 'S':
    return 0 if m == -0.5 else 1
  if level == 'P':
    return 2 if m == -0.5 else 3
  if level == 'D':
    return { -1.5: 4, -0.5: 5, 0.5: 6, 1.5: 7 }[m]
  raise ValueError(f"Unknown state ({level}, {m})")


# This maps Clebsh-Gordan coefficients to a Tuple which can then iterate over to get dipole moments for transitions 
# Generalized CGC computation using Wigner-3j and Yb171 J values

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


@timed()
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

  SP = build_pair('S', J_S, 'P', J_P)
  DP = build_pair('D', J_D, 'P', J_P)
  return {'SP': SP, 'DP': DP}

# Precompute once
CGC_DATA = cgc_lists_general()



# This computes the change in m_j for a given transition, this is used to determine the the transition type, and the polarization of a transition
def delta_mj(m_lower: float, m_upper: float) -> int:
  """Return Δm_j = m_upper - m_lower in {-1,0,+1} (rounded to nearest int)."""
  return int(round(m_upper - m_lower))

# The way we are dealing with different polarizations is to create sepereate systems for each component. (Think about how gravity only effects vertical motion, not horizontal motion)
# This dictionary maps the change in m_j to its corresponding polarization. We can use this to build separate Hamiltonians for each polarization component.
_q_to_label = {+1: 'left', 0: 'pi', -1: 'right'}


# This one just returns the index of a state in the local basis
def _index_from_local(level: Level, m: float) -> int:
  """Wrapper to reuse state_index in places that expect an index callback."""
  return state_index(level, m)



# This function builds H_SP and H_DP Hamiltonians for each polarization system
@timed()
def _build_systems(
  pair: str,
  Omega_value: float,
  index_of: Callable[[Level, float], int],
  dim: int,
) -> Dict[str, Qobj]:
  """Core builder: accumulate three Hamiltonians (left, pi, right).

  For each CGC-listed transition A,m_A → B,m_B, compute q=Δm and
  add ½ Ω · CGC (|u><l| + h.c.) into the appropriate system using
  the fixed Rabi rate Ω provided in Omega_value.

  pair is 'SP' or 'DP' to select which transitions to include.
  index_of maps (level, m) to local basis index.
  dim is the local basis dimension (8).
  """
  systems = {name: Qobj(np.zeros((dim, dim), dtype=complex)) for name in ('left', 'pi', 'right')}
  data = CGC_DATA[pair]
  for q_group in (-1, 0, +1):
    for (lower, upper, cgc) in data.get(q_group, []):
      m_l, m_u = lower[1], upper[1]
      q = delta_mj(m_l, m_u)
      print(f"[{pair}] Δm={q} for {lower}->{upper} (cgc={cgc})")
      if q not in (-1, 0, +1):
        continue
      Omega = float(Omega_value)
      li = index_of(lower[0], m_l)
      ui = index_of(upper[0], m_u)
      ket_l = basis(dim, li)
      ket_u = basis(dim, ui)
      sigma = ket_u * ket_l.dag()
      label = _q_to_label[q]
      systems[label] = systems[label] + 0.5 * HBAR * Omega * cgc * (sigma + sigma.dag())
  return systems



# This builds sigma componenets for H_SP and H_DP
SIGMA_CACHE: Dict[Tuple[str, int], Dict[int, Qobj]] = {}

@timed()
def transition_sigma_components(pair: str, dim: int = 8) -> Dict[int, Qobj]:
  """Build non-Hermitian atomic operators S_q = Σ_{Δm=q} CGC |upper><lower|.

  This is a local-basis helper (uses state_index) for composing Eq. 2.4.12-2.4.15
  pieces. It does not rely on any external ion object.
  """
  cache_key = (pair, dim)
  if cache_key in SIGMA_CACHE:
    return SIGMA_CACHE[cache_key]
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
  SIGMA_CACHE[cache_key] = comps
  return comps

@timed()
def cavity_sigma_DP_components(dim: int = 8) -> Dict[int, Qobj]:
    """Non-Hermitian atomic operators for the cavity D↔P coupling.

    Returns S_q = Σ_{Δm=q} CGC |P><D| for q ∈ {+1, -1}. (No π component.)
    """
    S = transition_sigma_components('DP', dim=dim)
    return {+1: S.get(+1, Qobj(np.zeros((dim, dim), dtype=complex))),
            -1: S.get(-1, Qobj(np.zeros((dim, dim), dtype=complex)))}

def precompute_lowering_tensors(N_a: int, N_b: int, dim_atom: int) -> Dict[str, List[Tuple[Qobj, float]]]:
    """Precompute and tensor atomic lowering operators into cavity⊗atom space."""
    Ia = qeye(N_a)
    Ib = qeye(N_b)
    tensors: Dict[str, List[Tuple[Qobj, float]]] = {}
    for pair in ('SP', 'DP'):
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


# This builds H_a and H_b for left and right polarizations
@timed()
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
  a = destroy(N_a)
  b = destroy(N_b)

  comps = cavity_sigma_DP_components(dim=dim_atom)
  S_p = comps[+1]
  S_m = comps[-1]

  H_a = HBAR* g * (tensor(a, I_b, S_p) + tensor(a.dag(), I_b, S_p.dag()))
  H_b = HBAR * g * (tensor(I_a, b, S_m) + tensor(I_a, b.dag(), S_m.dag()))
  return {'left': H_a, 'right': H_b}


@timed()
def H_laser_systems_from_E0(E0_driv: float, E0_rep: float, dim: int = 8) -> Dict[str, Qobj]:
  """Laser Hamiltonians using field amplitudes E0 (rad/s scale).

  Implements Eq. 2.1.8 with ħ = 1 and the polarization geometry handled by
  CGCs in the sigma terms; we therefore pass Ω = E0 to _build_systems, which
  already applies the 1/2 prefactor internally.
  """
  Omega_driv, Omega_rep = rabi_frequencies(E0_driv, E0_rep)
  Hsp = _build_systems('SP', float(Omega_driv), _index_from_local, dim)
  Hdp = _build_systems('DP', float(Omega_rep), _index_from_local, dim)
  return {
    'left': Hsp.get('left') + Hdp.get('left'),
    'pi':  Hsp.get('pi') + Hdp.get('pi'),
    'right':Hsp.get('right') + Hdp.get('right'),
  }


# Builds the interaction Hamiltonians between the cavity and the atomic states H_a and H_b and adds it to the laser-atom Hamiltonians for the total interaction Hamiltonian
@timed()
def H_interaction_with_cavity(
  H_laser_atom: Dict[str, Qobj],
  g: float,
  N_a: int,
  N_b: int,
) -> Dict[str, Qobj]:
  """Lift atomic laser Hamiltonians to 2-mode cavity ⊗ atom space and add H_a/H_b.

  - left/right get: tensor(I_a, I_b, H_atom_left/right) + H_a/H_b
  - pi gets:    tensor(I_a, I_b, H_atom_pi) (no cavity coupling)
  """
  # Detect atomic dimension from any provided operator
  sample = next(iter(H_laser_atom.values()))
  dim_atom = sample.shape[0]
  I_a = qeye(N_a)
  I_b = qeye(N_b)

  Hc = H_cavity_DP_modes(g=g, N_a=N_a, N_b=N_b, dim_atom=dim_atom)

  H_left = tensor(I_a, I_b, H_laser_atom.get('left')) + Hc['left']
  H_right = tensor(I_a, I_b, H_laser_atom.get('right')) + Hc['right']
  H_pi  = tensor(I_a, I_b, H_laser_atom.get('pi'))

  return {'left': H_left, 'pi': H_pi, 'right': H_right}


# Bare two-mode cavity Hamiltonian (Eq. 2.4.7): ħ(Δ_c − Δ_rep)(a†a + b†b)
@timed()
def H0_cavity(delta_c: float, delta_rep: float, N_a: int, N_b: int, dim_atom: int) -> Qobj:
  coeff = (delta_c - delta_rep)
  I_atom = qeye(dim_atom)
  a = destroy(N_a)
  b = destroy(N_b)
  Hc = coeff * HBAR * (tensor(a.dag() * a, qeye(N_b), I_atom) + tensor(qeye(N_a), b.dag() * b, I_atom))
  return Hc


# Bare-ion Hamiltonians (Eqs. 2.4.5 and 2.4.6)
@timed()
def eq_2_4_5_H_S(delta_drive: float, B_T: float, g_S: float = g_s, dim: int = 8) -> Qobj:
  """S-manifold diagonal Hamiltonian in the rotating frame. 7

  H_S = Σ_{m=±1/2} (Δ_driv + g_S μ_B B m) |S,m><S,m| (μ_B ≡ μB/ħ)
  """
  H = Qobj(np.zeros((dim, dim), dtype=complex))
  for m in (-0.5, +0.5):
    idx = state_index('S', m)
    ket = basis(dim, idx)
    E = delta_drive + g_S * MU_B * B_T * m
    H += E * (ket * ket.dag())
  return H

@timed()
def eq_2_4_6_H_D(delta_rep: float, B_T: float, g_D: float = g_d, dim: int = 8) -> Qobj:
  """D-manifold diagonal Hamiltonian in the rotating frame.

  H_D = Σ_{m∈{-3/2,-1/2,+1/2,+3/2}} (Δ_rep + g_D μ_B B m) |D,m><D,m| (μ_B ≡ μB/ħ)
  """
  H = Qobj(np.zeros((dim, dim), dtype=complex))
  for m in (-1.5, -0.5, +0.5, +1.5):
    idx = state_index('D', m)
    ket = basis(dim, idx)
    E = delta_rep + g_D * MU_B * B_T * m
    H += E * (ket * ket.dag())
  return H


# Adds all of of the terms together to get the full Hamiltonian
@timed()
def add_bare_terms_to_systems(
  H_systems: Dict[str, Qobj],
  H_S: Qobj,
  H_D: Qobj,
  delta_c: float | None = None,
  delta_rep: float | None = None,
  N_a: int | None = None,
  N_b: int | None = None,
) -> Dict[str, Qobj]:
  """Add bare-ion terms and bare-cavity terms to each system.

  - Always adds H_S + H_D to every polarization-resolved Hamiltonian in
   `H_systems`. If `N_a`, `N_b`, `delta_c`, and `delta_rep` are provided,
   also adds the two-mode cavity term H0_cavity.
  """
  
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


# Convenience: projectors onto manifolds in the local 8-state basis
@timed()
def projector_for_indices(indices: List[int], dim: int = 8) -> Qobj:
  P = Qobj(np.zeros((dim, dim), dtype=complex))
  for i in indices:
    ket = basis(dim, i)
    P += ket * ket.dag()
  return P


def build_collapse_operators(
    params: Dict[str, float],
    lowering_tensors: Dict[str, List[Tuple[Qobj, float]]],
    projector_tensors: Dict[str, Qobj],
    cavity_loss_ops: List[Qobj],
) -> List[Qobj]:
    """Assemble all collapse operators (atomic decay, cavity loss, dephasing)."""
    c_ops: List[Qobj] = []

    gamma_sp = params['gamma_SP']
    gamma_dp = params['gamma_DP']
    gamma_map = {'SP': gamma_sp, 'DP': gamma_dp}

    for pair, gamma in gamma_map.items():
        if gamma <= 0:
            continue
        for op, weight in lowering_tensors[pair]:
            if weight > 0:
                c_ops.append(np.sqrt(2 * gamma) * weight * op)

    kappa = 2 * np.pi * params['kappa_MHz'] * 1e6
    if kappa > 0:
        for loss_op in cavity_loss_ops:
            c_ops.append(np.sqrt(2 * kappa) * loss_op)

    delta_S = params['gamma_SD']
    if delta_S > 0:
        c_ops.append(np.sqrt(2 * delta_S) * projector_tensors['S'])
    delta_D = params['gamma_DP']
    if delta_D > 0:
        c_ops.append(np.sqrt(2 * delta_D) * projector_tensors['D'])

    return c_ops



# First-order coherence G^(1) and degree g^(1) for photon generation probability
@timed()
def first_order_coherence(
  H: Qobj,
  psi0: Qobj,
  tlist: np.ndarray,
  c_ops: list | None,
  a_field: Qobj,
  taulist: np.ndarray,
  states: List[Qobj] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Compute steady-state G¹(τ), model g¹(τ), and g²(τ).

  - H, psi0, tlist: system Hamiltonian, initial state, times
  - c_ops: collapse operators (may be empty list)
  - a_field: field annihilation operator in the full Hilbert space
  - taulist: delays τ to evaluate
  Returns three arrays of length len(taulist).
  """
  c_ops = [] if c_ops is None else c_ops

  def _zeros():
    zeros = np.zeros_like(taulist)
    return zeros, zeros, zeros

  if taulist.size == 0:
    return _zeros()

  def _steady_state() -> Qobj | None:
    if states:
      last = states[-1]
      return last if last.isoper else (last * last.dag())
    sol = sesolve(H, psi0, tlist, options=solver_options(store_states=True))
    if sol.states:
      last = sol.states[-1]
      return last if last.isoper else (last * last.dag())
    return None

  rho_final = _steady_state()
  if rho_final is None:
    return _zeros()

  a_dag = a_field.dag()
  number_op = a_dag * a_field
  intensity = float(np.real(expect(number_op, rho_final)))
  if abs(intensity) < 1e-12:
    return _zeros()

  corr = np.asarray(
    correlation_2op_1t(H, rho_final, taulist, c_ops, a_dag, a_field, 'me'),
    dtype=complex,
  )
  G1_tau = corr.real
  g1_tau_raw = corr / intensity
  tau_c = float(np.trapezoid(np.abs(g1_tau_raw) ** 2, x=taulist))
  g1_model = np.zeros_like(taulist) if tau_c <= 0 else np.exp(-(taulist / tau_c) ** 2)
  print(f"tau_c = {tau_c:.3e} s")

  g2_linear = np.asarray(
    correlation_3op_1t(
      H,
      rho_final,
      taulist,
      c_ops,
      a_dag,
      number_op,
      a_field,
      solver='me',
    ),
    dtype=complex,
  )
  denom_g2 = intensity ** 2
  g2_tau = np.zeros_like(taulist) if denom_g2 <= 1e-24 else np.real(g2_linear / denom_g2)

  return G1_tau, g1_model, g2_tau







# Interactive UI to explore Yb171+ populations
@timed()
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
    g_S=g_s,
    g_D=g_d,
    gamma_SP=0.2,
    gamma_DP=0.2,
    gamma_SD=0.2,
    kappa_MHz=0.5,
    tmax_us=50.0,
    density=8.0, # samples per microsecond
    system='pi',
    amp_driv_MHz=0.0, # driving laser amplitude E0 (MHz)
    amp_rep_MHz=0.0,  # repump laser amplitude E0 (MHz)
    tau_max_us=0.5,   # window for first-order coherence (keep small by default)
    n_tau=20,      # number of tau samples
  )


  # Initial atomic state |S, +1/2>
  psi0_atom = basis(8, state_index('S', + 0.5))
  P_S = projector_for_indices([state_index('S', -0.5), state_index('S', +0.5)], dim=8)
  P_P = projector_for_indices([state_index('P', -0.5), state_index('P', +0.5)], dim=8)
  P_D = projector_for_indices([state_index('D', -1.5), state_index('D', -0.5), state_index('D', +0.5), state_index('D', +1.5)], dim=8)

  N_a = 3
  N_b = 3
  Ia = qeye(N_a)
  Ib = qeye(N_b)
  I_atom = qeye(8)
  projector_tensors = {
    'S': tensor(Ia, Ib, P_S),
    'P': tensor(Ia, Ib, P_P),
    'D': tensor(Ia, Ib, P_D),
  }
  field_left = tensor(destroy(N_a), Ib, qeye(8))
  field_right = tensor(Ia, destroy(N_b), qeye(8))
  cavity_loss_ops = [
    tensor(destroy(N_a), Ib, I_atom),
    tensor(Ia, destroy(N_b), I_atom),
  ]
  lowering_tensors = precompute_lowering_tensors(N_a, N_b, 8)

  @timed(label='build_and_solve')
  def build_and_solve():
    B_T = params['B_G'] * 1e-4
    ddrv = 2 * np.pi * params['delta_drive_MHz'] * 1e6
    drep = 2 * np.pi * params['delta_rep_MHz'] * 1e6

    E0_driv = 2 * np.pi * params['amp_driv_MHz'] * 1e6
    E0_rep = 2 * np.pi * params['amp_rep_MHz'] * 1e6
    H_spdp = H_laser_systems_from_E0(E0_driv, E0_rep, dim=8)
    H_S = eq_2_4_5_H_S(ddrv, B_T, params['g_S'], dim=8)
    H_D = eq_2_4_6_H_D(drep, B_T, params['g_D'], dim=8)

    steps = max(100, int(round(params['tmax_us'] * params['density'])))
    tlist = np.linspace(0, params['tmax_us'] * 1e-6, steps)

    g = 2 * np.pi * s_gcav.val * 1e6
    dc = 2 * np.pi * s_dcav.val * 1e6
    if rb_sys.value_selected in ('left', 'right'):
      H_lift = H_interaction_with_cavity(H_spdp, g=g, N_a=N_a, N_b=N_b)
    else:
      H_lift = {name: tensor(Ia, Ib, H_atom) for name, H_atom in H_spdp.items()}
    H_full = add_bare_terms_to_systems(H_lift, H_S, H_D, dc, drep, N_a, N_b)
    pol = rb_sys.value_selected
    H = H_full[pol]
    psi0 = tensor(basis(N_a, 0), basis(N_b, 0), psi0_atom)
    c_ops = build_collapse_operators(params, lowering_tensors, projector_tensors, cavity_loss_ops)
    mesolve_opts = solver_options(store_states=False)
    res = mesolve(
      H,
      psi0,
      tlist,
      c_ops=c_ops,
      e_ops=[projector_tensors['S'], projector_tensors['P'], projector_tensors['D']],
      options=mesolve_opts,
    )

    # Build field operator for selected polarization (left/right)
    taulist = None
    g1_tau = None
    if pol in ('left', 'right'):
      tau_max = params['tau_max_us'] * 1e-6
      taulist = np.linspace(0.0, tau_max, int(params['n_tau']))
      field = field_left if pol == 'left' else field_right
      _G1, g1_tau, g2_tau = first_order_coherence(
        H,
        psi0,
        tlist,
        c_ops,
        field,
        taulist,
        states=res.states,
      )
    else:
      g2_tau = None
    return tlist, res, taulist, g1_tau, g2_tau

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

  ax_BG   = plt.axes([xL, row_y(0), wC, row_h], facecolor=axcolor)
  ax_tmax  = plt.axes([xR, row_y(0), wC, row_h], facecolor=axcolor)
  ax_ddrv  = plt.axes([xL, row_y(1), wC, row_h], facecolor=axcolor)
  ax_drep  = plt.axes([xR, row_y(1), wC, row_h], facecolor=axcolor)
  ax_dcav  = plt.axes([xL, row_y(2), wC, row_h], facecolor=axcolor)
  ax_dense = plt.axes([xR, row_y(2), wC, row_h], facecolor=axcolor)
  ax_gcav  = plt.axes([xL, row_y(3), wC, row_h], facecolor=axcolor)
  ax_ampdrv = plt.axes([xL, row_y(4), wC, row_h], facecolor=axcolor)
  ax_amprep = plt.axes([xR, row_y(4), wC, row_h], facecolor=axcolor)
  ax_taumax = plt.axes([xL, row_y(5), wC, row_h], facecolor=axcolor)
  ax_gammasp = plt.axes([xR, row_y(5), wC, row_h], facecolor=axcolor)
  ax_gammadp = plt.axes([xL, row_y(6), wC, row_h], facecolor=axcolor)
  ax_gammasd = plt.axes([xR, row_y(6), wC, row_h], facecolor=axcolor)
  ax_kappa = plt.axes([xL, row_y(7), wC, row_h], facecolor=axcolor)

  s_BG  = Slider(ax_BG,  'B (G)',    0.0, 50.0, valinit=params['B_G'])
  s_tmax = Slider(ax_tmax, 't_max (us)', 5.0, 200.0, valinit=params['tmax_us'])
  s_ddrv = Slider(ax_ddrv, 'Δ_driv (MHz)', -3.0, 3.0, valinit=params['delta_drive_MHz'])
  s_drep = Slider(ax_drep, 'Δ_rep (MHz)', -3.0, 3.0, valinit=params['delta_rep_MHz'])
  s_dcav = Slider(ax_dcav, 'Δ_c (MHz)',  -3.0, 3.0, valinit=params['delta_c_MHz'])

  s_dense= Slider(ax_dense,'samples/us', 1.0, 100.0, valinit=params['density'])
  s_gcav = Slider(ax_gcav, 'g_cav (MHz)', 0.0, 50.0, valinit=0.0)
  s_ampdrv = Slider(ax_ampdrv, 'Ω_drive (MHz)', -30.0, 30.0, valinit=params['amp_driv_MHz'])
  s_amprep = Slider(ax_amprep, 'Ω_rep (MHz)', -30.0, 30.0, valinit=params['amp_rep_MHz'])
  s_taumax = Slider(ax_taumax, 'τ_max (us)', 0.05, 10.0, valinit=params['tau_max_us'])
  s_gamma_sp = Slider(ax_gammasp, 'γ_SP', 0.0, 1.0, valinit=params['gamma_SP'])
  s_gamma_dp = Slider(ax_gammadp, 'γ_DP', 0.0, 1.0, valinit=params['gamma_DP'])
  s_gamma_sd = Slider(ax_gammasd, 'γ_SD', 0.0, 1.0, valinit=params['gamma_SD'])
  s_kappa = Slider(ax_kappa, 'κ (MHz)', 0.0, 1.0, valinit=params['kappa_MHz'])

  ax_sys = plt.axes([0.82, 0.08, 0.15, 0.12], facecolor=axcolor)
  rb_sys = RadioButtons(ax_sys, ('left', 'pi', 'right'), active=1)

  # Initial compute and line setup (after controls exist)
  t, res, taulist, g1_tau, g2_tau = build_and_solve()
  Pg, Pe, Ps = (np.array(res.expect[i]) for i in range(3))
  ln_e, = ax.plot(t * 1e6, Pe, label='P1/2 population')
  ln_g, = ax.plot(t * 1e6, Pg, label='S1/2 population')
  ln_s, = ax.plot(t * 1e6, Ps, label='D3/2 population')
  ax.set_xlabel('Time (us)')
  ax.set_ylabel('Population')
  ax.set_ylim(0, 1.05)
  ax.set_title('Yb171+ populations')
  ax.grid(True, alpha=0.3)
  ax.legend(loc='upper right')

  # First-order coherence plot (separate figure to keep UI simple)
  fig_g1, ax_g1 = plt.subplots(figsize=(8, 3))
  if g1_tau is not None:
    ln_g1, = ax_g1.plot(taulist * 1e6, g1_tau, label='g^(1)(τ)')
    ax_g1.set_xlim(taulist[0] * 1e6, taulist[-1] * 1e6)
  else:
    ln_g1, = ax_g1.plot([], [], label='g^(1)(τ)')
  ax_g1.set_xlabel('Delay τ (us)')
  ax_g1.set_ylabel('g^(1)')
  _set_y_limits(ax_g1, g1_tau, (0.0, 1.0))
  ax_g1.grid(True, alpha=0.3)
  ax_g1.legend(loc='upper right')

  fig_g2, ax_g2 = plt.subplots(figsize=(8, 3))
  if g2_tau is not None:
    ln_g2, = ax_g2.plot(taulist * 1e6, g2_tau, label='g^(2)(τ)')
    ax_g2.set_xlim(taulist[0] * 1e6, taulist[-1] * 1e6)
  else:
    ln_g2, = ax_g2.plot([], [], label='g^(2)(τ)')
  ax_g2.set_xlabel('Delay τ (us)')
  ax_g2.set_ylabel('g^(2)')
  _set_y_limits(ax_g2, g2_tau, (0.0, 2.0))
  ax_g2.grid(True, alpha=0.3)
  ax_g2.legend(loc='upper right')

  def recompute(_=None):
    params['B_G'] = s_BG.val
    params['tmax_us'] = s_tmax.val
    params['density'] = float(s_dense.val)
    params['delta_drive_MHz'] = s_ddrv.val
    params['delta_rep_MHz'] = s_drep.val
    params['delta_c_MHz'] = s_dcav.val
    params['system'] = rb_sys.value_selected
    params['amp_driv_MHz'] = s_ampdrv.val
    params['amp_rep_MHz'] = s_amprep.val
    params['tau_max_us']  = s_taumax.val
    params['gamma_SP'] = s_gamma_sp.val
    params['gamma_DP'] = s_gamma_dp.val
    params['gamma_SD'] = s_gamma_sd.val
    params['kappa_MHz'] = s_kappa.val
    t2, res2, tau2, g1_2, g2_2 = build_and_solve()
    ln_e.set_data(t2 * 1e6, np.array(res2.expect[1]))
    ln_g.set_data(t2 * 1e6, np.array(res2.expect[0]))
    ln_s.set_data(t2 * 1e6, np.array(res2.expect[2]))
    ax.set_xlim(t2[0] * 1e6, t2[-1] * 1e6)
    if g1_2 is not None:
      ln_g1.set_data(tau2 * 1e6, g1_2)
      ax_g1.set_xlim(tau2[0] * 1e6, tau2[-1] * 1e6)
      _set_y_limits(ax_g1, g1_2, (0.0, 1.0))
    else:
      _set_y_limits(ax_g1, None, (0.0, 1.0))
    if g2_2 is not None:
      ln_g2.set_data(tau2 * 1e6, g2_2)
      ax_g2.set_xlim(tau2[0] * 1e6, tau2[-1] * 1e6)
      _set_y_limits(ax_g2, g2_2, (0.0, 2.0))
    else:
      _set_y_limits(ax_g2, None, (0.0, 2.0))
    fig.canvas.draw_idle()
    fig_g1.canvas.draw_idle()
    fig_g2.canvas.draw_idle()

  for s in (s_BG, s_tmax, s_dense, s_ddrv, s_drep, s_dcav, s_gcav, s_ampdrv, s_amprep, s_taumax, s_gamma_sp, s_gamma_dp, s_gamma_sd, s_kappa):
    s.on_changed(recompute)
  rb_sys.on_clicked(recompute)

  plt.show()
  print_timing_summary()

if __name__ == "__main__":
  interactive_populations()
