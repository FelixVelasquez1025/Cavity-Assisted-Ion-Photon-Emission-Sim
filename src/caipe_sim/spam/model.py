"""Rotating-frame optical Bloch model; see docs/spam-physics-notes.md.

No motion, cavity, intensity calibration or detection-count model is included.
"""
from dataclasses import dataclass

import numpy as np
from qutip import Qobj, QobjEvo, liouvillian, mesolve, steadystate
from scipy.linalg import expm

from . import atomic as a

# Published rounded tone settings, NOT exact resonances from adopted splittings.
# Egan (2021) §2.2.2/3: +14.748 and +2.105 GHz. Olmschenk (2007) Sec. IV:
# 3.07 GHz modulation magnitude; minus sign follows its Fig. 5 level ordering.
SIDEBAND_HZ = {"doppler": 14.748e9, "pumping": 2.105e9, "D_F2": -3.07e9}
PAIRS = {"369": ("S", "P"), "935": ("D", "A")}


@dataclass(frozen=True)
class Tone:
    offset_hz: float
    rabi_rad_s: float
    phase_rad: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.offset_hz, self.rabi_rad_s, self.phase_rad)) or self.rabi_rad_s < 0:
            raise ValueError("Tone parameters must be finite; Rabi magnitude nonnegative")


@dataclass(frozen=True)
class Beam:
    """Carrier plus optional sidebands sharing a linear polarization.

    rabi_rad_s = electric-field amplitude * electronic reduced element / hbar.
    Each sublevel H_ul/hbar = rabi_rad_s * signed angular coefficient *
    epsilon_q / 2. This does NOT use the legacy cavity slider convention.
    Carrier detuning is laser minus S F1->P F0 or D F1->A F0 at B=0.
    Sideband Rabi frequencies are independent user inputs (no modulation
    index or intensity partition is inferred).
    """
    line: str
    rabi_rad_s: float
    theta_rad: float
    detuning_hz: float = 0.0
    sidebands: tuple[Tone, ...] = ()
    linewidth_dephasing_s: float = 0.0

    def __post_init__(self):
        if self.line not in PAIRS:
            raise ValueError("line must be '369' or '935'")
        Tone(self.detuning_hz, self.rabi_rad_s)
        if not np.isfinite(self.theta_rad) or not np.isfinite(self.linewidth_dephasing_s) or self.linewidth_dephasing_s < 0:
            raise ValueError("Angle and dephasing must be finite; dephasing nonnegative")
        offsets = [0.0]+[t.offset_hz for t in self.sidebands]
        if len(set(offsets)) != len(offsets):
            raise ValueError("Combine coherent tones at the same frequency before constructing a beam")

    @property
    def tones(self):
        return (Tone(0.0, self.rabi_rad_s),)+tuple(self.sidebands)


def linear_polarization(theta_rad: float) -> dict:
    """B along z, electric field in xz: epsilon_0=cos(theta), epsilon_±=∓sin(theta)/sqrt(2)."""
    return {0: np.cos(theta_rad), 1: -np.sin(theta_rad)/np.sqrt(2),
            -1: np.sin(theta_rad)/np.sqrt(2)}


@dataclass
class FreeSpaceModel:
    hamiltonian: Qobj
    collapses: list
    # (operator multiplying exp(-i*beat*t), beat rad/s). Hermitian conjugates implicit.
    omitted_tones: list
    assignments: dict

    def full_time_dependent_hamiltonian(self):
        """All tone/transition pairs, in exactly the same frame as the secular model."""
        terms = [self.hamiltonian]
        for op, beat in self.omitted_tones:
            def oscillation(t, beat=beat):
                return np.exp(-1j*beat*t)
            def conjugate(t, beat=beat):
                return np.exp(1j*beat*t)
            terms.extend([[op, oscillation], [op.dag(), conjugate]])
        return QobjEvo(terms)

    def stationary_state(self):
        """Steady solution of the secular model; check uniqueness for each use."""
        return steadystate(self.hamiltonian, self.collapses, method="direct")

    def evolve(self, initial, times_s):
        """Exact static Lindblad propagator exp(L*t), within floating-point error.

        Dense exponentiation avoids time-stepping at GHz internal coherences
        in long-run checks. No adiabatic elimination, projection or positivity
        repair is applied. Intended for small (20-state) validation runs.
        """
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) < 0):
            raise ValueError("times_s must be finite, nonnegative and ordered")
        rho = initial.proj() if initial.isket else initial
        if rho.shape != (a.DIMENSION, a.DIMENSION):
            raise ValueError("initial must use the 20-state basis")
        generator = liouvillian(self.hamiltonian, self.collapses).full()
        vector = rho.full().reshape(-1, order="F")
        states, previous, propagators = [], 0.0, {}
        steps = np.diff(times)
        uniform_step = (float(steps[0]) if len(steps) and steps[0] > 0
                        and np.allclose(steps, steps[0], rtol=1e-12, atol=0) else None)
        for t in times:
            # linspace intervals differ by roundoff: reuse one exact uniform
            # step rather than exponentiating the same generator repeatedly.
            dt = uniform_step if uniform_step is not None and previous != 0 else float(t-previous)
            if dt:
                if dt not in propagators:
                    propagators[dt] = expm(generator*dt)
                vector = propagators[dt] @ vector
            states.append(Qobj(vector.reshape((a.DIMENSION, a.DIMENSION), order="F")))
            previous = t
        return states

    def full_evolution(self, initial, times_s, *, max_step_s):
        """Short-window reference retaining every tone, within optical RWA."""
        return mesolve(self.full_time_dependent_hamiltonian(), initial, times_s,
                       self.collapses, options=dict(atol=1e-11, rtol=1e-9,
                       nsteps=1000000, max_step=max_step_s, normalize_output=False)).states


def build_model(beams=(), *, B_gauss=0.0, clock_quadratic=False) -> FreeSpaceModel:
    """Assign each hyperfine edge its nearest nonzero tone; keep all edges.

    These J=1/2 upper manifolds give a TREE of three allowed F->F' edges
    per beam. Thus a phase for every F block can make all assigned tones
    static simultaneously. With one tone, NOTHING is dropped: even the
    off-resonant S F0 and P F1 couplings stay in the Hamiltonian. With
    multiple tones only the additional, unassigned beat terms are dropped.
    `full_time_dependent_hamiltonian` puts them all back for validation.
    """
    beams = tuple(beams)
    if len({b.line for b in beams}) != len(beams):
        raise ValueError("Use one Beam with sidebands for each line")
    shifts, interactions, omitted, assignments = {}, [], [], {}
    for beam in beams:
        lower, upper = PAIRS[beam.line]
        ref = a.hyperfine_hz(upper, 0)-a.hyperfine_hz(lower, 1)
        tones = [t for t in beam.tones if t.rabi_rad_s > 0]
        if not tones:
            continue
        edges = []
        polarization = linear_polarization(beam.theta_rad)
        for F in a.MANIFOLDS[lower].Fs:
            for Fp in a.MANIFOLDS[upper].Fs:
                entries = [(l, u, a.coefficient(l, u)) for l in a.STATES for u in a.STATES
                           if l[0:2] == (lower, F) and u[0:2] == (upper, Fp) and abs(u[2]-l[2]) <= 1]
                if not any(c for _, _, c in entries):
                    continue
                resonance = a.hyperfine_hz(upper, Fp)-a.hyperfine_hz(lower, F)-ref
                selected = min(tones, key=lambda t: abs(resonance-beam.detuning_hz-t.offset_hz))
                edge_op = Qobj(np.zeros((a.DIMENSION, a.DIMENSION), dtype=complex))
                for l, u, c in entries:
                    edge_op += c*polarization[u[2]-l[2]] * a.ket(*u)*a.ket(*l).dag()/2
                selected_op = selected.rabi_rad_s*np.exp(-1j*selected.phase_rad)*edge_op
                interactions.append(selected_op+selected_op.dag())
                for tone in tones:
                    if tone is not selected:
                        omitted.append((tone.rabi_rad_s*np.exp(-1j*tone.phase_rad)*edge_op,
                                        2*np.pi*(tone.offset_hz-selected.offset_hz)))
                edges.append(((lower,F),(upper,Fp), ref+beam.detuning_hz+selected.offset_hz))
                assignments[lower,F,upper,Fp] = selected.offset_hz
        # Solve r_upper-r_lower=selected frequency on the connected F tree.
        rotations = {edges[0][0]: 0.0}
        while len(rotations) < len({node for l,u,_ in edges for node in (l,u)}):
            count = len(rotations)
            for l, u, frequency in edges:
                if l in rotations:
                    rotations[u] = rotations[l]+frequency
                elif u in rotations:
                    rotations[l] = rotations[u]-frequency
            if len(rotations) == count:
                raise ValueError("Disconnected transition graph")
        shifts.update(rotations)
    h = a.internal_hamiltonian(B_gauss, clock_quadratic=clock_quadratic)
    h -= Qobj(np.diag([2*np.pi*shifts.get((level,F), 0) for level,F,_ in a.STATES]))
    # Remove an irrelevant common offset per disconnected coherent pair.
    for lower, upper in PAIRS.values():
        indices = [n for n,s in enumerate(a.STATES) if s[0] in (lower,upper)]
        offset = h[indices[0],indices[0]].real
        h -= Qobj(np.diag([offset if n in indices else 0 for n in range(a.DIMENSION)]))
    for interaction in interactions:
        h += interaction
    collapses = a.collapse_operators()
    for beam in beams:
        if beam.linewidth_dephasing_s:
            upper = PAIRS[beam.line][1]
            projector = sum(a.ket(*s).proj() for s in a.STATES if s[0] == upper)
            collapses.append(np.sqrt(2*beam.linewidth_dephasing_s)*projector)
    # These selection-rule operators are sparse. Keeping them sparse avoids
    # dense 400x400 matrix products at every full-reference integration step.
    return FreeSpaceModel(h.to("csr"), [c.to("csr") for c in collapses],
                          [(op.to("csr"), beat) for op, beat in omitted], assignments)


def fluorescence_369_s(rho):
    """Total emitted P->S photons/s, before collection and detector losses."""
    return sum(rate*float(np.real(rho[a.INDEX[u],a.INDEX[u]]))
               for u,l,rate in a.decay_channels() if u[0] == "P" and l[0] == "S")
