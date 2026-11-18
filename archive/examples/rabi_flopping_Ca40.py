from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from qutip import Qobj, basis, mesolve, qeye, destroy, tensor
from scipy import constants as consts
from atomic_physics.ions.ca40 import Ca40
from atomic_physics.core import Atom

# Paper mapping (examples/rabi_hamiltonian.qmd):
# - Specializes the three-level Hamiltonian (S1/2 ≡ |g>, P1/2 ≡ |e>, D3/2 ≡ |s>)
#   to the 8 Zeeman-resolved sublevels of 40Ca+ with polarization selection rules.
# - H_0: rotating-frame detunings (Δ_driv, Δ_rep) and Zeeman shifts g_J μ_B B / ℏ.
# - H_drive (397 nm) and H_rep (866 nm): coherent couplings from normalized
#   dipole matrices (Clebsch–Gordan geometry) for a chosen polarization q.
# - Optional cavity: (Δ_c - Δ_rep) a^† a and JC coupling on D↔P.
# - Collapse: spontaneous emission P→S and P→D, and cavity damping κ (thermal n_th).



# This is used to build the sigmas in equations 2.4.5 , 2.4.6, 2.4.10
def _projector_for_states(dim: int, indices: np.ndarray) -> Qobj:
    """Projector onto a set of bare basis states using QuTiP operators."""
    proj = None
    for i in indices:
        ket_i = basis(dim, int(i))
        op_i = ket_i * ket_i.dag()
        proj = op_i if proj is None else proj + op_i
    if proj is None:
        # Empty set: return zero operator
        proj = Qobj(np.zeros((dim, dim), dtype=complex))
    return proj


# This function is used to compute dipole moments and H_sp / H_dp 
def _coupling_from_epole(
    ion: Atom,
    lower_states: np.ndarray,
    upper_states: np.ndarray,
    Omega: float,
    polarization_q: int,
) -> Qobj:
    
    """
    Build the coherent coupling operator between lower and upper manifolds
    from electric-dipole amplitudes returned by atomic_physics.

    - Omega: Rabi frequency (rad/s) prefactor from your drive, applied as
      (Omega/2) * (epsilon Â· d_norm) where d_norm encodes the CGC geometry.
    - polarization_q: +1 (sigma+), 0 (pi), -1 (sigma-). Enforces Mu - Ml = q.

    We normalize the electric-dipole amplitudes to remove absolute rates so that
    the user-specified Omega sets the overall scale, while the relative weights
    follow the Clebschâ€“Gordan geometry.
    """
    
    # Paper mapping: contributes H_drive (S↔P) or H_rep (D↔P) depending on manifolds.
    dim = ion.num_states
    epole = ion.get_electric_multipoles()  # (lower, upper)

    # Total spontaneous rate for each upper state (sum over lowers)
    total_rates = np.sum(np.abs(epole) ** 2, axis=0)

    Mu = ion.M[upper_states]
    Ml = ion.M[lower_states]

    op = None
    for l in lower_states:
        Ml_val = ion.M[l]
        for u in upper_states:
            Mu_val = ion.M[u]
            if (Mu_val - Ml_val) != polarization_q:
                continue
            Gamma_u = float(total_rates[u])
            if Gamma_u <= 0.0:
                continue
            d_norm = epole[l, u] / np.sqrt(Gamma_u)
            amp = 0.5 * Omega * d_norm
            # |u><l|
            term = amp * (basis(dim, int(u)) * basis(dim, int(l)).dag())
            op = term if op is None else op + term
    if op is None:
        op = Qobj(np.zeros((dim, dim), dtype=complex))
    return op


def _g_J_from_factory(factory, level) -> float:
    for ld in factory.level_data:
        if ld.level == level:
            return float(ld.g_J)
    raise ValueError("g_J not found for level")


def build_ca40_hamiltonian(
    B_T: float,
    Delta_driv: float,
    Delta_rep: float,
    Omega_driv: float,
    Omega_rep: float,
    Delta_c: float = 0.0,
    g_cav: float = 0.0,
    kappa: float = 0.0,
    n_th: float = 0.0,
    N_cav: int = 0,
    pol_cav: int = -1,
    pol_397: int = 0,
    pol_866: int = -1,
) -> tuple[Qobj, list[Qobj], dict[str, Qobj], Atom]:
    """
    8-level 40Ca+ Lambda system (S1/2, P1/2, D3/2) Hamiltonian.

    - B_T: magnetic field in Tesla.
    - intensity_*_s: s = I/Isat for 397 and 866 nm lasers.
    - detuning_*: laser detunings (rad/s) from the transition centres-of-gravity.
    - pol_*: +1 (sigma+), 0 (pi), -1 (sigma-).
    """
    # Paper mapping: H = H_0 + H_drive + H_rep (+ H_coupling if cavity enabled)
    factory = Ca40.filter_levels(level_filter=(Ca40.S12, Ca40.P12, Ca40.D32))
    ion = factory(magnetic_field=B_T)
    dim = ion.num_states

    s_slice = ion.get_slice_for_level(Ca40.S12)
    p_slice = ion.get_slice_for_level(Ca40.P12)
    d_slice = ion.get_slice_for_level(Ca40.D32)

    P_g = _projector_for_states(dim, np.arange(s_slice.start, s_slice.stop))
    P_e = _projector_for_states(dim, np.arange(p_slice.start, p_slice.stop))
    P_s = _projector_for_states(dim, np.arange(d_slice.start, d_slice.stop))

    # H_0 diagonal terms (rotating frame): detunings and Zeeman shifts in S and D
    gS = _g_J_from_factory(factory, Ca40.S12)
    gD = _g_J_from_factory(factory, Ca40.D32)
    muB_over_hbar = consts.physical_constants['Bohr magneton'][0] / consts.hbar

    # Build diagonal H_0 using QuTiP projectors rather than raw matrices
    H_rot = Qobj(np.zeros((dim, dim), dtype=complex))
    # S manifold detuning + Zeeman
    for i in range(s_slice.start, s_slice.stop):
        mJ = ion.M[i]
        Ei = Delta_driv + mJ * gS * muB_over_hbar * B_T
        H_rot += Ei * (basis(dim, i) * basis(dim, i).dag())
    # D manifold detuning + Zeeman
    for i in range(d_slice.start, d_slice.stop):
        mJ = ion.M[i]
        Ei = Delta_rep + mJ * gD * muB_over_hbar * B_T
        H_rot += Ei * (basis(dim, i) * basis(dim, i).dag())

    # H_drive (397 nm) and H_rep (866 nm) coherent couplings
    g_states = np.arange(s_slice.start, s_slice.stop)
    e_states = np.arange(p_slice.start, p_slice.stop)
    d_states = np.arange(d_slice.start, d_slice.stop)

    Omega_ge = _coupling_from_epole(
        ion, lower_states=g_states, upper_states=e_states,
        Omega=Omega_driv, polarization_q=pol_397
    )
    Omega_se = _coupling_from_epole(
        ion, lower_states=d_states, upper_states=e_states,
        Omega=Omega_rep, polarization_q=pol_866
    )

    # Omega_ge/Omega_se include the conventional 1/2 prefactor (see helper).
    # Paper mapping: H = H_0 + H_drive + H_rep here (atom-only Hamiltonian part)
    H_atomic = H_rot + (Omega_ge + Omega_ge.dag() + Omega_se + Omega_se.dag())

    # Spontaneous emission P1/2 -> S1/2 and P1/2 -> D3/2
    epole = ion.get_electric_multipoles()
    c_ops: list[Qobj] = []

    # Decide if we include a cavity mode: N_cav > 0 enables the tensor product space.
    if N_cav and (N_cav > 0):
        I_atom = qeye(dim)
        I_cav = qeye(N_cav)
        a = destroy(N_cav)

        # Lift atomic pieces to the composite space
        H = tensor(I_cav, H_atomic)
        # Free cavity Hamiltonian (rotating frame): (Δ_c - Δ_rep) a^† a

        # Free cavity Hamiltonian: ħ(Δ_c - Δ_rep) a† a
        H += tensor((Delta_c - Delta_rep) * (a.dag() * a), I_atom)

        # Cavity–atom coupling on the D<->P transition using normalized geometry
        # Build a normalized operator for selected polarization q=pol_cav
        def make_norm_matrix(lower_states, upper_states, q):
            """Build normalized |upper><lower| operator with Δm = q using QuTiP ops."""
            total_rates = np.sum(np.abs(epole) ** 2, axis=0)
            op = None
            for l in lower_states:
                Ml_val = ion.M[l]
                for u in upper_states:
                    Mu_val = ion.M[u]
                    if (Mu_val - Ml_val) != q:
                        continue
                    Gu = float(total_rates[u])
                    if Gu <= 0:
                        continue
                    amp = epole[l, u] / np.sqrt(Gu)
                    term = amp * (basis(dim, int(u)) * basis(dim, int(l)).dag())
                    op = term if op is None else op + term
            if op is None:
                op = Qobj(np.zeros((dim, dim), dtype=complex))
            return op

        M_PD = make_norm_matrix(d_states, e_states, pol_cav)  # |P><D|
        # Jaynes–Cummings type: g (a σ_PD + a† σ_DP)
        # Jaynes–Cummings coupling on D↔P: g (a σ_es + a^† σ_se)
        if abs(g_cav) > 0:
            H += g_cav * (tensor(a, M_PD) + tensor(a.dag(), M_PD.dag()))

        # Atomic spontaneous emission collapse ops lifted to composite space
        # Atomic spontaneous emission collapse ops (P→S and P→D), lifted to atom⊗cavity
        for u in e_states:
            for l in np.r_[g_states, d_states]:
                gamma_ul = float(np.abs(epole[l, u]) ** 2)
                if gamma_ul > 0.0:
                    jump = np.sqrt(gamma_ul) * (basis(dim, int(l)) * basis(dim, int(u)).dag())
                    c_ops.append(tensor(I_cav, jump))

        # Cavity damping
        # Cavity damping (κ; include thermal n_th if specified)
        if kappa > 0:
            c_ops.append(np.sqrt(kappa * (1 + n_th)) * tensor(a, I_atom))
            if n_th > 0:
                c_ops.append(np.sqrt(kappa * n_th) * tensor(a.dag(), I_atom))

        # Observables (projectors) lifted
        projs = {"P_g": tensor(I_cav, P_g), "P_e": tensor(I_cav, P_e), "P_s": tensor(I_cav, P_s)}
        return H, c_ops, projs, ion
    else:
        # Atom-only case (original behavior)
        # Paper mapping: H = H_0 + H_drive + H_rep with atomic collapse only.
        H = H_atomic
        for u in e_states:
            for l in np.r_[g_states, d_states]:
                gamma_ul = float(np.abs(epole[l, u]) ** 2)
                if gamma_ul > 0.0:
                    jump = np.sqrt(gamma_ul) * (basis(dim, int(l)) * basis(dim, int(u)).dag())
                    c_ops.append(jump)

        projs = {"P_g": P_g, "P_e": P_e, "P_s": P_s}
        return H, c_ops, projs, ion


def _precompute_couplings(ion: Atom, pols: tuple[int, int]):
    """Precompute normalized coupling matrices for q = -1,0,+1 for both
    S<->P (397) and D<->P (866). Returns dictionaries mapping q to Qobj.
    Also returns collapse operators derived from epole.

    Paper mapping: builds σ_ge and σ_se with Δm = q selection rules used in
    H_drive and H_rep of the three-level Hamiltonian.
    """
    s_slice = ion.get_slice_for_level(Ca40.S12)
    p_slice = ion.get_slice_for_level(Ca40.P12)
    d_slice = ion.get_slice_for_level(Ca40.D32)

    g_states = np.arange(s_slice.start, s_slice.stop)
    e_states = np.arange(p_slice.start, p_slice.stop)
    d_states = np.arange(d_slice.start, d_slice.stop)

    epole = ion.get_electric_multipoles()
    total_rates = np.sum(np.abs(epole) ** 2, axis=0)

    def make_norm_matrix(lower_states, upper_states, q):
        op = None
        for l in lower_states:
            Ml_val = ion.M[l]
            for u in upper_states:
                Mu_val = ion.M[u]
                if (Mu_val - Ml_val) != q:
                    continue
                Gu = float(total_rates[u])
                if Gu <= 0:
                    continue
                amp = epole[l, u] / np.sqrt(Gu)
                term = amp * (basis(ion.num_states, int(u)) * basis(ion.num_states, int(l)).dag())
                op = term if op is None else op + term
        if op is None:
            op = Qobj(np.zeros((ion.num_states, ion.num_states), dtype=complex))
        return op

    ge = {q: make_norm_matrix(g_states, e_states, q) for q in (-1, 0, +1)}
    se = {q: make_norm_matrix(d_states, e_states, q) for q in (-1, 0, +1)}

    # Collapse operators depend only on epole (B-dependent) not on drive params
    c_ops = []
    for u in e_states:
        for l in np.r_[g_states, d_states]:
            gamma_ul = float(np.abs(epole[l, u]) ** 2)
            if gamma_ul > 0:
                jump = np.sqrt(gamma_ul) * (basis(ion.num_states, int(l)) * basis(ion.num_states, int(u)).dag())
                c_ops.append(jump)

    projs = {
        "P_g": _projector_for_states(ion.num_states, g_states),
        "P_e": _projector_for_states(ion.num_states, e_states),
        "P_s": _projector_for_states(ion.num_states, d_states),
    }
    return ge, se, c_ops, projs


def run_example():
    B_T = 5e-4
    Delta_driv = 0.0
    Delta_rep = 0.0
    Omega_driv = 2 * np.pi * 5e6  # 5 MHz
    Omega_rep = 2 * np.pi * 2e6   # 2 MHz

    H, c_ops, projs, ion = build_ca40_hamiltonian(
        B_T=B_T,
        Delta_driv=Delta_driv,
        Delta_rep=Delta_rep,
        Omega_driv=Omega_driv,
        Omega_rep=Omega_rep,
        pol_397=0,
        pol_866=-1,
    )

    s_slice = ion.get_slice_for_level(Ca40.S12)
    psi0 = basis(ion.num_states, s_slice.start)  # highest-energy S state

    tlist = np.linspace(0, 50e-6, 400)
    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[projs["P_g"], projs["P_e"], projs["P_s"]])
    Pg, Pe, Ps = (np.array(result.expect[i]) for i in range(3))

    plt.figure(figsize=(8, 5))
    plt.plot(tlist * 1e6, Pe, label="P1/2 population")
    plt.plot(tlist * 1e6, Pg, label="S1/2 population")
    plt.plot(tlist * 1e6, Ps, label="D3/2 population")
    plt.xlabel("Time (us)")
    plt.ylabel("Population")
    plt.title("40Ca+ Lambda system using atomic_physics parameters")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
        def interactive_ca40():
            params = dict(
                B_G=5.0,
                Delta_driv_MHz=0.0,
                Delta_rep_MHz=0.0,
                Omega_driv_MHz=5.0,
                Omega_rep_MHz=2.0,
                # Cavity controls (single mode)
                Delta_c_MHz=0.0,
                g_cav_MHz=0.0,
                kappa_MHz=0.0,
                N_cav=3,
                pol_cav=-1,
                pol397=0,     # +1: sigma+, 0: pi, -1: sigma-
                pol866=-1,
                tmax_us=50.0,
                steps=400,
                init_M=+0.5,
            )

            # Cache for B/polarization-dependent pieces
            cache = {
                'B_T': None,
                'pol397': None,
                'pol866': None,
                'ion': None,
                'ge': None,
                'se': None,
                'c_ops': None,
                'projs': None,
            }

            def ensure_cache():
                B_T = params['B_G'] * 1e-4
                if (cache['B_T'] != B_T) or (cache['pol397'] is None):
                    factory = Ca40.filter_levels(level_filter=(Ca40.S12, Ca40.P12, Ca40.D32))
                    ion = factory(magnetic_field=B_T)
                    ge, se, c_ops, projs = _precompute_couplings(ion, (params['pol397'], params['pol866']))
                    cache.update({'B_T': B_T, 'pol397': params['pol397'], 'pol866': params['pol866'],
                                  'ion': ion, 'ge': ge, 'se': se, 'c_ops': c_ops, 'projs': projs})
                elif (cache['pol397'] != params['pol397']) or (cache['pol866'] != params['pol866']):
                    # Rebuild only polarization-composed maps from existing ion
                    ion = cache['ion']
                    ge, se, c_ops, projs = _precompute_couplings(ion, (params['pol397'], params['pol866']))
                    cache.update({'pol397': params['pol397'], 'pol866': params['pol866'],
                                  'ge': ge, 'se': se, 'c_ops': c_ops, 'projs': projs})

            def _estimate_fast_rate(ion: Atom, qg: int, qd: int, Og: float, Od: float, B_T: float, g: float, Dc: float) -> float:
                """Rough upper bound on the fastest relevant angular frequency (rad/s)
                to pick an appropriate time step. Uses:
                - Rabi rates (0.5 * Omega * max|d_norm|)
                - Zeeman splittings (|g_J mu_B B / ħ|)
                - Largest spontaneous rate from P states
                - Cavity frequencies (|g|, |Δ_c|)
                """
                # Rabi part: normalized matrices have |d_norm|<=1; use max element
                max_ge = float(np.max(np.abs(cache['ge'][qg].full()))) if cache['ge'] else 0.0
                max_se = float(np.max(np.abs(cache['se'][qd].full()))) if cache['se'] else 0.0
                rabi_max = 0.5 * max(Og * max_ge, Od * max_se, 1e-12)

                # Zeeman part
                factory = Ca40.filter_levels(level_filter=(Ca40.S12, Ca40.P12, Ca40.D32))
                gS = _g_J_from_factory(factory, Ca40.S12)
                gD = _g_J_from_factory(factory, Ca40.D32)
                muB_over_hbar = consts.physical_constants['Bohr magneton'][0] / consts.hbar
                zeeman_max = max(abs(gS * muB_over_hbar * B_T), abs(gD * muB_over_hbar * B_T))

                # Spontaneous part (max A coefficient)
                epole = cache['ion'].get_electric_multipoles()
                gamma_max = float(np.max(np.sum(np.abs(epole) ** 2, axis=0)))

                return max(rabi_max, zeeman_max, gamma_max, abs(g), abs(Dc), 1.0)

            def build_and_solve(steps_override: int | None = None, samples_per_cycle: int = 40):
                ensure_cache()
                ion = cache['ion']
                qg = params['pol397']
                qd = params['pol866']
                Og = params['Omega_driv_MHz'] * 2 * np.pi * 1e6
                Od = params['Omega_rep_MHz'] * 2 * np.pi * 1e6
                g = params['g_cav_MHz'] * 2 * np.pi * 1e6
                Dc = params['Delta_c_MHz'] * 2 * np.pi * 1e6

                # Build Hamiltonian and operators via the unified builder (handles cavity as needed)
                H, c_ops, projs, ion2 = build_ca40_hamiltonian(
                    B_T=cache['B_T'],
                    Delta_driv=params['Delta_driv_MHz'] * 2 * np.pi * 1e6,
                    Delta_rep=params['Delta_rep_MHz'] * 2 * np.pi * 1e6,
                    Omega_driv=Og,
                    Omega_rep=Od,
                    Delta_c=Dc,
                    g_cav=g,
                    kappa=params['kappa_MHz'] * 2 * np.pi * 1e6,
                    n_th=0.0,
                    N_cav=params['N_cav'] if (g != 0 or params['kappa_MHz'] > 0 or Dc != 0) else 0,
                    pol_cav=params['pol_cav'],
                    pol_397=qg,
                    pol_866=qd,
                )

                # Initial state
                s_slice = ion2.get_slice_for_level(Ca40.S12)
                s_states = ion2.get_states_for_M(Ca40.S12, M=params['init_M'])
                if len(s_states) == 0:
                    idx0 = s_slice.start
                else:
                    idx0 = int(s_states[0])
                if (params['N_cav'] if (g != 0 or params['kappa_MHz'] > 0 or Dc != 0) else 0) > 0:
                    psi0 = tensor(basis(params['N_cav'], 0), basis(ion2.num_states, idx0))
                else:
                    psi0 = basis(ion2.num_states, idx0)

                if steps_override is not None:
                    steps = steps_override
                else:
                    # Auto-choose steps to resolve the fastest oscillation
                    fastest = _estimate_fast_rate(ion, qg, qd, Og, Od, cache['B_T'], g, Dc)
                    tmax_s = params['tmax_us'] * 1e-6
                    # dt = T/(samples_per_cycle) = 2π/(ω * samples)
                    dt = (2 * np.pi) / (fastest * max(samples_per_cycle, 10))
                    steps = int(max(params['steps'], np.ceil(tmax_s / dt)))
                    steps = int(min(steps, 20000))  # keep interactive responsive
                tlist = np.linspace(0, params['tmax_us'] * 1e-6, steps)
                result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[projs['P_g'], projs['P_e'], projs['P_s']])
                return tlist, result

            t, res = build_and_solve()
            Pg, Pe, Ps = (np.array(res.expect[i]) for i in range(3))

            fig, ax = plt.subplots(figsize=(9, 6))
            # Leave clean band at bottom for a tidy two-column control panel
            plt.subplots_adjust(left=0.10, right=0.95, bottom=0.40)
            l_e, = ax.plot(t * 1e6, Pe, label="P1/2 population")
            l_g, = ax.plot(t * 1e6, Pg, label="S1/2 population")
            l_s, = ax.plot(t * 1e6, Ps, label="D3/2 population")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Population")
            ax.set_ylim(0, 1.05)
            ax.set_title("40Ca+ Lambda system")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # Controls: two aligned columns of sliders + radio panels at right
            axcolor = 'lightgoldenrodyellow'
            row_h, gap = 0.03, 0.007
            y0 = 0.34
            xL, wC = 0.10, 0.35
            xR = 0.55

            def row_y(i):
                return y0 - i * (row_h + gap)

            ax_BG   = plt.axes([xL, row_y(0), wC, row_h], facecolor=axcolor)
            ax_tmax = plt.axes([xR, row_y(0), wC, row_h], facecolor=axcolor)
            ax_ddrv = plt.axes([xL, row_y(1), wC, row_h], facecolor=axcolor)
            ax_odrv = plt.axes([xR, row_y(1), wC, row_h], facecolor=axcolor)
            ax_drep = plt.axes([xL, row_y(2), wC, row_h], facecolor=axcolor)
            ax_orep = plt.axes([xR, row_y(2), wC, row_h], facecolor=axcolor)
            ax_dcav = plt.axes([xL, row_y(3), wC, row_h], facecolor=axcolor)
            ax_gcav = plt.axes([xR, row_y(3), wC, row_h], facecolor=axcolor)
            ax_kcav = plt.axes([xR, row_y(4), wC, row_h], facecolor=axcolor)

            s_BG = Slider(ax_BG, 'B (G)', 0.0, 50.0, valinit=params['B_G'])
            s_tmax = Slider(ax_tmax, 't_max (us)', 5.0, 200.0, valinit=params['tmax_us'])
            s_ddrv = Slider(ax_ddrv, 'Δ_driv (MHz)', -50.0, 50.0, valinit=params['Delta_driv_MHz'])
            s_odrv = Slider(ax_odrv, 'Ω_driv (MHz)', 0.0, 50.0, valinit=params['Omega_driv_MHz'])
            s_drep = Slider(ax_drep, 'Δ_rep (MHz)', -50.0, 50.0, valinit=params['Delta_rep_MHz'])
            s_orep = Slider(ax_orep, 'Ω_rep (MHz)', 0.0, 50.0, valinit=params['Omega_rep_MHz'])
            s_dcav = Slider(ax_dcav, 'Δ_c (MHz)', -200.0, 200.0, valinit=params['Delta_c_MHz'])
            s_gcav = Slider(ax_gcav, 'g_cav (MHz)', 0.0, 50.0, valinit=params['g_cav_MHz'])
            s_kcav = Slider(ax_kcav, 'κ (MHz)', 0.0, 50.0, valinit=params['kappa_MHz'])

            # Radio buttons for polarizations and initial M
            ax_pol397 = plt.axes([0.84, 0.30, 0.12, 0.09], facecolor=axcolor)
            ax_pol866 = plt.axes([0.84, 0.19, 0.12, 0.09], facecolor=axcolor)
            ax_polcav = plt.axes([0.84, 0.08, 0.12, 0.09], facecolor=axcolor)
            ax_initM  = plt.axes([0.84, 0.345, 0.12, 0.04], facecolor=axcolor)

            labels = ['sigma-', 'pi', 'sigma+']
            to_q = {'sigma-': -1, 'pi': 0, 'sigma+': +1}
            rb_397 = RadioButtons(ax_pol397, labels, active=1)
            rb_866 = RadioButtons(ax_pol866, labels, active=0)
            rb_init = RadioButtons(ax_initM, ['M=-1/2', 'M=+1/2'], active=1)
            rb_cav = RadioButtons(ax_polcav, labels, active=0)

            # Debounced updates for smoother interaction
            timer = {'obj': None}

            def do_recompute(high_res: bool):
                # Low-res preview: modest samples_per_cycle; High-res: finer
                if high_res:
                    t, res = build_and_solve(None, samples_per_cycle=40)
                else:
                    t, res = build_and_solve(steps_override=120, samples_per_cycle=15)
                Pg, Pe, Ps = (np.array(res.expect[i]) for i in range(3))
                l_e.set_data(t * 1e6, Pe)
                l_g.set_data(t * 1e6, Pg)
                l_s.set_data(t * 1e6, Ps)
                ax.set_xlim(t[0] * 1e6, t[-1] * 1e6)
                fig.canvas.draw_idle()

            def update(_=None):
                params['B_G'] = s_BG.val
                params['Delta_driv_MHz'] = s_ddrv.val
                params['Delta_rep_MHz'] = s_drep.val
                params['Omega_driv_MHz'] = s_odrv.val
                params['Omega_rep_MHz'] = s_orep.val
                params['tmax_us'] = s_tmax.val
                params['Delta_c_MHz'] = s_dcav.val
                params['g_cav_MHz'] = s_gcav.val
                params['kappa_MHz'] = s_kcav.val
                params['pol397'] = to_q[rb_397.value_selected]
                params['pol866'] = to_q[rb_866.value_selected]
                params['pol_cav'] = to_q[rb_cav.value_selected]
                params['init_M'] = +0.5 if rb_init.value_selected.endswith('+1/2') else -0.5

                # debounce: quick preview; full solve after short delay
                if timer['obj'] is not None:
                    timer['obj'].stop()
                do_recompute(high_res=False)
                timer['obj'] = fig.canvas.new_timer(interval=200)
                timer['obj'].add_callback(lambda: do_recompute(high_res=True))
                timer['obj'].start()

            for s in (s_BG, s_ddrv, s_drep, s_odrv, s_orep, s_tmax, s_dcav, s_gcav, s_kcav):
                s.on_changed(update)
            rb_397.on_clicked(update)
            rb_866.on_clicked(update)
            rb_cav.on_clicked(update)
            rb_init.on_clicked(update)

            reset_ax = plt.axes([0.44, 0.045, 0.12, 0.04])
            b_reset = Button(reset_ax, 'Reset')
            def do_reset(_):
                s_BG.reset(); s_ddrv.reset(); s_drep.reset(); s_odrv.reset(); s_orep.reset(); s_tmax.reset(); s_dcav.reset(); s_gcav.reset(); s_kcav.reset()
                rb_397.set_active(1); rb_866.set_active(0); rb_cav.set_active(0); rb_init.set_active(1)
            b_reset.on_clicked(do_reset)

            plt.show()

        interactive_ca40()
