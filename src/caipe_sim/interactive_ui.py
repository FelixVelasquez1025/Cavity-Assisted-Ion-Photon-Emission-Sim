"""
Interactive Yb171+ UI built on the physics utilities in caipe_sim.
"""

from __future__ import annotations

import math
import time
from functools import wraps
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import qutip
from qutip import Qobj, basis, destroy, qeye, tensor, mesolve, expect
from qutip import correlation_2op_1t
from qutip import correlation_3op_1t

# Optional JAX/QuTiP-JAX backend
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    import qutip_jax

    HAVE_JAX = True
except Exception:
    HAVE_JAX = False
    jnp = None


USE_JAX_BACKEND = True

from .hamiltonian_utils import (
    HF_BASIS_DIM,
    HBAR,
    MU_B,
    add_bare_terms_to_systems,
    eq_2_4_5_H_S,
    eq_2_4_5_H_S_hf,
    eq_2_4_6_H_D,
    eq_2_4_6_H_D_hf,
    g_d,
    g_s,
    hf_state_index,
    precompute_e1_coeffs,
    precompute_e1_coeffs_hf,
    precompute_lowering_tensors,
    precompute_lowering_tensors_hf,
    projector_for_indices,
    projector_for_level_hf,
    state_index,
    transition_sigma_components,
    transition_sigma_components_hf,
    H_interaction_with_cavity,
    H_interaction_with_cavity_hf,
    H_laser_systems_from_E0,
    H_laser_systems_from_E0_hf,
)


# Timing helper ---------------------------------------------------------------
FUNCTION_TIMINGS: Dict[str, Dict[str, float | int]] = {}


def _record_timing(name: str, elapsed: float) -> None:
    stats = FUNCTION_TIMINGS.setdefault(name, {"count": 0, "total": 0.0, "max": 0.0})
    stats["count"] += 1
    stats["total"] += elapsed
    if elapsed > stats["max"]:
        stats["max"] = elapsed


def timed(func=None, *, label: str | None = None):
    def decorator(fn):
        name = label or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _record_timing(name, time.perf_counter() - start)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


def enable_jax_backend_if_available() -> None:
    """Enable JAX backend if available and requested."""
    if USE_JAX_BACKEND and HAVE_JAX:
        qutip_jax.set_as_default()


def solver_options(store_states: bool = False) -> dict:
    if USE_JAX_BACKEND and HAVE_JAX:
        return {
            "store_states": store_states,
            "progress_bar": None,
            "method": "diffrax",
            "normalize_output": False,  # avoid CPU-only Schur steps
        }
    else:
        return {
            "store_states": store_states,
            "progress_bar": None,
            "method": "bdf",
        }


def _record_results_csv(params: dict, tlist: np.ndarray, res) -> None:
    """Append run results (final populations) to a CSV under src/caipe_sim."""
    results_path = Path(__file__).resolve().parent / "results.csv"
    header = [
        "timestamp",
        "is_bosonic",
        "system",
        "B_G",
        "delta_drive_MHz",
        "delta_rep_MHz",
        "delta_c_MHz",
        "g_cav_MHz",
        "amp_driv_MHz",
        "amp_rep_MHz",
        "tmax_us",
        "density",
        "gamma_SP",
        "gamma_DP",
        "gamma_SD",
        "kappa_MHz",
        "final_P_S",
        "final_P_P",
        "final_P_D",
    ]
    final_P_S = float(np.array(res.expect[0])[-1])
    final_P_P = float(np.array(res.expect[1])[-1])
    final_P_D = float(np.array(res.expect[2])[-1])
    row = [
        datetime.utcnow().isoformat(),
        bool(params.get("is_bosonic", False)),
        params.get("system", ""),
        params.get("B_G", ""),
        params.get("delta_drive_MHz", ""),
        params.get("delta_rep_MHz", ""),
        params.get("delta_c_MHz", ""),
        params.get("g_cav_MHz", ""),
        params.get("amp_driv_MHz", ""),
        params.get("amp_rep_MHz", ""),
        params.get("tmax_us", ""),
        params.get("density", ""),
        params.get("gamma_SP", ""),
        params.get("gamma_DP", ""),
        params.get("gamma_SD", ""),
        params.get("kappa_MHz", ""),
        final_P_S,
        final_P_P,
        final_P_D,
    ]
    write_header = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _set_y_limits(ax, data: np.ndarray | None, default: tuple[float, float], padding: float = 0.1) -> None:
    if data is None or len(data) == 0:
        ax.set_ylim(*default)
        return
    low = float(np.nanmin(data))
    high = float(np.nanmax(data))
    if not np.isfinite(low) or not np.isfinite(high):
        ax.set_ylim(*default)
        return
    span = high - low
    pad = max(abs(high) * padding, abs(low) * padding, span * padding, 1e-6)
    ax.set_ylim(low - pad, high + pad)


# Collapse operators ----------------------------------------------------------
def build_collapse_operators(
    params: Dict[str, float],
    lowering_tensors: Dict[str, List[Tuple[Qobj, float]]],
    projector_tensors: Dict[str, Qobj],
    cavity_loss_ops: List[Qobj],
) -> List[Qobj]:
    c_ops: List[Qobj] = []
    gamma_sp = 2 * math.pi * params["gamma_SP"] * 1e6
    gamma_dp = 2 * math.pi * params["gamma_DP"] * 1e6
    gamma_map = {"SP": gamma_sp, "DP": gamma_dp}
    for pair, gamma in gamma_map.items():
        if gamma <= 0:
            continue
        for op, weight in lowering_tensors[pair]:
            if weight > 0:
                c_ops.append(math.sqrt(2 * gamma) * weight * op)

    kappa = 2 * math.pi * params["kappa_MHz"] * 1e6
    if kappa > 0:
        for loss_op in cavity_loss_ops:
            c_ops.append(math.sqrt(2 * kappa) * loss_op)

    delta_S = 2 * math.pi * params["gamma_SD"] * 1e6
    if delta_S > 0:
        c_ops.append(math.sqrt(2 * delta_S) * projector_tensors["S"])
        c_ops.append(math.sqrt(2 * delta_S) * projector_tensors["D"])
    return c_ops


# First-order coherence -------------------------------------------------------
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
    if c_ops is None:
        c_ops = []
    a_dag = a_field.dag()

    def _zeros_like(arr: np.ndarray) -> np.ndarray:
        return np.zeros_like(arr)

    def empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zeros = _zeros_like(taulist)
        return zeros, zeros, zeros

    if taulist.size == 0:
        return empty()

    rho_final: Qobj | None = None
    if states:
        last = states[-1]
        rho_final = last if last.isoper else (last * last.dag())
    if rho_final is None:
        sol = mesolve(H, psi0, tlist, c_ops=c_ops, options=solver_options(store_states=True))
        if sol.states:
            last = sol.states[-1]
            rho_final = last if last.isoper else (last * last.dag())
    if rho_final is None:
        return empty()

    # Correlation routines expect CPU-friendly data; convert if JAX is active.
    if USE_JAX_BACKEND and HAVE_JAX:
        H = H.to("csr")
        rho_final = rho_final.to("csr")

    number_op = a_dag * a_field
    intensity = float(np.real(expect(number_op, rho_final)))
    if abs(intensity) < 1e-18:
        return empty()

    corr = None
    if correlation_2op_1t is not None:
        try:
            corr = np.asarray(correlation_2op_1t(H, rho_final, taulist, c_ops, a_dag, a_field, "me"), dtype=complex)
        except Exception:
            corr = None
    if corr is None and correlation_3op_1t is not None:
        try:
            I_full = qeye(H.shape[0])
            corr = np.asarray(
                correlation_3op_1t(H, rho_final, taulist, c_ops, a_dag, a_field, I_full, "me"), dtype=complex
            )
        except Exception:
            corr = None
    if corr is None:
        return empty()

    G1_tau = np.real(corr)
    g1_tau_raw = corr / intensity
    tau_c = float(np.trapezoid(np.abs(g1_tau_raw) ** 2, x=taulist))
    g1_model = _zeros_like(taulist) if tau_c <= 0.0 else np.exp(-(taulist / tau_c) ** 2)
    return G1_tau, g1_model, _zeros_like(taulist)


# Interactive UI --------------------------------------------------------------
@timed()
def interactive_populations():
    # Enable GPU/JAX backend if available
    enable_jax_backend_if_available()

    params = dict(
        B_G=5.0,
        delta_drive_MHz=0.0,
        delta_rep_MHz=0.0,
        delta_c_MHz=0.0,
        g_S=g_s,
        g_D=g_d,
        gamma_SP=0.03,
        gamma_DP=0.034,
        gamma_SD=0.000136,
        kappa_MHz=0.5,
        tmax_us=50.0,
        density=8.0,
        system="left",
        is_bosonic=False,  # default to fermionic (hyperfine) basis
        g_cav_MHz=0.0,
        amp_driv_MHz=0.0,
        amp_rep_MHz=0.0,
        tau_max_us=0.5,
        n_tau=20,
    )

    use_bosonic = bool(params.get("is_bosonic", False))
    dim_atom = 8 if use_bosonic else HF_BASIS_DIM

    if use_bosonic:
        psi0_atom = basis(dim_atom, state_index("S", +0.5))
        P_S = projector_for_indices([state_index("S", -0.5), state_index("S", +0.5)], dim=dim_atom)
        P_P = projector_for_indices([state_index("P", -0.5), state_index("P", +0.5)], dim=dim_atom)
        P_D = projector_for_indices([state_index("D", -1.5), state_index("D", -0.5), state_index("D", +0.5), state_index("D", +1.5)], dim=dim_atom)
    else:
        psi0_atom = basis(dim_atom, hf_state_index("S", 0.0, 0.0))
        P_S = projector_for_level_hf("S", dim=dim_atom)
        P_P = projector_for_level_hf("P", dim=dim_atom)
        P_D = projector_for_level_hf("D", dim=dim_atom)

    N_a = 3
    N_b = 3
    Ia = qeye(N_a)
    Ib = qeye(N_b)
    I_atom = qeye(dim_atom)
    projector_tensors = {
        "S": tensor(Ia, Ib, P_S),
        "P": tensor(Ia, Ib, P_P),
        "D": tensor(Ia, Ib, P_D),
    }
    field_left = tensor(destroy(N_a), Ib, qeye(dim_atom))
    field_right = tensor(Ia, destroy(N_b), qeye(dim_atom))
    if use_bosonic:
        comps_sp = transition_sigma_components("SP", dim=dim_atom)
    else:
        comps_sp = transition_sigma_components_hf("SP", dim=dim_atom)
    sigma_left = tensor(Ia, Ib, comps_sp.get(+1, qeye(dim_atom) * 0))
    sigma_pi = tensor(Ia, Ib, comps_sp.get(0, qeye(dim_atom) * 0))
    sigma_right = tensor(Ia, Ib, comps_sp.get(-1, qeye(dim_atom) * 0))
    cavity_loss_ops = [
        tensor(destroy(N_a), Ib, I_atom),
        tensor(Ia, destroy(N_b), I_atom),
    ]
    lowering_tensors = precompute_lowering_tensors(N_a, N_b, dim_atom) if use_bosonic else precompute_lowering_tensors_hf(N_a, N_b, dim_atom)

    @timed(label="build_and_solve")
    def build_and_solve():
        use_bos = bool(params.get("is_bosonic", False))
        e1_coeffs = precompute_e1_coeffs(use_bos) if use_bos else precompute_e1_coeffs_hf()
        B_T = params["B_G"] * 1e-4
        ddrv = 2 * np.pi * params["delta_drive_MHz"] * 1e6
        drep = 2 * np.pi * params["delta_rep_MHz"] * 1e6
        E0_driv = 2 * np.pi * params["amp_driv_MHz"] * 1e6
        E0_rep = 2 * np.pi * params["amp_rep_MHz"] * 1e6
        if use_bos:
            H_spdp = H_laser_systems_from_E0(E0_driv, E0_rep, dim=dim_atom, e1_coeffs=e1_coeffs)
            H_S = eq_2_4_5_H_S(ddrv, B_T, params["g_S"], dim=dim_atom)
            H_D = eq_2_4_6_H_D(drep, B_T, params["g_D"], dim=dim_atom)
        else:
            H_spdp = H_laser_systems_from_E0_hf(E0_driv, E0_rep, dim=dim_atom, e1_coeffs_hf=e1_coeffs)
            H_S = eq_2_4_5_H_S_hf(ddrv, B_T, params["g_S"], dim=dim_atom)
            H_D = eq_2_4_6_H_D_hf(drep, B_T, params["g_D"], dim=dim_atom)

        steps = max(100, int(round(params["tmax_us"] * params["density"])))
        tlist = np.linspace(0, params["tmax_us"] * 1e-6, steps)

        g = 2 * np.pi * params.get("g_cav_MHz", 0.0) * 1e6
        dc = 2 * np.pi * params["delta_c_MHz"] * 1e6

        if params["system"] in ("left", "right"):
            H_lift = H_interaction_with_cavity(H_spdp, g=g, N_a=N_a, N_b=N_b) if use_bos else H_interaction_with_cavity_hf(H_spdp, g=g, N_a=N_a, N_b=N_b)
        else:
            H_lift = {name: tensor(Ia, Ib, H_atom) for name, H_atom in H_spdp.items()}
        H_full = add_bare_terms_to_systems(H_lift, H_S, H_D, dc, drep, N_a, N_b)
        pol = params["system"]
        H = H_full[pol]
        psi0 = tensor(basis(N_a, 0), basis(N_b, 0), psi0_atom)
        c_ops = build_collapse_operators(params, lowering_tensors, projector_tensors, cavity_loss_ops)
        res = mesolve(
            H,
            psi0,
            tlist,
            c_ops=c_ops,
            e_ops=[projector_tensors["S"], projector_tensors["P"], projector_tensors["D"]],
            options=solver_options(store_states=False),
        )

        tau_max = params["tau_max_us"] * 1e-6
        taulist = np.linspace(0.0, tau_max, int(params["n_tau"]))
        field = sigma_left if pol == "left" else sigma_right if pol == "right" else sigma_pi
        _G1, g1_tau, _ = first_order_coherence(
            H,
            psi0,
            tlist,
            c_ops,
            field,
            taulist,
            states=res.states,
        )
        # Append results to CSV for later analysis
        try:
            _record_results_csv(params, tlist, res)
        except Exception:
            pass
        return tlist, res, taulist, g1_tau, None

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.60)

    axcolor = "lightgoldenrodyellow"
    row_h, gap = 0.035, 0.007
    y0 = 0.46
    xL, wC = 0.10, 0.35
    xR = 0.55

    def row_y(i):
        return y0 - i * (row_h + gap)

    ax_BG = plt.axes([xL, row_y(0), wC, row_h], facecolor=axcolor)
    ax_tmax = plt.axes([xR, row_y(0), wC, row_h], facecolor=axcolor)
    ax_ddrv = plt.axes([xL, row_y(1), wC, row_h], facecolor=axcolor)
    ax_drep = plt.axes([xR, row_y(1), wC, row_h], facecolor=axcolor)
    ax_dcav = plt.axes([xL, row_y(2), wC, row_h], facecolor=axcolor)
    ax_dense = plt.axes([xR, row_y(2), wC, row_h], facecolor=axcolor)
    ax_gcav = plt.axes([xL, row_y(3), wC, row_h], facecolor=axcolor)
    ax_ampdrv = plt.axes([xL, row_y(4), wC, row_h], facecolor=axcolor)
    ax_amprep = plt.axes([xR, row_y(4), wC, row_h], facecolor=axcolor)
    ax_taumax = plt.axes([xL, row_y(5), wC, row_h], facecolor=axcolor)
    ax_gammasp = plt.axes([xR, row_y(5), wC, row_h], facecolor=axcolor)
    ax_gammadp = plt.axes([xL, row_y(6), wC, row_h], facecolor=axcolor)
    ax_gammasd = plt.axes([xR, row_y(6), wC, row_h], facecolor=axcolor)
    ax_kappa = plt.axes([xL, row_y(7), wC, row_h], facecolor=axcolor)

    s_BG = Slider(ax_BG, "B (G)", 0.0, 50.0, valinit=params["B_G"])
    s_tmax = Slider(ax_tmax, "t_max (us)", 5.0, 200.0, valinit=params["tmax_us"])
    s_ddrv = Slider(ax_ddrv, "Δ_driv (MHz)", -3.0, 3.0, valinit=params["delta_drive_MHz"])
    s_drep = Slider(ax_drep, "Δ_rep (MHz)", -3.0, 3.0, valinit=params["delta_rep_MHz"])
    s_dcav = Slider(ax_dcav, "Δ_c (MHz)", -3.0, 3.0, valinit=params["delta_c_MHz"])
    s_dense = Slider(ax_dense, "samples/us", 1.0, 100.0, valinit=params["density"])
    s_gcav = Slider(ax_gcav, "g_cav (MHz)", 0.0, 50.0, valinit=params.get("g_cav_MHz", 0.0))
    s_ampdrv = Slider(ax_ampdrv, "Ω_drive (MHz)", -30.0, 30.0, valinit=params["amp_driv_MHz"])
    s_amprep = Slider(ax_amprep, "Ω_rep (MHz)", -30.0, 30.0, valinit=params["amp_rep_MHz"])
    s_taumax = Slider(ax_taumax, "τ_max (us)", 0.05, 10.0, valinit=params["tau_max_us"])
    s_gamma_sp = Slider(ax_gammasp, "γ_SP", 0.0, 1.0, valinit=params["gamma_SP"])
    s_gamma_dp = Slider(ax_gammadp, "γ_DP", 0.0, 1.0, valinit=params["gamma_DP"])
    s_gamma_sd = Slider(ax_gammasd, "γ_SD", 0.0, 1.0, valinit=params["gamma_SD"])
    s_kappa = Slider(ax_kappa, "κ (MHz)", 0.0, 1.0, valinit=params["kappa_MHz"])

    ax_sys = plt.axes([0.82, 0.08, 0.15, 0.12], facecolor=axcolor)
    rb_sys = RadioButtons(ax_sys, ("left", "pi", "right"), active=0)

    t, res, taulist, g1_tau, _ = build_and_solve()
    Pg, Pe, Ps = (np.array(res.expect[i]) for i in range(3))
    ln_e, = ax.plot(t * 1e6, Pe, label="P1/2 population")
    ln_g, = ax.plot(t * 1e6, Pg, label="S1/2 population")
    ln_s, = ax.plot(t * 1e6, Ps, label="D3/2 population")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Population")
    ax.set_ylim(0, 1.05)
    ax.set_title("Yb171+ populations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig_g1, ax_g1 = plt.subplots(figsize=(8, 3))
    if g1_tau is not None:
        ln_g1, = ax_g1.plot(taulist * 1e6, g1_tau, label="g^(1)(τ)")
        ax_g1.set_xlim(taulist[0] * 1e6, taulist[-1] * 1e6)
    else:
        ln_g1, = ax_g1.plot([], [], label="g^(1)(τ)")
    ax_g1.set_xlabel("Delay τ (us)")
    ax_g1.set_ylabel("g^(1)")
    _set_y_limits(ax_g1, g1_tau, (0.0, 1.0))
    ax_g1.grid(True, alpha=0.3)
    ax_g1.legend(loc="upper right")

    def recompute(_=None):
        params["B_G"] = s_BG.val
        params["tmax_us"] = s_tmax.val
        params["density"] = float(s_dense.val)
        params["delta_drive_MHz"] = s_ddrv.val
        params["delta_rep_MHz"] = s_drep.val
        params["delta_c_MHz"] = s_dcav.val
        params["system"] = rb_sys.value_selected
        params["amp_driv_MHz"] = s_ampdrv.val
        params["amp_rep_MHz"] = s_amprep.val
        params["tau_max_us"] = s_taumax.val
        params["gamma_SP"] = s_gamma_sp.val
        params["gamma_DP"] = s_gamma_dp.val
        params["gamma_SD"] = s_gamma_sd.val
        params["kappa_MHz"] = s_kappa.val
        t2, res2, tau2, g1_2, _ = build_and_solve()
        ln_e.set_data(t2 * 1e6, np.array(res2.expect[1]))
        ln_g.set_data(t2 * 1e6, np.array(res2.expect[0]))
        ln_s.set_data(t2 * 1e6, np.array(res2.expect[2]))
        ax.set_xlim(t2[0] * 1e6, t2[-1] * 1e6)
        if g1_2 is not None:
            ln_g1.set_data(tau2 * 1e6, g1_2)
            ax_g1.set_xlim(tau2[0] * 1e6, tau2[-1] * 1e6)
            _set_y_limits(ax_g1, g1_2, (0.0, 1.0))
        else:
            ln_g1.set_data([], [])
            _set_y_limits(ax_g1, None, (0.0, 1.0))
        fig.canvas.draw_idle()
        fig_g1.canvas.draw_idle()

    for s in (s_BG, s_tmax, s_dense, s_ddrv, s_drep, s_dcav, s_gcav, s_ampdrv, s_amprep, s_taumax, s_gamma_sp, s_gamma_dp, s_gamma_sd, s_kappa):
        s.on_changed(recompute)
    rb_sys.on_clicked(recompute)
    plt.show()
