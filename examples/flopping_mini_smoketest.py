"""
Reduced-size Rabi flopping smoke test.

This is meant to be a tiny version of the full flopping simulation that can
run quickly (and fit in limited VRAM if JAX is enabled).

Run with:
    python examples/flopping_mini_smoketest.py

It tries to enable the JAX backend if available (like interactive_ui does),
but will fall back to CPU otherwise.
"""

from __future__ import annotations

import numpy as np
from qutip import  qeye

import os
import numpy as np
import qutip

# Match the optional JAX toggle style used in the main codebase
try:
    import qutip_jax  # noqa: F401

    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

USE_JAX_BACKEND = True  # set False to force CPU

# Avoid large upfront GPU preallocation if JAX is present.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

if USE_JAX_BACKEND and HAVE_JAX:
    import qutip_jax  # type: ignore  # noqa: F401

    qutip_jax.set_as_default()
    backend = "JAX"
else:
    backend = "CPU"

from caipe_sim.hamiltonian_utils import (  # noqa: E402
    HF_BASIS_DIM,
    g_s,
    g_d,
    state_index,
    hf_state_index,
    precompute_e1_coeffs,
    precompute_e1_coeffs_hf,
    eq_2_4_5_H_S,
    eq_2_4_5_H_S_hf,
    eq_2_4_6_H_D,
    eq_2_4_6_H_D_hf,
    add_bare_terms_to_systems,
    H_interaction_with_cavity,
    H_interaction_with_cavity_hf,
    H_laser_systems_from_E0,
    H_laser_systems_from_E0_hf,
    precompute_lowering_tensors,
    precompute_lowering_tensors_hf,
    transition_sigma_components,
    transition_sigma_components_hf,
)
from caipe_sim.interactive_ui import build_collapse_operators, solver_options  # noqa: E402


def main() -> None:
    use_bosonic = True  # toggle to False to try the small hyperfine basis
    dim_atom = 8 if use_bosonic else HF_BASIS_DIM

    # Minimal-but-valid cavity truncation to keep the space tiny (must be >=2)
    N_a = 2
    N_b = 2

    Ia = qeye(N_a)
    Ib = qeye(N_b)

    if use_bosonic:
        psi0_atom = qutip.basis(dim_atom, state_index("S", +0.5))
        P_S = qutip.qeye(dim_atom)
        P_P = qutip.qeye(dim_atom)
        P_D = qutip.qeye(dim_atom)
        comps_sp = transition_sigma_components("SP", dim=dim_atom)
        e1_coeffs = precompute_e1_coeffs(True)
    else:
        psi0_atom = qutip.basis(dim_atom, hf_state_index("S", 0.0, 0.0))
        P_S = qutip.qeye(dim_atom)
        P_P = qutip.qeye(dim_atom)
        P_D = qutip.qeye(dim_atom)
        comps_sp = transition_sigma_components_hf("SP", dim=dim_atom)
        e1_coeffs = precompute_e1_coeffs_hf()

    projector_tensors = {
        "S": qutip.tensor(Ia, Ib, P_S),
        "P": qutip.tensor(Ia, Ib, P_P),
        "D": qutip.tensor(Ia, Ib, P_D),
    }

    field_left = qutip.tensor(qutip.destroy(N_a), Ib, qutip.qeye(dim_atom))
    field_right = qutip.tensor(Ia, qutip.destroy(N_b), qutip.qeye(dim_atom))
    sigma_left = qutip.tensor(Ia, Ib, comps_sp.get(+1, qutip.qeye(dim_atom) * 0))
    sigma_right = qutip.tensor(Ia, Ib, comps_sp.get(-1, qutip.qeye(dim_atom) * 0))

    lowering_tensors = (
        precompute_lowering_tensors(N_a, N_b, dim_atom)
        if use_bosonic
        else precompute_lowering_tensors_hf(N_a, N_b, dim_atom)
    )

    params = dict(
        B_G=5.0,
        delta_drive_MHz=0.0,
        delta_rep_MHz=0.0,
        delta_c_MHz=0.0,
        g_cav_MHz=0.5,
        amp_driv_MHz=5.0,
        amp_rep_MHz=5.0,
        tmax_us=2.0,
        density=10.0,
        gamma_SP=0.05,
        gamma_DP=0.05,
        gamma_SD=0.0,
        kappa_MHz=0.5,
        system="left",
    )

    B_T = params["B_G"] * 1e-4
    ddrv = 2 * np.pi * params["delta_drive_MHz"] * 1e6
    drep = 2 * np.pi * params["delta_rep_MHz"] * 1e6
    E0_driv = 2 * np.pi * params["amp_driv_MHz"] * 1e6
    E0_rep = 2 * np.pi * params["amp_rep_MHz"] * 1e6

    if use_bosonic:
        H_spdp = H_laser_systems_from_E0(E0_driv, E0_rep, dim=dim_atom, e1_coeffs=e1_coeffs)
        H_S = eq_2_4_5_H_S(ddrv, B_T, g_s, dim=dim_atom)
        H_D = eq_2_4_6_H_D(drep, B_T, g_d, dim=dim_atom)
    else:
        H_spdp = H_laser_systems_from_E0_hf(E0_driv, E0_rep, dim=dim_atom, e1_coeffs_hf=e1_coeffs)
        H_S = eq_2_4_5_H_S_hf(ddrv, B_T, g_s, dim=dim_atom)
        H_D = eq_2_4_6_H_D_hf(drep, B_T, g_d, dim=dim_atom)

    g = 2 * np.pi * params["g_cav_MHz"] * 1e6
    dc = 2 * np.pi * params["delta_c_MHz"] * 1e6

    if params["system"] in ("left", "right"):
        H_lift = (
            H_interaction_with_cavity(H_spdp, g=g, N_a=N_a, N_b=N_b)
            if use_bosonic
            else H_interaction_with_cavity_hf(H_spdp, g=g, N_a=N_a, N_b=N_b)
        )
    else:
        H_lift = {name: qutip.tensor(Ia, Ib, H_atom) for name, H_atom in H_spdp.items()}

    H_full = add_bare_terms_to_systems(H_lift, H_S, H_D, dc, drep, N_a, N_b)
    H = H_full[params["system"]]

    psi0 = qutip.tensor(qutip.basis(N_a, 0), qutip.basis(N_b, 0), psi0_atom)

    cavity_loss_ops = [
        qutip.tensor(qutip.destroy(N_a), Ib, qutip.qeye(dim_atom)),
        qutip.tensor(Ia, qutip.destroy(N_b), qutip.qeye(dim_atom)),
    ]
    c_ops = build_collapse_operators(params, lowering_tensors, projector_tensors, cavity_loss_ops)

    steps = max(20, int(round(params["tmax_us"] * params["density"])))
    tlist = np.linspace(0, params["tmax_us"] * 1e-6, steps)

    res = qutip.mesolve(
        H,
        psi0,
        tlist,
        c_ops=c_ops,
        e_ops=[projector_tensors["S"], projector_tensors["P"], projector_tensors["D"]],
        options=solver_options(store_states=True),
    )

    Pg, Pe, Pd = (np.real(res.expect[i]) for i in range(3))
    print(f"Backend: {backend}")
    print("P_S final:", Pg[-1])
    print("P_P final:", Pe[-1])
    print("P_D final:", Pd[-1])
    print("First 5 time points (us):", np.round(tlist[:5] * 1e6, 6))
    print("First 5 P_S:", np.round(Pg[:5], 6))


if __name__ == "__main__":
    main()
