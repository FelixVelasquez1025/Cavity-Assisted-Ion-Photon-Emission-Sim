"""
Headless batch runner for the Yb171+ model.

This reuses the physics helpers from interactive_ui/hamiltonian_utils to run a
grid of parameter combinations without launching the UI. Results are appended
to a CSV for offline analysis.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from qutip import basis, destroy, qeye, tensor, mesolve

# Force CPU for batch runs to avoid GPU/JAX OOM issues.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Allow running as a script (python src/caipe_sim/batch_runner.py) or as a module
# (python -m caipe_sim.batch_runner) by ensuring the repo root is on sys.path
if __package__ in (None, ""):
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import caipe_sim.interactive_ui as interactive_ui
interactive_ui.USE_JAX_BACKEND = False  # hard-disable JAX for batch runs

from caipe_sim.interactive_ui import (
    build_collapse_operators,
    enable_jax_backend_if_available,
    hf_state_index,
    projector_for_indices,
    projector_for_level_hf,
    solver_options,
)
from caipe_sim.hamiltonian_utils import (
    HF_BASIS_DIM,
    E1_DATA_HF,
    add_bare_terms_to_systems,
    eq_2_4_5_H_S,
    eq_2_4_5_H_S_hf,
    eq_2_4_6_H_D,
    eq_2_4_6_H_D_hf,
    g_d,
    g_s,
    precompute_e1_coeffs,
    precompute_lowering_tensors,
    precompute_lowering_tensors_hf,
    transition_sigma_components,
    transition_sigma_components_hf,
    H_interaction_with_cavity,
    H_interaction_with_cavity_hf,
    H_laser_systems_from_E0,
    H_laser_systems_from_E0_hf,
    state_index,
)


def _simulate_once(params: Dict[str, float], use_bosonic: bool) -> Dict[str, object]:
    print(f"[simulate_once] Starting run with params: {params}")
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

    # Cavity truncation
    N_a = N_b = 3
    Ia = qeye(N_a)
    Ib = qeye(N_b)
    projector_tensors = {
        "S": tensor(Ia, Ib, P_S),
        "P": tensor(Ia, Ib, P_P),
        "D": tensor(Ia, Ib, P_D),
    }
    cavity_loss_ops = [
        tensor(destroy(N_a), Ib, qeye(dim_atom)),
        tensor(Ia, destroy(N_b), qeye(dim_atom)),
    ]
    lowering_tensors = precompute_lowering_tensors(N_a, N_b, dim_atom) if use_bosonic else precompute_lowering_tensors_hf(N_a, N_b, dim_atom)

    # Use the same E1 coefficients source as the sandbox:
    e1_coeffs = precompute_e1_coeffs(True) if use_bosonic else E1_DATA_HF
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

    steps = max(100, int(round(params["tmax_us"] * params["density"])))
    tlist = np.linspace(0, params["tmax_us"] * 1e-6, steps)

    g = 2 * np.pi * params.get("g_cav_MHz", 0.0) * 1e6
    dc = 2 * np.pi * params["delta_c_MHz"] * 1e6
    if params["system"] in ("left", "right"):
        H_lift = H_interaction_with_cavity(H_spdp, g=g, N_a=N_a, N_b=N_b) if use_bosonic else H_interaction_with_cavity_hf(H_spdp, g=g, N_a=N_a, N_b=N_b)
    else:
        H_lift = {name: tensor(Ia, Ib, H_atom) for name, H_atom in H_spdp.items()}
    H_full = add_bare_terms_to_systems(H_lift, H_S, H_D, dc, drep, N_a, N_b)
    pol = params["system"]
    H = H_full[pol]
    psi0 = tensor(basis(N_a, 0), basis(N_b, 0), psi0_atom)
    c_ops = build_collapse_operators(params, lowering_tensors, projector_tensors, cavity_loss_ops)
    mesolve_opts = solver_options(store_states=False)
    # Give the integrator more headroom as in the sandbox.
    if "nsteps" not in mesolve_opts:
        mesolve_opts["nsteps"] = 200000
    res = mesolve(
        H,
        psi0,
        tlist,
        c_ops=c_ops,
        e_ops=[projector_tensors["S"], projector_tensors["P"], projector_tensors["D"]],
        options=mesolve_opts,
    )

    P_S_arr = np.array(res.expect[0], dtype=float)
    P_P_arr = np.array(res.expect[1], dtype=float)
    P_D_arr = np.array(res.expect[2], dtype=float)

    final_P_S = float(P_S_arr[-1])
    final_P_P = float(P_P_arr[-1])
    final_P_D = float(P_D_arr[-1])
    out: Dict[str, object] = params.copy()
    out.update(
        {
            "tlist": np.array(tlist, dtype=float),
            "P_S": P_S_arr,
            "P_P": P_P_arr,
            "P_D": P_D_arr,
            "final_P_S": final_P_S,
            "final_P_P": final_P_P,
            "final_P_D": final_P_D,
        }
    )
    print(f"[simulate_once] Finished run: P_S={final_P_S}, P_P={final_P_P}, P_D={final_P_D}")
    return out


def run_batch(
    grid: Dict[str, Iterable[float]],
    base_params: Dict[str, float] | None = None,
    output: Path | str | None = None,
    use_bosonic: bool = False,
    save_timeseries: bool = True,
    timeseries_dir: str | Path | None = None,
) -> Path:
    """Run a batch over a parameter grid and append results to CSV."""
    enable_jax_backend_if_available()
    if base_params is None:
        base_params = dict(
            B_G=0.0,
            delta_drive_MHz=0.0,
            delta_rep_MHz=0.0,
            delta_c_MHz=0.0,
            g_cav_MHz=0.0,
            amp_driv_MHz=0.0,
            amp_rep_MHz=0.0,
            # Match the interactive defaults for clearer dynamics
            tmax_us=50.0,
            density=8.0,
            gamma_SP=0.03,
            gamma_DP=0.034,
            gamma_SD=0.000136,
            kappa_MHz=0.5,
            system="right",
            n_tau=5,
            tau_max_us=0.2,
        )
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    out_path = Path(output) if output else Path(__file__).resolve().parent / "batch_results.csv"
    ts_dir = Path(timeseries_dir) if timeseries_dir else Path(__file__).resolve().parent / "batch_timeseries"
    if save_timeseries:
        ts_dir.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
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

    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        total = 1
        for vals in values:
            total *= len(vals)
        print(f"[run_batch] Starting batch: {total} combos, bosonic={use_bosonic}")

        for idx, combo in enumerate(itertools.product(*values)):
            print(f"[run_batch] Combo {idx+1}/{total}: {dict(zip(keys, combo))}")
            p = base_params.copy()
            for k, v in zip(keys, combo):
                p[k] = v
            row = _simulate_once(p, use_bosonic=use_bosonic)
            if save_timeseries:
                np.savez(
                    ts_dir / f"run_{idx:05d}.npz",
                    params=json.dumps({k: row.get(k) for k in keys}),
                    t=row["tlist"],
                    P_S=row["P_S"],
                    P_P=row["P_P"],
                    P_D=row["P_D"],
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    use_bosonic,
                    row.get("system", ""),
                    row.get("B_G", ""),
                    row.get("delta_drive_MHz", ""),
                    row.get("delta_rep_MHz", ""),
                    row.get("delta_c_MHz", ""),
                    row.get("g_cav_MHz", ""),
                    row.get("amp_driv_MHz", ""),
                    row.get("amp_rep_MHz", ""),
                    row.get("tmax_us", ""),
                    row.get("density", ""),
                    row.get("gamma_SP", ""),
                    row.get("gamma_DP", ""),
                    row.get("gamma_SD", ""),
                    row.get("kappa_MHz", ""),
                    row.get("final_P_S", ""),
                    row.get("final_P_P", ""),
                    row.get("final_P_D", ""),
                ]
            )
            print(f"[run_batch] Finished combo {idx+1}/{total}: P_S={row.get('final_P_S')}, P_P={row.get('final_P_P')}, P_D={row.get('final_P_D')}")
    return out_path


if __name__ == "__main__":
    # Small test grid: vary drive amplitude only, keep other params standard.
    grid = {
        "B_G": [5.0],
        "delta_drive_MHz": [0.0],
        "delta_rep_MHz": [0.0],
        "amp_driv_MHz": [5.0, 10.0],
        "amp_rep_MHz": [5.0],
        "system": ["left"],
        "gamma_SP": [0.03],
        "gamma_DP": [0.034],
        "gamma_SD": [0.000136],
    }
    out = run_batch(grid, use_bosonic=False)
    print(f"Wrote batch results to {out}")
