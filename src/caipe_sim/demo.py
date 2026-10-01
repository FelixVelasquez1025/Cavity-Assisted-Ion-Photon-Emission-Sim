"""Small headless run of the full hyperfine basis with two cavity modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qutip import basis, destroy, mesolve, qeye, tensor


def simulate_demo() -> dict[str, np.ndarray]:
    # Use the same pulse-enabled physics as the interactive application.
    from . import yb171_simulator as model

    dim = model.HF_BASIS_DIM
    n = 2
    identity = qeye(n)
    projectors = {
        level: tensor(identity, identity, model.projector_for_level_hf(level))
        for level in ("S", "P", "D")
    }
    cavity_ops = [
        tensor(destroy(n), identity, qeye(dim)),
        tensor(identity, destroy(n), qeye(dim)),
    ]
    # Preserve the existing amplitude and decay conventions in this example.
    params = dict(gamma_SP=0.01, gamma_DP=0.01, gamma_SD=0.000136, kappa_MHz=0.1)
    c_ops = model.build_collapse_operators(
        params, model.precompute_lowering_tensors_hf(n, n), projectors, cavity_ops
    )
    zero = qeye(dim) * 0
    cavity = model.H_interaction_with_cavity_hf(
        {pol: zero for pol in ("left", "pi", "right")}, g=2 * np.pi * 0.5e6, N_a=n, N_b=n
    )
    h_s = model.eq_2_4_5_H_S_hf(0.0, 0.0)
    h_d = model.eq_2_4_6_H_D_hf(0.0, 0.0)
    h_static = model.add_bare_terms_to_systems(cavity, h_s, h_d, 0.0, 0.0, n, n)["left"]
    h_drive = tensor(identity, identity, model._build_systems_hf("SP", 1.0, dim)["left"])
    h_repump = tensor(identity, identity, model._build_systems_hf("DP", 1.0, dim)["left"])
    drive_amp, rep_amp = model.rabi_frequencies(2 * np.pi * 5e6, 2 * np.pi * 3e6)
    pulse_args = dict(
        drive_amp=drive_amp, drive_t0=0.6e-6, drive_sigma=0.2e-6,
        rep_amp=rep_amp, rep_t0=1.1e-6, rep_sigma=0.25e-6,
    )
    times = np.linspace(0.0, 2e-6, 201)
    psi0 = tensor(basis(n, 0), basis(n, 0), basis(dim, model.hf_state_index("S", 0.0, 0.0)))
    result = mesolve(
        [h_static, [h_drive, model.omega_drive], [h_repump, model.omega_rep]],
        psi0, times, c_ops=c_ops,
        e_ops=[*projectors.values(), *(op.dag() * op for op in cavity_ops)],
        args=pulse_args, options=model.solver_options(),
    )
    return dict(
        time_us=times * 1e6,
        P_S=np.asarray(result.expect[0]), P_P=np.asarray(result.expect[1]),
        P_D=np.asarray(result.expect[2]),
        N_left=np.asarray(result.expect[3]), N_right=np.asarray(result.expect[4]),
        drive_MHz=np.asarray([model.omega_drive(t, **pulse_args) for t in times]) / (2 * np.pi * 1e6),
        repump_MHz=np.asarray([model.omega_rep(t, **pulse_args) for t in times]) / (2 * np.pi * 1e6),
    )


def export_demo(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = simulate_demo()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "yb171_demo.npz", **data)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
        for key, label in [("P_S", "S manifold"), ("P_P", "P manifold"), ("P_D", "D manifold")]:
            axes[0].plot(data["time_us"], data[key], label=label)
        axes[0].set(ylabel="Population", ylim=(-0.02, 1.02), title="Yb-171: Gaussian pulses and cavity coupling")
        for key, label in [("drive_MHz", "Drive envelope"), ("repump_MHz", "Repump envelope")]:
            axes[1].plot(data["time_us"], data[key], label=label)
        axes[1].set_ylabel("Envelope / 2π (MHz)")
        axes[2].plot(data["time_us"], data["N_left"], label="Left cavity mode")
        axes[2].plot(data["time_us"], data["N_right"], label="Right cavity mode")
        axes[2].set(xlabel="Time (µs)", ylabel="Mean photon number")
        for ax in axes:
            ax.legend(loc="upper right")
            ax.grid(alpha=0.2)
        fig.savefig(output / "yb171_demo.png", dpi=160)
        plt.close(fig)
    print(f"Saved {output / 'yb171_demo.png'} and {output / 'yb171_demo.npz'}")
