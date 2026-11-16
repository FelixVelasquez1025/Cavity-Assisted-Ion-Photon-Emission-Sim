"""Generate the README figures from the active solver and analytic test setup.

Run: uv run python scripts/make_readme_figures.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from caipe_sim.config import DEFAULT_FOCK_DIMENSION
from caipe_sim.demo import simulate_demo
from caipe_sim.validation import run_rabi_validation


def make_figures(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 2}):
        data = run_rabi_validation()
        np.testing.assert_allclose(data["simulated"], data["analytic"], atol=2e-6, rtol=0)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
        axes[0].plot(data["time_us"], data["analytic"], color="#173f5f", label="Analytic pulse-area solution")
        axes[0].plot(data["time_us"][::8], data["simulated"][::8], "o", ms=4.5,
                     mfc="white", color="#d46b20", label="QuTiP simulation")
        axes[0].set(ylabel="Excited-state population", ylim=(-.02, 1.05),
                    title="Resonant Gaussian Rabi pulse\nS(F=0, mF=0) → P(F=1, mF=0)")
        axes[0].legend(loc="upper left", frameon=False, fontsize=10)
        axes[1].plot(data["time_us"], data["residual"], color="#d46b20")
        axes[1].axhline(0, color="0.7", lw=.7)
        axes[1].set(xlabel="Time (µs)", ylabel="Residual")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.suptitle(f"Analytic validation · {DEFAULT_FOCK_DIMENSION} Fock states per mode; cavity, decay and reset off", fontsize=10)
        fig.savefig(output/"rabi_validation.png", dpi=180)
        plt.close(fig)

        demo = simulate_demo()
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
        for key, label, color in [("P_S", "S", "#173f5f"), ("P_P", "P", "#d46b20"), ("P_D", "D", "#318274")]:
            axes[0].plot(demo["time_us"], demo[key], label=f"{label} manifold", color=color)
        axes[0].set(ylabel="Population", ylim=(-.02, 1.05),
                    title=f"Illustrative Yb-171 dynamics · {DEFAULT_FOCK_DIMENSION} Fock states per mode")
        axes[1].plot(demo["time_us"], demo["drive_MHz"], label="S–P drive", color="#173f5f")
        axes[1].plot(demo["time_us"], demo["reset_MHz"], label="D–P simplified reset", color="#d46b20")
        axes[1].set_ylabel("Envelope / 2π (MHz)")
        axes[2].plot(demo["time_us"], demo["N_left"], label="Left cavity mode", color="#318274")
        axes[2].plot(demo["time_us"], demo["N_right"], "--", label="Right mode (uncoupled)", color="0.5", lw=1.5)
        axes[2].set(xlabel="Time (µs)", ylabel="Mean photon number")
        for ax in axes:
            ax.legend(loc="upper right", frameon=False, fontsize=10)
        fig.savefig(output/"demo_dynamics.png", dpi=180)
        plt.close(fig)
    print(f"Saved figures to {output}; max Rabi residual = {np.max(np.abs(data['residual'])):.9g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    make_figures(parser.parse_args().output_dir)
