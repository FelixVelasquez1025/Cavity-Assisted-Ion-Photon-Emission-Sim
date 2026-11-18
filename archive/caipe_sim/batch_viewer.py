"""
Batch results viewer.

Loads precomputed time series produced by batch_runner (NPZ files in
src/caipe_sim/batch_timeseries) and lets you browse them with sliders and
radio buttons. No recomputation is performed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

# Keys we match numerically
NUM_KEYS = [
    "B_G",
    "delta_drive_MHz",
    "delta_rep_MHz",
    "delta_c_MHz",
    "g_cav_MHz",
    "amp_driv_MHz",
    "amp_rep_MHz",
    "gamma_SP",
    "gamma_DP",
    "gamma_SD",
]
SYS_KEY = "system"


def _load_runs(ts_dir: Path) -> List[Dict]:
    runs: List[Dict] = []
    for npz_path in sorted(ts_dir.glob("run_*.npz")):
        with np.load(npz_path) as data:
            params = json.loads(data["params"].item())
            runs.append(
                {
                    "path": npz_path,
                    "params": params,
                    "t": np.array(data["t"], dtype=float),
                    "P_S": np.array(data["P_S"], dtype=float),
                    "P_P": np.array(data["P_P"], dtype=float),
                    "P_D": np.array(data["P_D"], dtype=float),
                }
            )
    return runs


def _nearest_run(runs: List[Dict], target: Dict, unique_vals: Dict[str, List[float]]) -> Dict | None:
    sys_sel = target[SYS_KEY]
    cands = [r for r in runs if str(r["params"].get(SYS_KEY, "")) == sys_sel]
    if not cands:
        return None

    def dist(r):
        p = r["params"]
        return sum((float(p.get(k, 0.0)) - float(target[k])) ** 2 for k in NUM_KEYS)

    return min(cands, key=dist)


def main() -> None:
    # batch_timeseries lives alongside this module (src/caipe_sim/batch_timeseries)
    ts_dir = Path(__file__).resolve().parent / "batch_timeseries"
    runs = _load_runs(ts_dir)
    if not runs:
        raise SystemExit(f"No runs found in {ts_dir}")

    mins = {k: float("inf") for k in NUM_KEYS}
    maxs = {k: float("-inf") for k in NUM_KEYS}
    unique_vals: Dict[str, List[float]] = {k: [] for k in NUM_KEYS}
    systems = set()
    for r in runs:
        p = r["params"]
        systems.add(str(p.get(SYS_KEY, "left")))
        for k in NUM_KEYS:
            v = float(p.get(k, 0.0))
            mins[k] = min(mins[k], v)
            maxs[k] = max(maxs[k], v)
            unique_vals[k].append(v)
    for k in NUM_KEYS:
        unique_vals[k] = sorted(set(unique_vals[k]))

    target = {k: float(runs[0]["params"].get(k, 0.0)) for k in NUM_KEYS}
    target[SYS_KEY] = str(runs[0]["params"].get(SYS_KEY, "left"))

    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.55)
    line_s, = ax.plot([], [], label="P_S")
    line_p, = ax.plot([], [], label="P_P")
    line_d, = ax.plot([], [], label="P_D")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Population")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    axcolor = "lightgoldenrodyellow"
    row_h, gap = 0.03, 0.005
    y0 = 0.50
    xL, wC = 0.10, 0.35
    xR = 0.55

    def row_y(i: int) -> float:
        return y0 - i * (row_h + gap)

    sliders: Dict[str, Slider] = {}
    axes_info = [
        ("B_G", xL, row_y(0), wC, row_h, False),
        ("delta_drive_MHz", xL, row_y(1), wC, row_h, True),
        ("delta_rep_MHz", xR, row_y(1), wC, row_h, True),
        ("delta_c_MHz", xL, row_y(2), wC, row_h, True),
        ("g_cav_MHz", xR, row_y(2), wC, row_h, False),
        ("amp_driv_MHz", xL, row_y(3), wC, row_h, False),
        ("amp_rep_MHz", xR, row_y(3), wC, row_h, False),
        ("gamma_SP", xL, row_y(4), wC, row_h, False),
        ("gamma_DP", xR, row_y(4), wC, row_h, False),
        ("gamma_SD", xL, row_y(5), wC, row_h, False),
    ]

    for name, x, y, w, h, allow_neg in axes_info:
        ax_s = plt.axes([x, y, w, h], facecolor=axcolor)
        vmin, vmax = mins[name], maxs[name]
        if not allow_neg and vmin < 0:
            vmin = 0.0
        sliders[name] = Slider(ax_s, name, vmin, vmax, valinit=target[name])

    ax_sys = plt.axes([0.82, 0.08, 0.15, 0.12], facecolor=axcolor)
    systems_sorted = tuple(sorted(systems))
    rb_sys = RadioButtons(ax_sys, systems_sorted, active=systems_sorted.index(target[SYS_KEY]) if target[SYS_KEY] in systems_sorted else 0)

    status_text = plt.figtext(0.5, 0.02, "", ha="center", va="center")

    def update_plot(*_):
        # Snap each slider to the nearest precomputed value to avoid picking
        # parameters that were never simulated.
        for k, s in sliders.items():
            val = float(s.val)
            vals = unique_vals[k]
            if vals:
                snap = min(vals, key=lambda x: abs(x - val))
            else:
                snap = val
            target[k] = snap
        target[SYS_KEY] = rb_sys.value_selected
        run = _nearest_run(runs, target, unique_vals)
        if run is None:
            status_text.set_text("No matching run found")
            fig.canvas.draw_idle()
            return
        t_us = run["t"] * 1e6
        line_s.set_data(t_us, run["P_S"])
        line_p.set_data(t_us, run["P_P"])
        line_d.set_data(t_us, run["P_D"])
        ax.set_xlim(t_us[0], t_us[-1])
        ax.set_ylim(0, 1.05)
        # Show which run was loaded and its params for clarity
        p = run["params"]
        short_params = {k: p.get(k) for k in [SYS_KEY, "B_G", "delta_drive_MHz", "delta_rep_MHz", "amp_driv_MHz", "amp_rep_MHz"]}
        status_text.set_text(f"Loaded {run['path'].name} | {short_params}")
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update_plot)
    rb_sys.on_clicked(update_plot)

    update_plot()
    plt.show()


if __name__ == "__main__":
    main()
