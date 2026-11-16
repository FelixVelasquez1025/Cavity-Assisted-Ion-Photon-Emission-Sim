"""
Tiny smoke test for the JAX/QuTiP-JAX backend.

Run with:
    python examples/jax_gpu_smoketest.py

It tries to enable the JAX backend, then evolves a 2-level system with a
very small time grid. If JAX/qutip_jax is unavailable, it falls back to CPU.
"""

from __future__ import annotations

import os
import numpy as np
import qutip


# Avoid large upfront GPU preallocation if JAX is present.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

backend = "CPU fallback"
use_jax = False

try:
    import qutip_jax  # noqa: F401

    qutip_jax.set_as_default()
    backend = "JAX"
    use_jax = True
except Exception as exc:  # pragma: no cover - best-effort optional import
    backend = f"CPU fallback ({exc})"
    use_jax = False


def main() -> None:
    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    H = 5.0 * sx  # faster Rabi rate for visible evolution
    psi0 = qutip.basis(2, 0)
    tlist = np.linspace(0, 1e-3, 50)  # longer evolution for clearer change

    opts = qutip.solver.Options(
        method="diffrax" if use_jax else "bdf",
        normalize_output=False if use_jax else True,
        progress_bar=None,
        store_states=True,
    )

    res = qutip.mesolve(H, psi0, tlist, c_ops=[], e_ops=[sz], options=opts)
    z_vals = np.real(res.expect[0])

    print(f"Backend: {backend}")
    print("First five <sigma_z>(t):", np.round(z_vals[:5], 6))
    if res.states:
        print("Final state norm:", float(res.states[-1].norm()))
    else:
        print("States not stored (unexpected for this test).")


if __name__ == "__main__":
    main()
