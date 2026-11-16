# Demo Fock truncation check

This compares the corrected Phase 1b demo with Fock dimensions 2 and 3 per
mode (respectively n=0,1 and n=0,1,2). No other physical or solver inputs
are changed. The default remains 2. Source: the executed command below,
[simulate_demo](../src/caipe_sim/demo.py), and
[comparison script](../scripts/check_fock_truncation.py).

```bash
uv run python scripts/check_fock_truncation.py
```

Maximum absolute differences over the demo's 201 sample times from 0 to
2 microseconds, with QuTiP 5.2.0:

| Observable | max abs(dimension 3 - dimension 2) |
| --- | --- |
| S-manifold population | 0.00236851929328 |
| P-manifold population | 0.00734178682857 |
| D-manifold population | 0.00863983443916 |
| Left-mode mean photon number | 0.00875758113188 |
| Right-mode mean photon number | 0 |

The largest population discrepancy is 0.00863983443916 (about 0.864 percentage
points), in D. This is a measured truncation sensitivity, not proof that
dimension 3 is converged or that the default supports a precision study.
The other mode remains vacuum because the demo couples only the left branch.
Solver settings: BDF, absolute tolerance 1e-8, relative tolerance 1e-6,
internal step allowance 200000, from `solver_options()`.

The repeated excitation/reset process can populate higher photon states;
the default truncation excludes these. Larger truncations and tighter
integration checks remain necessary for quantitative emission studies.
