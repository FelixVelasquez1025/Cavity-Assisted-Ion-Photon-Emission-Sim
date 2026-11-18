# Demo Fock truncation check

Run `uv run python scripts/check_fock_truncation.py` to compare dimensions
2, 3, 4 and 5 per mode with identical demo pulses and solver settings.
The pre-merge decision rule is that every population and photon-number
maximum absolute difference between dimensions 4 and 5 must be below 1e-3.
**The condition passes, so the demo/UI default is now 4.** Both figures and
the analytic test use this same default. The earlier 3−4 check failed. This is an illustrative model, not a converged emission study.

Measured after the Phase 1c NIST and 369 nm wavelength refinements, with
QuTiP 5.2.0 and 201 samples from 0 to 2 microseconds:

| Observable | max abs(2 - 3) | max abs(3 - 4) | max abs(4 - 5) |
| --- | --- | --- | --- |
| P_S | 0.0023708679657 | 0.000225433529746 | 0.000223598631773 |
| P_P | 0.0073434450683 | 0.00214468824286 | 0.000241316721655 |
| P_D | 0.00864233158656 | 0.00223388488984 | 0.000105121008392 |
| N_left | 0.00876353718433 | 0.00147452473536 | 7.53242015548e-05 |
| N_right | 0 | 0 | 0 |

Source: executed [comparison script](../scripts/check_fock_truncation.py),
[simulate_demo](../src/caipe_sim/demo.py), and `solver_options()`.
Solver: BDF, atol=1e-8, rtol=1e-6, internal step allowance 200000.
The right mode stays vacuum because only the left branch couples.
Larger truncations and tighter integration checks remain necessary for
quantitative studies. Repeated drive/reset excitation can populate photon
states excluded by the small default truncation.
