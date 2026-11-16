# Demo Fock truncation check

Run `uv run python scripts/check_fock_truncation.py` to compare dimensions
2, 3 and 4 per mode with identical demo pulses and solver settings.
The Phase 1c decision rule is that every population and photon-number
maximum absolute difference between dimensions 3 and 4 must be below 1e-3.
**The condition fails, so the default remains 2.** All figures use this
same default. This is an illustrative model, not a converged emission study.

Measured after the Phase 1c NIST and 369 nm wavelength refinements, with
QuTiP 5.2.0 and 201 samples from 0 to 2 microseconds:

| Observable | max abs(2 - 3) | max abs(3 - 4) |
| --- | --- | --- |
| P_S | 0.0023708679657 | 0.000225433529746 |
| P_P | 0.0073434450683 | 0.00214468824286 |
| P_D | 0.00864233158656 | 0.00223388488984 |
| N_left | 0.00876353718433 | 0.00147452473536 |
| N_right | 0 | 0 |

Source: executed [comparison script](../scripts/check_fock_truncation.py),
[simulate_demo](../src/caipe_sim/demo.py), and `solver_options()`.
Solver: BDF, atol=1e-8, rtol=1e-6, internal step allowance 200000.
The right mode stays vacuum because only the left branch couples.
Larger truncations and tighter integration checks remain necessary for
quantitative studies. Repeated drive/reset excitation can populate photon
states excluded by the small default truncation.
