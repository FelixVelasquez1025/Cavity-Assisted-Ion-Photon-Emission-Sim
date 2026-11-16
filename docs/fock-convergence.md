# Demo Fock truncation check

Run `uv run python scripts/check_fock_truncation.py` to compare dimensions
2, 3 and 4 per mode with identical demo pulses and solver settings.
The Phase 1c decision rule is that every population and photon-number
maximum absolute difference between dimensions 3 and 4 must be below 1e-3.
**The condition fails, so the default remains 2.** All figures use this
same default. This is an illustrative model, not a converged emission study.

Measured at the start of Phase 1c, before the wavelength refinements, with
QuTiP 5.2.0 and 201 samples from 0 to 2 microseconds:

| Observable | max abs(2 - 3) | max abs(3 - 4) |
| --- | --- | --- |
| P_S | 0.00236851929328 | 0.000225446410837 |
| P_P | 0.00734178682857 | 0.00214173258909 |
| P_D | 0.00863983443916 | 0.00223070870496 |
| N_left | 0.00875758113188 | 0.00147267381391 |
| N_right | 0 | 0 |

Source: executed [comparison script](../scripts/check_fock_truncation.py),
[simulate_demo](../src/caipe_sim/demo.py), and `solver_options()`.
Solver: BDF, atol=1e-8, rtol=1e-6, internal step allowance 200000.
The right mode stays vacuum because only the left branch couples.
Larger truncations and tighter integration checks remain necessary for
quantitative studies. Repeated drive/reset excitation can populate photon
states excluded by the small default truncation.
