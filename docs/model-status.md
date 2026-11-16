# Model status

The portfolio application uses `src/caipe_sim/yb171_simulator.py`, preserved
from the latest pulse-enabled Yb-171 example. Packaging changes do not retune
the existing Hamiltonian or decay conventions. This is an exploratory model;
the current tests establish numerical invariants and an analytic pulse limit,
not a completed photon-generation performance study. The Yb numerical inputs
were computed separately and are treated as verified, as confirmed by the user.
See [the plan review](plan-review.md) for the comparison with the founding thesis.
Foot's *Atomic Physics* (2005) is the primary physics reference; Ward provides
the specialized cavity model. The [reference guide](physics-references.md)
records the relevant sections and translates their conventions.

The DP pulse is the **D3/2→P1/2 reset coupling (simplified stand-in for the
935 nm repump via 3D[3/2]1/2)**. It directly couples the existing D and P
manifolds on the 2.438 µm line; the intermediate state is metadata only.
`omega_reset` accepts `reset_amp`, `reset_t0`, and `reset_sigma`.
`omega_rep` and `rep_*` pulse arguments remain compatibility aliases;
legacy UI/cache parameter keys (`amp_rep_MHz`, `delta_rep_MHz`,
`rep_t0_us`, `rep_sigma_us`) retain their serialized names. The demo exports
`reset_MHz`, with `repump_MHz` as a compatibility alias for older NPZ readers.
Neither the aliases nor the name “reset” imply a physical 935 nm simulation.

## Implemented behavior

- A 16-state `(level, F, mF)` basis: four S states, four P states, eight D states.
- Polarization-resolved E1 coefficients constructed with Wigner 3j/6j factors.
- Separate Gaussian drive and simplified DP reset envelopes with amplitude, center, and width.
- Two truncated cavity modes, atomic decay, cavity leakage, and manifold dephasing.
- Population and cavity photon observables; optional fluorescence correlations
  in the interactive application.

UI frequency inputs are labeled MHz and time inputs are microseconds. The
solver uses seconds and angular frequencies. The existing `rabi_frequencies`
helper halves the input amplitude, and the transition builders apply another
one-half factor and a reduced matrix element. Treat the amplitude sliders as
model inputs under the existing convention, and document the relationship to
the separately verified transition-specific Rabi frequencies.

## Exact amplitude convention (behavior preserved)

Let `A` be either amplitude slider input in MHz and `c_ul` the signed
SP or DP entry in `E1_DATA_HF`, including its reduced dipole element in
atomic units. The code computes `E0 = 2*pi*1e6*A`, then
`rabi_frequencies(E0) = E0/2`. The unit-amplitude laser builder contributes
`c_ul/2`, so, with `f(t) = exp(-(t-t0)^2/(2*sigma^2))`,

```text
(H/hbar)_ul(t) = (2*pi*1e6*A) * c_ul * f(t) / 4   [rad/s]
Omega_ul(t)    = (2*pi*1e6*A) * c_ul * f(t) / 2   [rad/s]
```

The latter is the signed transition Rabi frequency in the convention
`H_ul/hbar = Omega_ul/2`; its magnitude enters the isolated resonant
two-level population formula in Foot, Eq. 7.29. The slider is not a
calibrated electric field or measured transition Rabi frequency. The demo's
envelope export is the intermediate `E0*f(t)/2`, divided by `2*pi*1e6`,
not `Omega_ul/(2*pi*1e6)`. The analytic test supplies that intermediate
envelope directly and includes the actual matrix element in its pulse area.

Sources: `rabi_frequencies`, `_build_systems_hf`, `build_and_solve` in
[yb171_simulator.py](../src/caipe_sim/yb171_simulator.py),
[demo.py](../src/caipe_sim/demo.py), and
[test_gaussian_pi_pulse_matches_analytic_rabi_solution](../tests/portfolio/test_yb171.py).
Only the contradictory docstrings have changed; no amplitude factors were
removed. The older `hamiltonian_utils.py` uses a different convention and
must not be treated as the same solver.

## Remaining model and analysis work

1. **Rotating-frame mapping.** Ward's thesis references the excited P state and
   expresses the Hamiltonian in detunings, so absolute optical energies need
   not appear. Document which Yb hyperfine transitions define those detunings
   and how relative offsets are handled. The active code builds S and D Zeeman
   terms; the thesis also includes a P term (Eq. 2.18). This concerns using the
   verified constants in the Hamiltonian, not recalculating the constants.
2. **Both cavity polarizations.** The thesis uses `H_a + H_b` (Eq. 2.26). The
   current selected branch includes one cavity coupling; pi includes none.
   Establish whether the intended protocol deliberately restricts the modes
   or should evolve both simultaneously.
3. **Output efficiency.** In the thesis convention, useful output flux is
   `2 * kappa_out * <n(t)>`. Separate useful transmission from other cavity
   losses, integrate the flux, and distinguish mean emitted photon count from
   the probability of exactly one photon. The older steady-state efficiency
   proxy is not this measurement.
4. **Pulse-resolved correlations.** The UI evaluates correlations using the
   atomic SP fluorescence operator, a static Hamiltonian, and a final state.
   Implement the intended time-dependent cavity-output correlations before
   interpreting these plots as single-photon performance.
5. **Cavity convergence and studies.** Compare larger photon truncations and
   integration settings, then quantify pulse speed, emission, and coherence
   over a parameter grid. The small demo is a numerical illustration.

The `sqrt(2 * gamma)` and `sqrt(2 * kappa)` collapse amplitudes match Ward's
thesis, Eqs. 2.30–2.35. That factor of two is intentional. Record the rate and
angular-weight conventions alongside the verified Yb inputs so that older
and newer model results can be compared consistently.

## Software boundaries

`hamiltonian_utils.py`, `interactive_ui.py`, and `batch_runner.py` contain an
older parallel implementation. They remain for comparison and are not the
portfolio CLI's physics backend. Bring batch sweeps onto the pulse-enabled
backend before treating them as equivalent results. JAX is optional for those
older experiments; the portfolio application and demo currently use CPU QuTiP.

The active validation suite is `tests/portfolio/`. Historical tests under
`tests/tests/` and `caipe_sim/` target obsolete APIs and are retained as project
history; they are explicitly outside the default pytest collection. Reconcile
or archive them before a final public release.
