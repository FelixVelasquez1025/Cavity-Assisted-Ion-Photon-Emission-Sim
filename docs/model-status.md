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

## Implemented behavior

- A 16-state `(level, F, mF)` basis: four S states, four P states, eight D states.
- Polarization-resolved E1 coefficients constructed with Wigner 3j/6j factors.
- Separate Gaussian drive and repump envelopes with amplitude, center, and width.
- Two truncated cavity modes, atomic decay, cavity leakage, and manifold dephasing.
- Population and cavity photon observables; optional fluorescence correlations
  in the interactive application.

UI frequency inputs are labeled MHz and time inputs are microseconds. The
solver uses seconds and angular frequencies. The existing `rabi_frequencies`
helper halves the input amplitude, and the transition builders apply another
one-half factor and a reduced matrix element. Treat the amplitude sliders as
model inputs under the existing convention, and document the relationship to
the separately verified transition-specific Rabi frequencies.

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
