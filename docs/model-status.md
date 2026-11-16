# Model status

The portfolio application uses `src/caipe_sim/yb171_simulator.py`, preserved
from the latest pulse-enabled Yb-171 example. Phase 1b corrected atomic inputs and the
signed hyperfine phase; its [results report](phase1b-physics-review.md)
records the numerical effects. Amplitude and decay conventions were preserved. This is an exploratory model;
the current tests establish numerical invariants and an analytic pulse limit,
not a completed photon-generation performance study. Atomic inputs were corrected in Phase 1b and are now individually sourced;
the source-precision and metadata qualifications are explicit in the reference guide.
See [the plan review](plan-review.md) for the comparison with the founding thesis.
Foot's *Atomic Physics* (2005) is the primary physics reference; Thomas Walker provides
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
solver uses seconds and angular frequencies. Parameter choices in the UI
and demo are illustrative, not a fit to experimental data. The active tests
check constructed-model invariants and limiting cases, not a completed
calibration or photon-generation performance study.

## Known limitations

- **No diagonal hyperfine splittings.** The F,mF states and angular weights
  are present, but `HFS_A_GHZ` and metadata `Ahfs` do not enter the active
  diagonal energies. The stored D value also needs an A-versus-splitting,
  sign, and source-precision decision before it can be used there.
- **No P1/2 Zeeman Hamiltonian.** Only S and D receive linear `g_F*mF*B`
  terms. The P g_F formula is now tested, but its energy shifts are not
  applied. No field-induced hyperfine mixing or quadratic Zeeman term is
  implemented. See the proposed work below; no P term was added in Phase 1b.
- **One cavity polarization couples per run.** The left/right selection adds
  only its corresponding interaction; pi adds none. Both Fock spaces share
  one detuning and coupling input, with no polarization splitting.
- **The cavity line is D3/2–P1/2 at 2.438 µm, not S–P at 369 nm.** The
  simplified reset coupling acts on this same DP transition. It does not
  reproduce the dynamics or branching of the physical 935 nm pathway.
- **Decay-rate inputs are not normalized physical rates.** The collapse
  amplitudes contain full E1 matrix-element weights, including reduced
  dipoles. The supplied `gamma_SP`/`gamma_DP` are independent illustrative
  scales, not measured total branch rates. Metadata lifetimes and branching
  fractions do not automatically normalize the channel sums. `gamma_SD`
  controls S/D projector dephasing, not a radiative D→S decay channel.
- **Truncation sensitivity is measurable.** The default demo has two Fock
  states per mode. Increasing this to three changes a manifold population
  by as much as 0.00863983443916 for the current pulse sequence; see the
  [executed comparison](fock-convergence.md). This is not a convergence proof.
- **Current correlations are fluorescence diagnostics.** They use an SP
  atomic transition operator and the final state with a static Hamiltonian,
  not pulse-resolved cavity-output correlations. Intracavity photon number
  is not itself an output probability. No polarization-entanglement claim
  is established by the present calculations.
- **UI configuration discrepancy.** `params['system']` starts as pi, but the
  radio selector starts at left and determines the actual initial solve.
  Zero default cavity coupling still disables cavity exchange. This existing
  discrepancy is reported, not changed in Phase 1b.

## What adding P1/2 Zeeman shifts would involve (not implemented)

Add a P-manifold diagonal builder using the existing `lande_g_factors` and
`MU_B=mu_B/hbar`, with terms `g_F(P)*MU_B*B_T*mF` for each P basis state.
F=0 has zero first-order shift. Include that operator in the tensor-lifted
static Hamiltonian alongside S and D in **both** the demo and UI; currently
`add_bare_terms_to_systems` accepts only the S and D operators. Define the
common rotating-frame reference and hyperfine detunings first: a common
energy offset is removable, but distinct P sublevel shifts are not.

Verify the weak-field slope, opposite mF symmetry, F=0 and zero-field
limits, Hermiticity, and detuned Rabi transition frequencies. The existing
`test_zeeman_matches_reference` checks the P g_F calculation only, not an
implemented P term. The expected coefficient and its source are recorded
in the [Phase 1b report](phase1b-physics-review.md). This change would alter
finite-field dynamics and requires a separate reviewed physics commit.

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

## Next steps (outside Phase 1b)

1. Map the rotating-frame detunings to specific Yb hyperfine transitions;
   resolve the stored D metadata before adding splittings or P Zeeman terms.
2. Decide and implement simultaneous two-polarization coupling if required
   by the protocol. Walker adds `H_a + H_b` in Eqs. 2.4.14–2.4.15.
3. Separate useful output transmission from other losses; evaluate
   `2*kappa_out*<n(t)>` and its integral. Distinguish mean emitted count
   from exactly-one-photon probability. Walker: Eqs. 2.6.3–2.6.5.
4. Implement pulse-resolved cavity-field correlations before interpreting
   single-photon purity or indistinguishability.
5. Check larger Fock truncations and tighter integration settings over the
   intended parameter range, then study pulse speed, emission, and coherence.
6. Establish normalized physical decay rates and an explicit experimental
   reset pathway if that is needed for calibration.

The `sqrt(2*gamma)` and `sqrt(2*kappa)` factors follow Walker's collapse
convention (Eqs. 2.2.6–2.2.7 and 2.4.16–2.4.19); they have not been removed.
Full source definitions are in [physics-references.md](physics-references.md).

## Software boundaries

`hamiltonian_utils.py`, `interactive_ui.py`, and the batch modules are older
parallel implementations, not the active CLI backend. Their duplicate
wavelength, lifetime, branching, and phase entries were corrected where
requested, but this does not make their Hamiltonian or amplitude conventions
equivalent. Batch sweeps still need to move to the pulse-enabled backend.
JAX is optional for historical experiments; the active application uses CPU
QuTiP. Default test collection is `tests/portfolio/`; historical tests under
`tests/tests/` and `caipe_sim/` are excluded. Archival and the README landing
page rewrite remain Steps 2–7, pending the user's go-ahead.
