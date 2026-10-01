# Review against the original seven-week plan

Reviewed October 1, 2026, using the local code, saved results, Git history, and
the supplied thesis: Travers Ward, *Generation of phase controlled single
photons*, PhD thesis, University of Sussex, September 2022.

The separately calculated Yb numerical values are treated as verified inputs.
This review concerns implemented behavior and completed analysis. Work not
found in these checkouts may exist elsewhere.

## Where development stopped

The project developed into an interactive Yb-171 dynamics simulator, with
initial parameter-sweep tools. The main outstanding work is converting those
dynamics into an automated single-photon generation study with defined output
metrics and documented operating recommendations.

Original development milestones include the generic cavity model in August
2025, Rabi examples in October, Yb and dissipation work in November, batch/GPU
experiments in December, and the hyperfine and Gaussian pulse work in January
2026. The latest committed cleanup is February 2, 2026. Installation, CLI demo,
and the active eight-test suite were added in the recent portfolio cleanup;
they are not evidence that the original research analyses were completed.

| Week | Evidence in the project | Remaining work |
| --- | --- | --- |
| 1: Lambda atom and cavity | Implemented: `three_level_model.py` evolves atom/cavity populations with loss; later examples add driven Rabi dynamics. | Establish an end-to-end emitted-photon benchmark, including losses and per-attempt counting. |
| 2: Cooperativity and rate sweeps | Partial: steady-state g/kappa/gamma sweeps and plots exist in the week-02 report. | Replace the old efficiency proxy with output observables for the chosen generation protocol. |
| 3: Temporal dynamics and pulse shaping | Partial: independent Gaussian drive/repump controls, atomic traces, and cavity photon traces exist. | Export emitted wavepacket intensity, duration, and integrated emission; compare speed/efficiency systematically. |
| 4: Realistic structure, branching, purity | Substantial implementation: 16-state F,mF basis, 3j/6j couplings, decay/dephasing operators, and correlations. | Quantify cavity-output single-photon statistics and temporal coherence under pulses. |
| 5: Multidimensional optimization | Early infrastructure: a Cartesian parameter-grid runner exists. | Define objectives/constraints, search with the current pulse model, and report optima and trade-offs. |
| 6: Complete analysis framework | Partial: UI, CSV output, NPZ export, and a batch-result viewer. | Unify the solver used by UI and batch; add photon-generation metrics to each saved run. |
| 7: Final report and cleanup | Partial: physics notes, presentation source, skeleton report, and recent packaging/tests. | Produce the results report and reproducible parameter recommendations. No dedicated entanglement study was found. |

## How the simulator works

The Lambda process is laser excitation S-to-P followed by cavity-mediated
P-to-D emission. A repump laser also couples D-to-P. The Yb extension replaces
each manifold with its hyperfine sublevels: four S, four P, and eight D states.
Two truncated cavity Fock spaces are tensored with this atomic basis.

The Hamiltonian contains rotating-frame detunings, Zeeman terms, drive and
repump transitions weighted by angular-momentum factors, and cavity coupling.
Separate Gaussian envelopes make the drive and repump terms time dependent.
Collapse operators represent spontaneous emission, cavity leakage, and
dephasing. QuTiP `mesolve` evolves the state; projectors give populations and
`a.dag() * a`/`b.dag() * b` give intracavity photon numbers. Sliders rebuild
and solve the model for new parameters.

## What the thesis clarifies

- Section 2.3, Eqs. 2.30–2.35, uses collapse operators proportional to
  `sqrt(2 gamma)` and `sqrt(2 kappa)`. The factor of two in the newer model is
  consistent with that convention.
- Section 2.2 uses a rotating frame referenced to the P state. Absence of
  absolute optical energies is expected. For the Yb extension, document which
  hyperfine transition defines each detuning. The thesis also includes the P
  Zeeman term (Eq. 2.18), while the active implementation builds S and D terms.
- The thesis adds both cavity couplings, `H_a + H_b` (Eq. 2.26). The active
  builder selects the left or right coupling with its polarization branch;
  its pi branch adds no cavity coupling. This is a restricted implementation,
  not yet the complete bimodal model described in the thesis.
- Section 2.5, Eqs. 2.47–2.48, converts intracavity occupancy into output flux
  and integrates it. In the newer code's convention, useful output has flux
  `2 kappa_out <n(t)>`; total cavity decay can include undesired loss too.
  That integral is a mean emitted photon count. It equals a single-photon
  generation probability only for a protocol emitting at most one photon.

## Most useful next steps

1. Extract a common headless pulse solver and use it for the UI and batch runs.
2. Define the intended cavity modes, initial state, pulse sequence, and useful
   output versus other losses. Complete the thesis-to-code Hamiltonian mapping.
3. Add emitted intensity, integrated count, emission timing, per-pulse photon
   counting, and time-dependent cavity-field correlations. Current UI
   correlations use the atomic S/P fluorescence operator and a static
   Hamiltonian at the final state; they do not establish pulsed cavity-photon
   purity. Photon-number purity and temporal-mode indistinguishability should
   be reported separately.
4. Run a small pulse-width/drive/detuning grid and show efficiency-duration and
   coherence trade-offs. Use those results to choose the later optimization.

The old sweep efficiency uses `a * a.dag()` rather than the photon-number
operator `a.dag() * a`, and likewise uses a reversed atomic operator product.
It is not the thesis's emitted-photon efficiency calculation. The batch runner
uses the older constant-drive backend and saves atomic populations, not cavity
emission metrics. The local `results.csv` contains 44 exploratory runs; no batch
time-series directory was found. `examples/run_precompute.py`, despite its
name, currently plots unrelated physical-qubit estimates and does not perform
simulation precomputation.
