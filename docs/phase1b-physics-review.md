# Phase 1b physics corrections and checks

Historical checkpoint: current values, paths and review status are superseded
by [Phase 1c and the final review](portfolio-review.md).

The pre-correction Step 1 audit (answers transferred to the final PR) remains the baseline. This work
continues on `portfolio-cleanup` and stops before Steps 2–7. No source PDFs,
level-diagram images, internal reference documents, or generated demo arrays
were committed. Atomic inputs and source qualifications are catalogued in
[physics-references.md](physics-references.md).

## Changes, before and after

| Item | Before | After / scope |
| --- | --- | --- |
| DP wavelength | 2.400 µm | 2.438 µm, as requested; `OMEGA_2400`/`omega_2400` → `OMEGA_2438`/`omega_2438`; `A_2400_to_D32` → `A_2438_to_D32`; transition key `2400` → `2438` |
| Stored D hyperfine input | `HFS_A_GHZ`: 0.86372 GHz; `nu_D32`: 0.86 GHz | Both 0.86375 GHz; no diagonal hyperfine term added; A-versus-splitting qualification below |
| P1/2 lifetime | 8.07 ns | 8.12 ns, PRA 80, 022502 (2009) |
| Auxiliary 3D[3/2]1/2 | J=5/2; splitting 2.21 GHz | J=1/2; splitting 2.2095 GHz; lifetime remains 37.7 ns; absent from active basis |
| P→D branching | 0.005 | 0.00501, published central value from PRA 76, 052314 (2007) |
| P→S branching | 0.995 | 0.99499, complement of P→D under the retained two-branch approximation |
| SD E1-table entry | Unsourced factor 0.982; inferred element 0.00174312829024 ea0 in the active table | Removed; SD remains unused 435.5 nm E2 metadata. The historical helper's analogous entry was removed too; no replacement E2 estimate invented. |
| Unused linewidths | `LINEWIDTH_SP=2*pi*21.57e6`, `LINEWIDTH_DP=2*pi*1.482e6`, `LINEWIDTH_SD=2*pi*136e3` rad/s | Deleted, including historical exports; active decay inputs unchanged |
| Nuclear g metadata | S negative, P/D positive, magnitude 0.0005362 | All negative; active `G_I=-0.0005362` unchanged |
| Hyperfine phase | Parity of `2*phase_exp` | Parity of the integer `phase_exp`; signed reference regression added |
| Zeeman checks | No reference-slope checks | Four signed weak-field coefficients checked to within 2% |
| Fock truncation | Fixed default dimension 2 per mode | Default still 2; optional `fock_dimension` permits dimension-3 comparison; measured results below |
| Amplitude helper | Docstring claimed `Omega=E0`, while returning `E0/2` | Docstring and exact mapping corrected; all amplitude factors unchanged |
| DP pulse naming | “Repump” without a pathway qualification | “D3/2→P1/2 reset coupling (simplified stand-in for the 935 nm repump via 3D[3/2]1/2)”; labels shortened to “reset” after that definition |
| Compatibility | `omega_rep`, `rep_*`, `repump_MHz` | Canonical `omega_reset`, `reset_*`, `reset_MHz`; old names remain compatibility aliases; serialized UI/cache keys preserved |
| References | Incorrect cavity-thesis attribution; atomic diagram attributions | Thomas Walker, Sussex PhD, June 2020; corrected equation references and published atomic-source catalogue with precision caveats |
| Model status | Incomplete limitations and amplitude mapping | Explicit diagonal-energy, P Zeeman, cavity-line, polarization, decay-normalization, and truncation limitations |
| README | Ambiguous cavity/reset description | Explicit 2.438 µm DP cavity and simplified reset; three deferred cavity-output tasks listed under Next steps; existing image marked as pre-correction; full landing-page rewrite and figure replacement deferred |
| License | No LICENSE | MIT, copyright Felix Velasquez 2026, as instructed |

Duplicate requested constants and hyperfine-phase code in
`hamiltonian_utils.py` and the historical pre-time-dependent Yb example were
corrected too. These remain separate, non-equivalent historical models;
archiving and software consolidation have not started. The user's existing
local edit to `docs/AUDIT.md` was preserved and excluded from these commits.

## Nuclear sign and provenance

The sign is **negative**. arXiv:2509.04416, Eqs. 16–19, defines the nuclear
term with a positive `g_I I·B` coefficient and the Bohr magneton, giving
`g_I=-0.0005362`. This matches the active Hamiltonian convention; no active
Zeeman coefficient changed when the metadata signs were made consistent.
The reference treats neutral Yb. Its convention and generic formulas are
used here, not its neutral-atom level tables as ion spectroscopy data.
See the [source catalogue](physics-references.md).

## Reduced elements and demo changes

The baseline was measured before code changes at `bdb4765`. The A1–A3
checkpoint was measured at `17e2946`, before the branching and phase changes.
The final model includes the branching central-value update and corrected
phase. All comparisons use the unchanged default demo pulse sequence,
dimension 2 per mode, QuTiP 5.2.0, and `solver_options()`.
These are numerical results from this checkout, not literature measurements.

Reduced dipoles in ea0, calculated by `_rme_eq54` from the inputs:

| Element | Baseline | After A1–A3 only | Final corrected model |
| --- | --- | --- | --- |
| SP | 2.47813125525 | 2.47048975924 | 2.47047734469 |
| DP | 2.90768111381 | 2.96783134735 | 2.97079769626 |

A1–A3 change SP by -0.308357% and DP by +2.068667% relative to baseline.
Including the branching central-value update gives -0.308858% and +2.170684%.
The D hyperfine metadata correction does not affect this demo because
hyperfine offsets are absent from the Hamiltonian.

Population and occupation values from `simulate_demo()`; “final” is at
2 microseconds, “peak” is the maximum over the 201 exported sample times:

| Observable | Statistic | Baseline | After A1–A3 only | Final corrected model |
| --- | --- | --- | --- | --- |
| P_S | peak | 1 | 1 | 1 |
| P_S | final | 0.775429538604 | 0.770164365585 | 0.770088579615 |
| P_P | peak | 0.385394902928 | 0.381837849698 | 0.381664546664 |
| P_P | final | 0.0140636137094 | 0.01664237687 | 0.016781698954 |
| P_D | peak | 0.215145493268 | 0.219608489563 | 0.219644534776 |
| P_D | final | 0.210506847687 | 0.213193257545 | 0.213129721431 |
| N_left | peak | 0.141937252704 | 0.146849793149 | 0.147079126479 |
| N_left | final | 0.0623092542532 | 0.0621640948799 | 0.0620784653894 |
| N_right | peak | 0 | 0 | 0 |
| N_right | final | 0 | 0 | 0 |

Maximum absolute difference between complete sampled traces and baseline:

| Observable | A1–A3 only | Final model |
| --- | --- | --- |
| P_S | 0.00762695200395 | 0.00791635572072 |
| P_P | 0.00559891738641 | 0.00572512367489 |
| P_D | 0.00720581026422 | 0.00753268558797 |
| N_left | 0.00500894552200 | 0.00523320260001 |
| N_right | 0 | 0 |

The common source of these numbers is the recorded before/after execution
of `simulate_demo()` and `REDUCED_E1_ELEMENTS`. To inspect current outputs:

```bash
uv run caipe-sim --demo
uv run python - <<'PYCODE'
import numpy as np
from caipe_sim import yb171_simulator as m
from caipe_sim.demo import simulate_demo
print(m.REDUCED_E1_ELEMENTS)
for key, values in simulate_demo().items():
    if key.startswith(('P_', 'N_')):
        print(key, 'peak', np.max(values), 'final', values[-1])
PYCODE
```

## Phase-bug verdict

**Confirmed and corrected.** The new signed reference test had four failing
odd-exponent cases and three passing even-exponent cases before the fix;
all seven pass afterward. The test includes explicit evaluated 3j and 6j
factors and expected signed full coefficients. It does not generate expected
values using the implementation under test. Convention: Steck's
[Rb data notes](https://steck.us/alkalidata/rubidium87numbers.pdf), Sec. 3.3,
translated to the Racah normalization and absorption ordering documented
in [physics-references.md](physics-references.md).

The exponent is `Jp+I+F+1 = F+2` for the retained manifolds. All nonzero
coefficients from lower F=1 change sign:

- S(F=1,mF) → P(F'=0 or 1,mF'): 9 allowed polarized transitions.
- D(F=1,mF) → P(F'=0 or 1,mF'): 9 allowed polarized transitions.

All allowed q and mF combinations in those groups change; S(F=0) and
D(F=2) do not. Sources: enumerating `E1_DATA_HF` and the fixed phase formula.
Collapse magnitudes are unchanged because they use absolute coefficients.
At fixed corrected atomic inputs, the before/after-phase maximum difference
was exactly zero in every sampled demo population and cavity occupation.
For this Hamiltonian the flips amount to rephasing the lower F=1 basis
states; this agreement is expected. It does not make the old coefficients
correct under the documented convention.

## Zeeman reference checks

Source: executed `lande_g_factors` with the existing spin/orbital/nuclear
inputs and `g_F*MU_B*1e-4/(2*pi*1e6)`. References are the rounded values
specified in Phase 1b; the angular-momentum formula and magnetic convention
are sourced in [physics-references.md](physics-references.md).

| Manifold and F | Calculated MHz/G per mF | Requested reference | Tolerance |
| --- | --- | --- | --- |
| S1/2, F=1 | 1.40087232997 | 1.4 | 2% |
| P1/2, F=1 | 0.465622236848 | 0.47 | 2% |
| D3/2, F=1 | 1.39899383338 | 1.4 | 2% |
| D3/2, F=2 | 0.839096108567 | 0.84 | 2% |

All pass. The S/D checks also verify the actual diagonal energy increments.
The P check verifies only the g_F calculation: there is no P Zeeman term
in the active model.

## Fock dimension 3 comparison

The [reproducible script](../scripts/check_fock_truncation.py) compares the
same corrected demo at dimensions 2 and 3 per mode. See
[fock-convergence.md](fock-convergence.md) for settings and individual values.
Maximum manifold-population difference: **0.00863983443916**, in D
(about 0.864 percentage points). Maximum left-mode photon-number difference:
**0.00875758113188**. Right-mode difference: zero. The default remains 2.
These changes are not small compared to solver tolerances; neither dimension
is certified as converged for a quantitative emission study.

## P1/2 Zeeman term: proposed work only

Construct a diagonal P operator with `g_F(P)*MU_B*B_T*mF`; F=0 has zero
first-order shift. Include it with S and D in the tensor-lifted static
Hamiltonian for both UI and demo. Currently `add_bare_terms_to_systems`
accepts only S and D. Define the common P reference and specific hyperfine
transition detunings before adding level energies: distinct sublevel shifts
cannot all be removed as a common offset. Add tests for slopes, mF symmetry,
zero-field and F=0 limits, and detuned Rabi resonances. No new constant is
needed for the present weak-field approximation, but finite-field dynamics
would change. This was not implemented; see [model-status.md](model-status.md).

## New or clarified concerns, left unresolved

- The D value requested as A is sourced to a paper reporting an F-level
  splitting of 0.86(2) GHz. For I=1/2,J=3/2, splitting is `2*A`, with
  the sign determined by level ordering. The requested 0.86375 is retained
  as adopted metadata, not silently converted or inserted in energies.
- The supplied wavelength paper's Table III reports 369.52604(6) nm for
  S(F=1)–P(F=0), whereas the retained code input is 369.5262 nm. That value
  was not changed because it was not an authorized atomic-data correction.
  A direct published source for the adopted DP and E2 wavelength precision
  is also still needed. These qualifications are explicit in the source guide.
- arXiv:2509.04416 is about neutral Yb. Generic magnetic formulas and the
  nuclear convention are useful here; neutral level-table g_J and hyperfine
  data must not be substituted for ion data. The auxiliary metadata's
  retained S,L fields are not an LS-coupled configuration consistent with
  its corrected J, so it must not be used to build extra couplings as-is.
- The existing UI dictionary starts with `system='pi'`, while the radio
  selector starts at left and controls the solve. This was documented,
  not changed. Default cavity coupling is still zero.
- A similar doubled-parity pattern remains in `_clebsch_gordan` in the
  historical/bosonic code. The active hyperfine E1 construction uses
  Wigner symbols directly; the historical helper needs its own convention
  tests before any broader fix. No speculative change was made.
- Hyperfine diagonal offsets, P Zeeman shifts, normalized physical decay
  rates, and intermediate-level repump dynamics remain absent. Simultaneous
  two-polarization coupling, pulse-resolved correlations, and useful-output
  loss separation remain outside scope and are listed in README Next steps.

## Validation and stopping point

Every committed fix was preceded by successful `uv sync`, `uv run pytest`,
and `uv run caipe-sim --demo`. The final suite contains **20 passing cases**:
the original 8, 7 signed coefficient cases, 4 Zeeman cases, and 1 reset-alias
compatibility check. The intentional pre-fix phase test failure was resolved
before committing or continuing. The final Fock comparison script ran
successfully. The renamed UI initialized headlessly and its labels were
inspected; desktop interaction was not exercised.

No default pulse or cavity dimension was changed. No amplitude half-factor
was removed. No normalized decay model, P Zeeman term, or deferred cavity
feature was added. The existing README result image is explicitly labeled pre-correction; its
replacement is deferred to the requested later figure step. Baseline audit
retained. No push, PR, archival operation,
or Steps 2–7 work was performed; wait for the user's go-ahead.
