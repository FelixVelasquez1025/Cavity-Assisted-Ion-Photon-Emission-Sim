# Validation notes

Current conventions and sources are in [model status](model-status.md) and
[physics references](physics-references.md). Earlier numerical checkpoints
are retained below with their original inputs; they are not current defaults.

## Current inputs and numerical checks

- D splitting remains 0.86375 GHz; D A/h is now 0.431875 GHz, half the
  splitting. S/P have J=1/2, so A/h equals splitting. Their metadata now
  matches the existing precise S/P model inputs rather than rounded copies.
- UI defaults are left polarization, cavity coupling input 0.5 MHz and
  loss input 0.1 MHz; the radio selector reads the dictionary value.
- S–P wavelength is now the vacuum S(F=1)↔P(F=0) component at 369.52604 nm
  from PRA 83, 013406 (2011), Table III. This changes SP's reduced dipole
  from 2.470477344688475 to 2.4704757401621955 ea0.
- NIST ASD v5.12 (accessed 2026-11-16) gives S=0.00, D=22960.80 and
  P=27061.82 cm⁻¹. Vacuum Ritz wavelengths from these levels are
  2438.417759484226 nm (DP) and 435.5248946029755 nm (SD). DP's reduced
  element changes from 2.9707976962590688 to 2.971561313205892 ea0.
  These are fine-structure Ritz values, not hyperfine-component measurements.
- All atomic provenance and precision limitations are in the source guide;
  these calculations add no new calibrated experimental parameter.

The extended [Fock check](fock-convergence.md) now compares dimensions 2, 3,
4 and 5. The 3−4 comparison failed, but every 4−5 difference passes 1e-3:
maximum population difference 0.000241316721655 and photon-number difference
0.0000753242015548. The demo/UI default is therefore **4**, shared by both
README figures and the analytic test. This is a sequence-specific check.

The analytic solver check at dimension 2 reduced the maximum
residual from 1.99684274e-5 to 1.64187332e-7 (about 122×) with atol=1e-10,
rtol=1e-8. At the new default dimension 4 the residual is 3.40955548e-7;
the test/figure comparison uses absolute tolerance 3.5e-6 and rtol=0
(about 10× that residual). Only the analytic experiment uses tighter solver
tolerances; the demo/truncation solver remains unchanged. Source: executed
comparison/figure scripts and `run_rabi_validation`; no experimental
accuracy is implied.

The Zeeman test explicitly cites Foot §6.3 and writes out the analytic
Landé g_F formula, including the signed nuclear term in Bohr-magneton
units. Its reference is the formula, not a level diagram.

## Basis and model scope

The retained basis, in code order (manifold,F,mF), is:

```text
S: (0,0), (1,-1), (1,0), (1,+1)
P: (0,0), (1,-1), (1,0), (1,+1)
D: (1,-1), (1,0), (1,+1), (2,-2), (2,-1), (2,0), (2,+1), (2,+2)
```

The cavity spaces are circular polarizations on D–P, with one selected
coupling per run. Drive is S–P; the direct DP reset is a simplified stand-in
for the 935 nm pathway, not an explicit intermediate-level repump. UI and
demo input defaults are documented in code and README; fixed atomic inputs
and derived elements are catalogued with citations in the reference guide.

Approximations: RWA, linear weak-field S/D Zeeman shifts, truncated Fock
spaces and Markovian losses. Diagonal hyperfine and P Zeeman energies,
normalized physical decay rates and motional dynamics are absent. The
three deferred cavity tasks are simultaneous polarization coupling,
pulse-resolved correlations, and useful-output flux/loss separation.

## Atomic-data correction checkpoints

The following table records an earlier correction checkpoint. Current values
above supersede its intermediate wavelength and D A/h entries.

| Item | Before | After / scope |
| --- | --- | --- |
| DP wavelength | 2.400 µm | 2.438 µm; `OMEGA_2400`/`omega_2400` → `OMEGA_2438`/`omega_2438`; `A_2400_to_D32` → `A_2438_to_D32`; transition key `2400` → `2438` |
| Stored D hyperfine input | `HFS_A_GHZ`: 0.86372 GHz; `nu_D32`: 0.86 GHz | Both 0.86375 GHz; no diagonal hyperfine term added; current A/h = 0.431875 GHz, as explained above |
| P1/2 lifetime | 8.07 ns | 8.12 ns, PRA 80, 022502 (2009) |
| Auxiliary 3D[3/2]1/2 | J=5/2; splitting 2.21 GHz | J=1/2; splitting 2.2095 GHz; lifetime remains 37.7 ns; absent from active basis |
| P→D branching | 0.005 | 0.00501, published central value from PRA 76, 052314 (2007) |
| P→S branching | 0.995 | 0.99499, complement of P→D under the retained two-branch approximation |
| SD E1-table entry | Unsourced factor 0.982; inferred element 0.00174312829024 ea0 in the active table | Removed; SD remains unused 435.5 nm E2 metadata. The historical helper's analogous entry was removed too; no replacement E2 estimate invented. |
| Unused linewidths | `LINEWIDTH_SP=2*pi*21.57e6`, `LINEWIDTH_DP=2*pi*1.482e6`, `LINEWIDTH_SD=2*pi*136e3` rad/s | Deleted, including historical exports; active decay inputs unchanged |
| Nuclear g metadata | S negative, P/D positive, magnitude 0.0005362 | All negative; active `G_I=-0.0005362` unchanged |
| Hyperfine phase | Parity of `2*phase_exp` | Parity of the integer `phase_exp`; signed reference regression added |
| Zeeman checks | No reference-slope checks | Four signed weak-field coefficients checked to within 2% |
| Amplitude helper | Docstring claimed `Omega=E0`, while returning `E0/2` | Docstring and exact mapping corrected; all amplitude factors unchanged |
| DP pulse naming | “Repump” without a pathway qualification | “D3/2→P1/2 reset coupling (simplified stand-in for the 935 nm repump via 3D[3/2]1/2)”; labels shortened to “reset” after that definition |
| Compatibility | `omega_rep`, `rep_*`, `repump_MHz` | Canonical `omega_reset`, `reset_*`, `reset_MHz`; old names remain compatibility aliases; serialized UI/cache keys preserved |

## Nuclear sign and provenance

The sign is **negative**. arXiv:2509.04416, Eqs. 16–19, defines the nuclear
term with a positive `g_I I·B` coefficient and the Bohr magneton, giving
`g_I=-0.0005362`. This matches the active Hamiltonian convention; no active
Zeeman coefficient changed when the metadata signs were made consistent.
The reference treats neutral Yb. Its convention and generic formulas are
used here, not its neutral-atom level tables as ion spectroscopy data.
See the [source catalogue](physics-references.md).

## Reduced elements and demo changes

The baseline is the earlier atomic-data model (`bdb4765`). The intermediate
checkpoint (`17e2946`) applies only the DP wavelength, D splitting and P
lifetime corrections, before the branching and phase updates. The corrected-data
checkpoint includes the branching central-value update and corrected phase,
but predates the current NIST/SP wavelength refinements. All comparisons use the unchanged default demo pulse sequence,
dimension 2 per mode, QuTiP 5.2.0, and `solver_options()`.
These are numerical results from this checkout, not literature measurements.

Reduced dipoles in ea0, calculated by `_rme_eq54` from the inputs:

| Element | Baseline | Intermediate data | Corrected-data checkpoint |
| --- | --- | --- | --- |
| SP | 2.47813125525 | 2.47048975924 | 2.47047734469 |
| DP | 2.90768111381 | 2.96783134735 | 2.97079769626 |

The intermediate data change SP by -0.308357% and DP by +2.068667% relative to baseline.
Including the branching central-value update gives -0.308858% and +2.170684%.
The D hyperfine metadata correction does not affect this demo because
hyperfine offsets are absent from the Hamiltonian.

Population and occupation values from `simulate_demo()`; “final” is at
2 microseconds, “peak” is the maximum over the 201 exported sample times:

| Observable | Statistic | Baseline | Intermediate data | Corrected-data checkpoint |
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

| Observable | Intermediate data | Corrected-data checkpoint |
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
inputs and `g_F*MU_B*1e-4/(2*pi*1e6)`. The analytic reference is Foot, Atomic Physics (2005), §6.3. Numerical
magnetic inputs and conventions are catalogued in
[physics-references.md](physics-references.md).

| Manifold and F | Calculated MHz/G per mF | Comparison slope | Tolerance |
| --- | --- | --- | --- |
| S1/2, F=1 | 1.40087232997 | 1.4 | 2% |
| P1/2, F=1 | 0.465622236848 | 0.47 | 2% |
| D3/2, F=1 | 1.39899383338 | 1.4 | 2% |
| D3/2, F=2 | 0.839096108567 | 0.84 | 2% |

All pass. The S/D checks also verify the actual diagonal energy increments.
The P check verifies only the g_F calculation: there is no P Zeeman term
in the active model.

## Fock dimension 3 comparison

An earlier run compared the corrected-data checkpoint demo at dimensions 2
and 3 per mode, before the NIST/SP wavelength refinements. The current
[reproducible script](../scripts/check_fock_truncation.py) compares dimensions 2–5. See
[fock-convergence.md](fock-convergence.md) for settings and individual values.
Maximum manifold-population difference: **0.00863983443916**, in D
(about 0.864 percentage points). Maximum left-mode photon-number difference:
**0.00875758113188**. Right-mode difference: zero. The default was 2 at that checkpoint; it is now 4.
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


## Remaining physics qualifications

- The adopted D splitting 0.86375 GHz has more digits than the cited 0.86(2) GHz measurement; A/h is now correctly half that splitting. The current SP wavelength and NIST-derived DP/SD wavelengths supersede the historical checkpoint values.
- arXiv:2509.04416 treats neutral Yb: use its generic magnetic formulas and nuclear convention, not neutral level spectroscopy as ion data. The auxiliary level’s retained S,L fields do not form an LS configuration consistent with its corrected J, and must not be used to build extra couplings as-is.
- A doubled-parity pattern remains in the local bosonic `_clebsch_gordan` helper and archived code. The active hyperfine E1 construction uses Wigner symbols directly. Separate convention tests are needed before changing that helper.
- Hyperfine diagonal offsets, P Zeeman shifts, normalized physical decay rates and intermediate-level repump dynamics remain absent. Simultaneous polarization coupling, pulse-resolved correlations and output-loss separation remain future work.

## Software validation scope

The suite checks selection rules, Hermiticity, population conservation, analytic Rabi dynamics, signed hyperfine coefficients, Zeeman slopes, hyperfine metadata, compatibility aliases and package boundaries. Historical models under `archive/` are excluded from the active distribution and tests. No active package imports historical `_clebsch_gordan` helpers; the distinct local bosonic helper still exists.

Both direct-file and CLI UI launching have completed initial simulation and widget creation with Matplotlib Agg. An isolated subprocess regression checks file loading without package context from another working directory. Headless checks do not establish desktop interaction behavior. No experimental calibration or emission-performance claim follows from these numerical checks.
