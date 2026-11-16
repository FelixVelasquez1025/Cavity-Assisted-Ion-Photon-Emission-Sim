# Professor-review preparation

The active model and scientific conventions are described in
[model-status.md](model-status.md) and [physics-references.md](physics-references.md).
The approved [Phase 1b report](phase1b-physics-review.md) is a historical
checkpoint; the following changes supersede its unresolved items.

## Phase 1c corrections

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

The pre-merge analytic solver check at dimension 2 reduced the maximum
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

## Repository changes

Historical code, examples, notebooks and tests are archived with git moves.
Their optional JAX/notebook/custom-atomic dependencies were removed from the
active project metadata. The active import guard excludes historical
modules and imported `_clebsch_gordan` helpers. The active module's separate
local bosonic helper remains; its default hyperfine path uses Wigner
symbols directly. No further historical phase correction was made.

The README leads with analytic Rabi validation and residuals, links current
demo dynamics, provides sourced/demo input tables, and states limitations.
`make_readme_figures.py` shares the Rabi experiment with the test and uses
the same Fock default as the demo/UI. CI runs the active suite, demo, figure
script and truncation comparison using the lockfile. MIT licensing is in
[LICENSE](../LICENSE), copyright Felix Velasquez 2026.

## Audit

The final PR description carries the full six audit answers. The retained
basis, in code order (manifold,F,mF), is:

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

## Validation and GitHub handoff

Local sync, tests and demo passed after every committed step. The README's
five commands also passed in an isolated clone: locked install, **22 tests**,
demo, figures and the dimension 2/3/4 comparison. The wheel and source
distribution built successfully and were inspected to exclude `archive/`.
Both generated figures were visually checked. The renamed UI initialized
headlessly; desktop interaction was not exercised.

Hygiene removed the pre-correction image, generated Hamiltonian matrices,
old generated slide deck and stray editor-settings text. Generated NPZ,
`outputs/`, macOS metadata and the optional UI cache are ignored. No tracked
personal absolute paths, source PDFs or level diagrams remain. The Step 1
audit file is removed as requested; its six answers are transferred to the
PR description, alongside final remote fresh-clone and CI evidence.
No PR is merged by this work.

Merge the reviewed PR into `main`, then confirm `main` as GitHub's default
branch. Suggested description: “Open-system Yb-171 ion–cavity simulation
with pulse shaping, atomic angular couplings and analytic validation.”
Suggested topics: `trapped-ions`, `ytterbium`, `cavity-qed`, `qutip`,
`quantum-optics`. These metadata changes are left for the repository owner.
