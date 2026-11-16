# CAIPE-sim audit for professor review

Step 1 only. Audited the working tree based on `Yb171_Rabi_Flopping` at
`909cc1c`, on the new `portfolio-cleanup` branch. No model changes have been
made. This report describes implemented behavior, not experimental validation.

## 1. Atomic basis

The active basis is `HF_BASIS_ORDER` in
[yb171_simulator.py](../src/caipe_sim/yb171_simulator.py#L112).
The nuclear spin is `I_NUCLEAR = 0.5`. Entries below are in code order;
angular-momentum labels are dimensionless (in units of hbar).

| Index | Manifold | F | mF |
| --- | --- | --- | --- |
| 0 | 6s 2S1/2 | 0 | 0 |
| 1 | 6s 2S1/2 | 1 | -1 |
| 2 | 6s 2S1/2 | 1 | 0 |
| 3 | 6s 2S1/2 | 1 | +1 |
| 4 | 6p 2P1/2 | 0 | 0 |
| 5 | 6p 2P1/2 | 1 | -1 |
| 6 | 6p 2P1/2 | 1 | 0 |
| 7 | 6p 2P1/2 | 1 | +1 |
| 8 | 5d 2D3/2 | 1 | -1 |
| 9 | 5d 2D3/2 | 1 | 0 |
| 10 | 5d 2D3/2 | 1 | +1 |
| 11 | 5d 2D3/2 | 2 | -2 |
| 12 | 5d 2D3/2 | 2 | -1 |
| 13 | 5d 2D3/2 | 2 | 0 |
| 14 | 5d 2D3/2 | 2 | +1 |
| 15 | 5d 2D3/2 | 2 | +2 |

The manifold assignments also appear in [Yb171.py](../src/caipe_sim/Yb171.py).
Both the UI and demo initialize S, F=0, mF=0 and cavity vacuum. The UI plots
S,F=0; P,F=1; and S,F=1 groups, not full manifold populations. The demo plots
full S/P/D populations. Sources: `interactive_populations` and
[simulate_demo](../src/caipe_sim/demo.py).

## 2. Cavity modes

The two Fock spaces represent circular polarization modes on D3/2 <-> P1/2,
with a common coupling input and cavity detuning, not separately tunable
frequencies. The shared frequency term assumes no polarization splitting.
The code's convention labels absorption q=+1 as left and q=-1 as right.

From `H_cavity_DP_modes` in
[yb171_simulator.py](../src/caipe_sim/yb171_simulator.py#L751):

```python
- a: left-circular (σ+), accumulates q=+1 terms.
- b: right-circular (σ−), accumulates q=−1 terms.
```

The active hyperfine implementation `H_cavity_DP_modes_hf` explicitly builds:

```python
S_p = comps[+1]
S_m = comps[-1]
H_a = g * (tensor(a, I_b, S_p) + tensor(a.dag(), I_b, S_p.dag()))
H_b = g * (tensor(I_a, b, S_m) + tensor(I_a, b.dag(), S_m.dag()))
return {'left': H_a, 'right': H_b}
```

However, `H_interaction_with_cavity_hf` adds only `Hc['left']` to the left
branch and only `Hc['right']` to the right branch; pi has no cavity coupling.
`H0_cavity` applies `(delta_c - delta_rep)` to the sum of the two photon-number
operators. Both spaces exist, but both interactions do not evolve together
in a selected run. The default UI uses pi with zero cavity coupling; the
demo selects left. Sources: the named builders and `simulate_demo`.

## 3. Driven transitions and repumping

| Interaction | Implemented transition | Wavelength input | Source |
| --- | --- | --- | --- |
| Drive | 6s 2S1/2 <-> 6p 2P1/2 | 369.5262 nm | `OMEGA_369`; `omega_369` |
| Repump | 5d 2D3/2 <-> 6p 2P1/2 | 2.400 micrometres | `OMEGA_2400`; `omega_2400` |
| Cavity | 5d 2D3/2 <-> 6p 2P1/2 | Same DP transition | `H_cavity_DP_modes_hf` |
| Metadata only | 6s 2S1/2 <-> 5d 2D3/2 | 435.5 nm | `OMEGA_435`; `omega_435`; not an active laser/collapse channel |

Sources: [atomic inputs](../src/caipe_sim/yb171_simulator.py#L102),
[Yb171 transition metadata](../src/caipe_sim/Yb171.py#L47), and
`precompute_e1_coeffs_hyperfine`. Drive and repump are independent Gaussian
envelopes, with the same selected polarization in the current UI/demo.

**There is no modeled 935 nm repump**, either as an effective direct channel
or through an intermediate level. The implemented repump directly couples D
to the existing P manifold. `Yb171.py` contains lifetime/splitting metadata
labeled `3D[3/2]1/2`, but that level is not included in the active basis or
Hamiltonian. It must not be described as an implemented experimental repump.

## 4. Parameters, defaults, and provenance

There are separate UI and demo configurations. Values below are copied from
the cited code. MHz inputs are multiplied by `2*pi*1e6` for the solver;
microseconds become seconds and gauss becomes tesla. The Hamiltonian uses
angular-frequency units with hbar set to unity.

### User-facing model inputs

All values in this table are **illustrative**, not literature-calibrated or
fit to experimental data. Sources: `params` in
[interactive_populations](../src/caipe_sim/yb171_simulator.py#L1155) and
[simulate_demo](../src/caipe_sim/demo.py#L12).

| Parameter | UI default | Headless demo | Units / meaning |
| --- | --- | --- | --- |
| `B_G` | 1.0 | 0.0 | G |
| `delta_drive_MHz` | 0.0 | 0.0 | MHz; drive detuning |
| `delta_rep_MHz` | 0.0 | 0.0 | MHz; repump detuning |
| `delta_c_MHz` | 0.0 | 0.0 | MHz; cavity detuning |
| `g_cav_MHz` | 0.0 | 0.5 | MHz; cavity input before transition weighting |
| `amp_driv_MHz` | 5.0 | 5.0 | MHz-scale drive input; see amplitude convention below |
| `amp_rep_MHz` | 3.0 | 3.0 | MHz-scale repump input; see amplitude convention below |
| `drive_t0_us` | 5.0 | 0.6 | microseconds; Gaussian centre |
| `drive_sigma_us` | 1.0 | 0.2 | microseconds; Gaussian standard deviation |
| `rep_t0_us` | 5.0 | 1.1 | microseconds; Gaussian centre |
| `rep_sigma_us` | 1.0 | 0.25 | microseconds; Gaussian standard deviation |
| `gamma_SP` | 0.01 | 0.01 | MHz input; P -> S collapse scale |
| `gamma_DP` | 0.01 | 0.01 | MHz input; P -> D collapse scale |
| `gamma_SD` | 0.000136 | 0.000136 | MHz input; S/D manifold dephasing, not D -> S decay |
| `kappa_MHz` | 0.0 | 0.1 | MHz input; loss scale common to both modes |
| Polarization `system` | pi | left | Discrete branch selection |
| Initial atomic state | S,F=0,mF=0 | S,F=0,mF=0 | Pure state |
| Initial cavity occupation | vacuum in both | vacuum in both | Photon number |

`rabi_frequencies` halves each amplitude input; `_build_systems_hf` applies
another factor of one-half and the full transition coefficient (including
the reduced matrix element in atomic units). Thus the slider amplitude is
not directly the transition-specific Rabi frequency in Foot's two-level
formula. `g` is also multiplied by the DP transition coefficients. The
Gaussian coefficients default to zero if called without their arguments;
the configured runs supply all envelope arguments. Sources:
`rabi_frequencies`, `_build_systems_hf`, `omega_drive`, `omega_rep`.

### Fixed atomic inputs and derived quantities

The existing [reference guide](physics-references.md) calls the user's Yb
inputs separately verified, but does not supply a traceable literature
source for every value. The following inventory preserves that distinction:
it does not relabel undocumented atomic data as illustrative, or claim
independent literature verification. This is a provenance gap for review.

| Parameter | Value / definition | Units | Provenance and use |
| --- | --- | --- | --- |
| `I_NUCLEAR`, `Yb171.nuclear_spin` | 0.5 | Dimensionless | Yb atomic input, repeated in both modules |
| `G_S` | 2.00231930436092 | Dimensionless | Code attributes to arXiv:2509.04416; used by `lande_g_factors` |
| `G_L` | 0.99999679 | Dimensionless | Same code attribution; used by `lande_g_factors` |
| `G_I` | -0.0005362 | Dimensionless | Same code attribution; used by `lande_g_factors` |
| `HFS_A_GHZ`, S/P/D | 12.642812118466 / 2.105 / 0.86372 | GHz | Declared atomic inputs; absent from active diagonal energies |
| `TAU_P12`, `tau_P12` | 8.07e-9 | s | Comments say diagram-derived; diagram citation unspecified; used for SP/DP reduced elements |
| `TAU_D32`, `tau_D32` | 52.7e-3 | s | Same provenance; SD metadata/reduced element only |
| `tau_D3212` | 37.7e-9 | s | Same provenance; metadata only |
| `BR_P12_TO_S12`, `BR_P12_to_S12` | 0.995 | Dimensionless | Same provenance; SP reduced element and metadata A |
| `BR_P12_TO_D32`, `BR_P12_to_D32` | 0.005 | Dimensionless | Same provenance; DP reduced element and metadata A |
| Wavelengths for `OMEGA_*`, `omega_*` | 369.5262e-9 / 2.400e-6 / 435.5e-9 | m | Diagram-derived atomic inputs; angular frequencies calculated as `2*pi*c/lambda` |
| `nu_S12`, `nu_P12`, `nu_D32`, `nu_D3212` | 12.64e9 / 2.11e9 / 0.86e9 / 2.21e9 | Hz | `Yb171.py` says diagram-derived; metadata only, not active splittings |
| `LevelData.Ahfs`, S/P/D | `h*nu_S12`, `h*nu_P12`, `h*nu_D32` | J | Derived metadata; not used in active diagonal energies |
| `LevelData.Bhfs`, S/P/D | 0 | J | Zero quadrupole metadata; not used in active evolution |
| `LevelData.g_I`, S/P/D | -0.0005362 / +0.0005362 / +0.0005362 | Dimensionless | Header cites arXiv:2509.04416; active hyperfine builder instead uses common `G_I` |
| `g_s`, `g_p`, `g_d` | `lande_g_factor(S12/P12/D32)` | Dimensionless | Derived LS-coupling formula in `Yb171.py`; UI `g_S=g_s`, `g_D=g_d` are ignored by hyperfine builders |
| `g_F` | `lande_g_factors(F,I,J,L,S)[1]` | Dimensionless | Derived from `G_S`, `G_L`, `G_I`; F=0 returns zero |
| `A_P12_total` | `1/tau_P12` | inverse s | Derived metadata |
| `A_369_to_S12`, `A_2400_to_D32` | Respective branching fraction times `A_P12_total` | inverse s | Derived metadata; not automatically used as collapse rates |
| `A_D32_to_S12` | `1/tau_D32` | inverse s | Derived metadata; no corresponding active collapse |
| `REDUCED_E1_ELEMENTS`, SP/DP | `_rme_eq54(branch,omega,tau,Je=0.5)/_EA0` | Atomic dipole units | Derived from inputs above; used in laser, cavity, and decay weights |
| `REDUCED_E1_ELEMENTS`, SD | `_rme_eq54(0.982,OMEGA_435,TAU_D32,Je=1.5)/_EA0` | Atomic dipole units | Extra 0.982 input has no stated source; unused by active SP/DP dynamics |
| `LINEWIDTH_SP` | `2*pi*21.57e6` | rad/s | Unused constant; no nearby source |
| `LINEWIDTH_DP` | `2*pi*1.482e6` | rad/s | Unused constant; nearby comment says 852 nm, inconsistent with DP input above |
| `LINEWIDTH_SD` | `2*pi*136e3` | rad/s | Unused constant; comment conflicts with expression |
| `HBAR`, `MU_B`, `_EA0`, optical `c`, `epsilon_0`, `e`, `h` | SciPy constants; `MU_B=mu_B/hbar`, `_EA0=e*a0` | SI, with `MU_B` in rad/s/T | Library constants; no manually chosen numerical replacements |

Sources: [Yb171.py](../src/caipe_sim/Yb171.py),
[model constants](../src/caipe_sim/yb171_simulator.py#L79),
`lande_g_factors`, `_rme_eq54`, `REDUCED_E1_ELEMENTS`.
The `Yb171.py` header names “171Yb Reference Data (Kroeze, Kristensen,
Pucher, 2025), Table 5” and arXiv:2509.04416. This report records the code's
attribution; it has not verified the paper or extended that attribution to
the unspecified diagram values.

### Numerical settings (not physical measurements)

Sources: `interactive_populations`, `simulate_demo`, `solver_options`, and
the [analytic validation test](../tests/portfolio/test_yb171.py).

| Setting | UI | Demo / validation |
| --- | --- | --- |
| Atomic model | `is_bosonic=False` (hyperfine) | Same; optional UI bosonic path is separate |
| Fock dimension per mode | 2 | 2; levels n=0,1 only |
| Evolution window | 0 to 20 microseconds | Demo: 0 to 2 microseconds |
| Sampling | `density=40` samples/microsecond; minimum 50 samples | Demo: 201 time samples |
| Correlation window | `tau_max_us=0.5`, `n_tau=20` | Not evaluated in demo |
| Solver | `mesolve`, BDF | Same |
| Solver tolerances | `atol=1e-8`, `rtol=1e-6` | Same |
| Internal step allowance | `nsteps=200000` | Same |
| State storage / cache | States stored for correlation requests; `use_cache=False` | No stored-state analysis |
| Analytic Rabi test | Not the UI configuration | SP pi, S(F=0,mF=0) -> P(F=1,mF=0); no decay, repump, cavity, or bare terms; 201 samples over 0 to 2 microseconds; envelope coefficient peak `2*pi*1e6` rad/s, centre 1 microsecond, sigma 0.2 microseconds; population comparison `atol=2e-5` |

## 5. Approximations and physics concerns (not fixed)

- **Rotating-frame/RWA form:** laser raising/lowering terms and photon-number
  conserving atom-cavity exchange terms omit counter-rotating optical terms.
  The bare cavity term is `(delta_c-delta_rep)*(n_a+n_b)`; S and D use common
  manifold detunings. This form is consistent with an RWA model, but the Yb
  hyperfine reference transitions and detuning signs still need a complete
  mapping. Sources: `_build_systems_hf`, `H_cavity_DP_modes_hf`,
  `H0_cavity`, `eq_2_4_5_H_S_hf`, `eq_2_4_6_H_D_hf`.
- **Zeeman treatment:** linear diagonal `g_F*MU_B*B*mF` shifts for S and D
  in an F,mF basis, with no field-induced hyperfine mixing, quadratic Zeeman
  correction, or P Zeeman term in the active construction. The declared
  hyperfine constants do not enter these energies. Having hyperfine state
  labels therefore does not establish hyperfine-resolved resonance physics.
- **Restricted cavity:** common detuning/coupling parameters and only one
  interacting polarization per selected run; no mode splitting, motional
  dynamics, or position-dependent coupling. Small Fock truncation needs
  convergence checks. Source: active builders and `build_and_solve`.
- **Markovian Lindblad losses:** separate P->S and P->D lowering operators
  with `sqrt(2*gamma)*abs(coeff)` amplitudes, both cavity losses with
  `sqrt(2*kappa)`, and S/D projector dephasing with `sqrt(2*gamma_SD)` after
  unit conversion. No active D->S radiative channel or intermediate-level
  repump. Rates are illustrative independent inputs, not automatically
  constrained by the metadata lifetimes. Sources: `precompute_lowering_tensors_hf`
  and `build_collapse_operators`.
- **Branch normalization needs review:** decay coefficients include reduced
  dipole elements as well as angular factors. Thus `gamma_SP`/`gamma_DP`
  cannot simply be identified with total physical branch rates, and metadata
  branching fractions do not by themselves establish normalized decay
  probabilities. The factor of two is intentional per
  [physics-references.md](physics-references.md); the weighted channel sums
  still need validation.
- **Possible phase bug:** `hyperfine_angular_factor` tests the parity of
  `2*phase_exp` rather than `phase_exp`. For the present integer exponents,
  the implemented test always returns a positive phase, including cases where
  the exponent is odd. This warrants checking against the intended recoupling
  convention. Existing tests check selection rules/Hermiticity, not relative
  transition phases. No correction made.
- **Amplitude convention:** the extra halving in `rabi_frequencies` conflicts
  with its own “Omega = E0” docstring, and full dipole coefficients multiply
  the nominal MHz inputs. The analytic test verifies the constructed
  Hamiltonian's evolution; it does not calibrate slider inputs to electric
  field, laser power, or measured transition Rabi frequencies.
- **Metadata concerns:** `D3212` is labeled as `3D[3/2]1/2` in comments but
  constructed with `J=5/2`; nuclear g-factor signs differ between metadata
  and active constants; SD is entered in an E1-named reduced-element table
  despite not being an active SP/DP channel. The unused linewidth constants
  have inconsistent units/wavelength comments. These require review before
  reuse and were left untouched.
- **Observables:** UI correlations use an SP atomic transition operator and
  a static Hamiltonian at the final state, rather than pulse-resolved cavity
  output. Demo photon number is intracavity occupation, not emission
  probability. No measured-data fit, photon-counting calibration, or
  entanglement-fidelity validation is established by these tests.
- **Reference mismatch:** the existing guide cites Travers Ward's thesis,
  while the user's supplied dissertation was Thomas Walker's. This audit
  preserves the existing citations and flags the attribution for later
  reconciliation; no new numerical values were imported from either thesis.

## 6. Remaining work recorded in model-status.md

The existing [model status](model-status.md) requests:

1. Map rotating-frame detunings to Yb hyperfine transitions and relative
   offsets; review inclusion of the P Zeeman term.
2. Decide whether both cavity polarizations should interact simultaneously
   or whether the restricted protocol is intentional.
3. Separate useful output transmission from other losses; integrate useful
   flux and distinguish mean emitted count from exactly-one-photon probability.
4. Implement pulse-resolved cavity-output correlations before claiming
   single-photon performance.
5. Check cavity truncation and integration convergence, then study pulse
   speed, emission, and coherence across a parameter grid.

Its software notes also request aligning batch sweeps with the pulse-enabled
backend, reconciling/archiving historical tests, and recording the amplitude,
rate, and angular-weight conventions. These remain research work beyond
the requested behavior-preserving cleanup.

## Step 1 verification

Passed on this branch with Python 3.12.13 and the locked QuTiP 5.2.0
environment (versions reported by `uv sync` and pytest):

- `uv sync`: successful.
- `uv run pytest`: all 8 active tests passed.
- `uv run caipe-sim --demo`: successfully wrote `outputs/yb171_demo.png`
  and `outputs/yb171_demo.npz` (ignored generated files).

Temporary uv and Matplotlib cache directories and a headless Matplotlib
backend were set for the restricted local environment. No dependencies,
model parameters, or numerical behavior were changed. This is numerical
regression evidence, not experimental validation. Stop after this report
is presented for user review, before Step 2.
