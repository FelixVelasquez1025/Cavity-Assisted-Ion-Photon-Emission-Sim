# Free-space ¹⁷¹Yb⁺: physics and validation notes

This model is separate from the cavity application. `spam.atomic` defines
20 internal states with no photon Hilbert space. S and P each have F=0,1;
D has F=1,2; A denotes the auxiliary 3D[3/2]₁/₂ level with F=0,1.
Each F contains m=−F,…,+F. I=1/2 follows Olmschenk et al. (2007), Sec. I.

## Atomic inputs and provenance

| Input | Adopted value | Source and scope |
| --- | --- | --- |
| S splitting, E(F=1)−E(F=0) | +12.642812118466 GHz | Fisk et al., [IEEE UFFC 44, 344 (1997)](https://doi.org/10.1109/58.585119) |
| P splitting, E(F=1)−E(F=0) | +2.105 GHz | [PRA 49, 3351 (1994)](https://doi.org/10.1103/PhysRevA.49.3351) |
| D splitting, E(F=2)−E(F=1) | +0.86375 GHz | Adopted input; [Olmschenk et al., PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314), Fig. 5 and Appendix B report 0.86(2) GHz. **UNVERIFIED: precision beyond that measurement.** |
| A splitting, E(F=1)−E(F=0) | −2.2095 GHz | Same paper, Fig. 5 and Appendix B, magnitude 2.2095(11) GHz; F=0 is higher. |
| P lifetime | 8.12 ns | [PRA 80, 022502 (2009)](https://doi.org/10.1103/PhysRevA.80.022502), 8.12(2) ns |
| D lifetime | 52.7 ms | [PRA 61, 022507 (2000)](https://doi.org/10.1103/PhysRevA.61.022507), 52.7(2.4) ms |
| A lifetime | 37.7 ns | Berends et al., [J. Phys. B 26, L701 (1993)](https://doi.org/10.1088/0953-4075/26/20/003) |
| P→D branching | 0.00501 | Olmschenk 2007, Appendix A, 0.00501(15) |
| P→S branching | 0.99499 | Complement in the retained two-branch model |
| A→S branching | 0.982 | Mazzanti et al., [PRA 110, 043105 (2024)](https://doi.org/10.1103/PhysRevA.110.043105), Sec. II explicitly states 98.2%; Appendix A has inconsistent branch subscripts, so the main-text assignment is used. |
| A→D branching | 0.018 | Complement; retained two-branch approximation |
| D→S branching | 1 | E2 decay treated as the complete measured D lifetime in this restricted basis; not a separate branching measurement |
| A electronic g_J | 1.320 | [NIST ASD](https://doi.org/10.18434/T4W30F), v5.12, Yb II level 33653.86 cm⁻¹, accessed **2026-11-18**. Tabulated value, not computed from LS coupling. |
| S/P/D g_J | LS projection with corrected orbital/spin factors | Foot, *Atomic Physics* (2005), §§5.5, 6.3; shared validated cavity helper. An approximation, not measured ion g_J. |
| Electron spin g | 2.00231930436092 | [arXiv:2509.04416](https://arxiv.org/abs/2509.04416), Table 2 |
| Orbital g | 0.99999679 | Same reference, Eq. 17, retained mass approximation |
| Nuclear g in μ_B units | −0.0005362 | Same reference, Eqs. 16–19, positive g_I I·B Hamiltonian convention. Neutral-Yb generic convention, not ion spectroscopy. |
| Optional clock quadratic coefficient | 310.8 Hz/G² | Fisk 1997; also Olmschenk 2007 Sec. IV and Egan 2021 §2.2.5 |
| Physical constants | SciPy SI constants | μ_B and ℏ from `scipy.constants` |

[NIST level query](https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=Yb%20II&units=0&format=3&output=0&page_size=500&conf_out=on&term_out=on&level_out=on&j_out=on&lande_out=on&biblio=on).
No source PDFs or level-diagram images are distributed with the code.

## Why hyperfine energies belong in the Hamiltonian

An electronic label such as S₁/₂ does not specify the full atomic state.
Coupling the electron's J to nuclear spin I produces total F and magnetic
projection m. Light tuned to one F→F′ line is generally detuned from the
others. Keeping those detunings is essential: weak excitation of the wrong
P hyperfine level can change the stored qubit state during detection.

For a magnetic-dipole hyperfine interaction, use

\[
E_F/h=\frac{A/h}{2}\{F(F+1)-I(I+1)-J(J+1)\}.
\]

The input is the **signed F-level splitting**, not A. Dividing by the
appropriate difference in the expression in braces gives A. Thus A/h is
+12.642812118466 GHz for S, +2.105 GHz for P, +0.431875 GHz for D, and
−2.2095 GHz for A. Every manifold's degeneracy-weighted centroid is zero.
Absolute optical electronic energies are not needed when detunings are
specified relative to a measured hyperfine transition. Removing a common
centroid does not remove the observable F-level differences.

## Magnetic field and the optional clock correction

A weak field defines the z quantization axis and separates magnetic
sublevels. Dark-state destabilization follows the mechanism discussed by
Berkeland and Boshier, [PRA 65, 033413 (2002)](https://doi.org/10.1103/PhysRevA.65.033413). It also prevents population remaining in coherent combinations
that the detection light cannot excite. The linear shift is

\[
H_Z/\hbar=g_F(\mu_B/\hbar)(10^{-4}B_{\rm G})m,
\quad
g_F=g_J\frac{F(F+1)+J(J+1)-I(I+1)}{2F(F+1)}
+g_I\frac{F(F+1)+I(I+1)-J(J+1)}{2F(F+1)}.
\]

The factor 10⁻⁴ converts gauss to tesla. For F=0 the first-order shift is
zero; the code handles that limit explicitly. S/P/D use the shared LS
approximation for g_J. A instead uses the tabulated NIST value, because its
bracket-coupled configuration is not the LS term suggested by its legacy
metadata. The predicted A F=1 slope is about 0.9234 MHz/G per m.

`clock_quadratic=True` adds opposite half-shifts to S F=0,m=0 and S F=1,m=0,
so their frequency difference increases by 310.8 B_G² Hz. It is off by
default. It represents only the clock transition coefficient, not a full
Breit–Rabi treatment: shifts of other sublevels, field-dependent mixing and
changes to dipole coefficients are omitted. The entire model assumes
Zeeman energies small compared with hyperfine splittings.

## Light intensity, polarization and sidebands

Each `Beam` accepts an electronic **reduced Rabi frequency in rad/s**,
not intensity or saturation parameter. The user supplies electric-field
strength times the relevant reduced electronic matrix element divided by
ℏ. The matrix element for each sublevel is

\[
H_{u\ell}/\hbar=\frac{\Omega_{\rm red}}{2}
\epsilon_q(-1)^{F'-m'}
\begin{pmatrix}F'&1&F\\-m'&q&m\end{pmatrix}
(-1)^{J'+I+F+1}\sqrt{(2F'+1)(2F+1)}
\left\{\begin{matrix}J'&F'&I\\F&J&1\end{matrix}\right\}.
\]

Here q=m′−m. The 6j hyperfine factor and the 3j evaluation reuse the
validated cavity functions; no second E1 recoupling implementation is
introduced. The auxiliary state's electronic character affects its reduced
matrix element and g_J, not this angular-momentum projection law. There
is no conversion from watts or s=I/I_sat and no inferred 935 nm dipole
strength. This choice makes both beams usable without inventing those
calibrations. The Hamiltonian coefficient includes exactly one factor 1/2;
it does not inherit the older cavity slider's additional factors.

Assume a plane wave with propagation along y, magnetic field along z,
and a real electric field in the xz plane. For linear polarization making
angle θ with z, the spherical amplitudes are

\[
\epsilon_0=\cos\theta,\qquad
\epsilon_{+1}=-\sin\theta/\sqrt2,\qquad
\epsilon_{-1}=+\sin\theta/\sqrt2.
\]

Their squared magnitudes sum to one. π drives Δm=0, and σ± drive Δm=±1.
The relative signs matter for coherent dark states; using probabilities
in place of amplitudes would erase interference. At B=0, θ is measured
from the axis approached as B→0; rotating that axis cannot remove the dark
subspace. These beams are perfectly linear; ellipticity, spatial gradients,
motion and polarization noise are not modeled.

Carrier detuning is laser frequency **minus** the zero-field reference
line: S F=1→P F=0 for 369 nm and D F=1→A F=0 for 935 nm. Blue detuning
is positive. A `Tone` specifies an offset from that carrier, an independent
Rabi frequency and a phase. The carrier and sidebands share polarization;
no modulation-depth or laser-power partition is assumed.

| Optional offset | Source | Interpretation |
| --- | --- | --- |
| +14.748 GHz on 369 | [Egan (2021)](https://doi.org/10.13016/epvo-4n5k), §2.2.2 | Returns S F=0 population through P F=1 during Doppler operation. With the adopted splittings, exact resonance is +14.747812118466 GHz, so this rounded setting is 0.187881534 MHz blue. |
| +2.105 GHz on 369 | Egan §2.2.3; Olmschenk 2007 Sec. IV | Addresses S F=1→P F=1 for pumping toward S F=0. |
| −3.070 GHz on 935 | Olmschenk Sec. IV, Fig. 5 | Addresses D F=2→A F=1. Exact offset from the adopted splittings is −3.07325 GHz: the rounded setting is 3.25 MHz blue. Egan §2.2.2 instead quotes a 3.072 GHz modulation magnitude; these rounded settings are not silently equated. |

Egan's *Scaling Quantum Computers with Long Chains of Trapped Ions*
(University of Maryland, 2021), Ch. 2 is the operating-protocol reference.
No beam intensities from the thesis are installed as calibrated defaults.

## Why the rotating-frame model is static

Optical frequencies are too fast to follow directly. The optical rotating
wave approximation drops terms oscillating near twice an optical frequency.
It retains detunings, hyperfine structure and all near- and far-detuned
sublevel couplings of each single laser tone. It does not eliminate the
excited states or reduce them to scattering-rate equations.

For a single-tone laser, rotate all its upper states at that laser frequency.
The resulting Hamiltonian is exactly time independent within optical RWA.
In particular, a detection carrier still couples to P F=1, about 2.105 GHz
away, and S F=0→P F=1, about 14.748 GHz away. These matrix elements must
remain: their small excitation probabilities produce leakage. A test checks
that these off-resonant matrix elements are nonzero and that their detuning
energies remain present.

With multiple tones, choose the nearest nonzero tone for **each allowed
hyperfine F→F′ edge**, before resolving m. Zero-amplitude tones do not
change the frame. These manifolds each have three allowed edges connecting
four F blocks, a tree. Therefore block rotation frequencies r_F can satisfy
r_upper−r_lower=the selected tone frequency on every edge simultaneously.
Subtract 2πr_F from each block's diagonal energy; assigned couplings become
static. Relative phases of all m transitions and assigned tones are kept.
This F-dependent extension of the per-laser frame adds no extra drive.

Every **additional tone on that same edge** oscillates at the difference
between its frequency and the assigned frequency. The secular model drops
these extra beat terms. It keeps every hyperfine edge; for a one-tone beam
it drops none. `full_time_dependent_hamiltonian()` restores every discarded
term as V exp(−i Δω t)+V† exp(+i Δω t) in exactly the same frame, allowing
a direct comparison without a gauge transformation of the states.

This approximation needs discarded beats large compared with their Rabi
couplings, linewidths and relevant slow dynamics. It omits fast micromotion
and AC Stark shifts caused by discarded tones. Near-degenerate tones or
very strong sidebands require the full time-dependent model. The nearest-tone
assignment can also drop a far-detuned leakage contribution of a second
tone: it is not a guarantee of accurate small leakage rates. Stage 1a's
short-window comparison checks one weak two-tone case only. Long detection
windows and each intended leakage configuration need further convergence
checks before Stage 1b conclusions can be trusted.

## Physical spontaneous decay, including the metastable state

A lifetime τ means total population loss Γ=1/τ, in s⁻¹. For each upper
sublevel u and each lower manifold L with branching fraction b_L, evaluate
all allowed angular coefficients c_lu and set

\[
w_{\ell u}=\frac{|c_{\ell u}|^2}{\sum_{\ell'\in L}|c_{\ell'u}|^2},\qquad
C_{\ell u}=\sqrt{\frac{b_L}{\tau}w_{\ell u}}|\ell\rangle\langle u|.
\]

Consequently Σ C†C on every u is exactly 1/τ and the summed probability
of each branch is b_L. No factor 2 or conversion from MHz appears in these
physical rates. This is independent of the older cavity model's unnormalized
rate inputs, which are unchanged. The same angular coefficients determine
laser strengths and E1 decay weights; normalizing each branch avoids
accidentally imposing an electronic reduced-element normalization twice.

D→S is an E2 transition and uses **rank 2**, including Δm=0,±1,±2.
The rank-2 Wigner–Eckart formula replaces the two rank-1 entries in the 3j
and 6j symbols by 2 and changes the recoupling phase accordingly. It reuses
the existing Wigner evaluators, with no guessed E2 reduced element. The
measured D lifetime supplies its total rate, only about 19 s⁻¹ compared
with P's roughly 1.23×10⁸ s⁻¹. It is included even where negligible.

Separate jumps for each resolved pair implement a Markovian, secular vacuum
bath. There is no cross-damping coherence between decay channels, collision
loss, radiation trapping or additional electronic manifold. At exactly
B=0 some magnetic channels are degenerate: neglecting cross-channel
spontaneously transferred coherence is an additional bath approximation,
so the dark-state result is explicitly conditional on this model.

Optional `linewidth_dephasing_s=gamma_phi` appends
C_phi=√(2 gamma_phi) Π_upper per beam. This damps that beam's optical
coherence at gamma_phi without removing population. It is off by default,
and describes independent white phase noise, not arbitrary laser spectra
or correlated sideband phase diffusion. It is a user input, not a claimed
measured linewidth.

## Propagation, observables and numerical meaning

The Lindblad equation is

\[
\dot\rho=-i[H/\hbar,\rho]+\sum_j(C_j\rho C_j^\dagger
-\tfrac12\{C_j^\dagger C_j,\rho\}).
\]

The explicit collapse construction generates a completely positive,
trace-preserving semigroup in exact arithmetic. Long-run validation uses
dense exp(L Δt) on the 400-component density vector rather than time stepping
through GHz coherences; no populations are clipped and no trace or
positivity repair is applied. Matrix-exponential roundoff is measured by
the tests. The short full-tone reference instead uses QuTiP's adaptive
integrator with explicit small maximum steps to resolve sideband beats.
A stationary solve is valid only if the selected parameters give a unique
attractive stationary state; exact dark subspaces can make it nonunique.

`fluorescence_369_s` sums Γ_P b_PS ρ_uu over P states. It counts total
emitted 369 nm photons per second, not collected photons, total P decays,
or detector counts. This internal-state model can describe the fluorescence
used for Doppler cooling, but has no velocity or temperature and cannot
predict the cooling limit itself. Stage 1b's collection, counting statistics
and experimental benchmarks are not implemented here.

## What the weak-probe spectrum measures

A fully pumped steady state is the wrong way to locate all optical lines:
a weak resonant beam can eventually pump its own starting population out of
the addressed F manifold or into a coherent dark state. Instead the scan
holds the initial ground populations fixed and evaluates the leading-order
optical susceptibility. For an incoherent uniform mixture in a selected S F,

\[
R_{369}(\Delta)=\sum_{\ell,u}p_\ell
\frac{\Gamma_{PS}|(H/\hbar)_{u\ell}|^2}
{(\Gamma_P/2)^2+(\omega_{u\ell}-\omega_L)^2}.
\]

This comes from solving the linear optical-coherence equation at fixed
ground population. It is valid after coherence settles but before appreciable
optical pumping, for Rabi frequencies much smaller than the natural width.
The code takes its matrix elements and detunings from the actual single-tone
Hamiltonian. The scan combines a coarse full range with 0.1 MHz sampling
near the lines. It is also corroborated with full 100 ns master-equation
weak pulses at the three resonances and at ±80 MHz: all three on-resonance
populations exceed either wing by more than 60×, while P population stays
below 1.1×10⁻⁶. Thus the peak locations are not solely a check of a separately
written Lorentzian formula.

From S F=1 the two features lie at 0 and +2.105 GHz relative to detection.
From S F=0 the allowed line is at +12.642812118466+2.105 GHz, or
+14.747812118466 GHz, which rounds to +14.748 GHz. S F=0→P F=0 is forbidden.
These are optical frequency differences, not missing laser sidebands.

## Reproducing the Stage 1a checks

From the repository root:

```bash
uv run pytest
uv run caipe-sim --demo
uv run python scripts/check_spam_atomic.py --output outputs/spam-stage1a.json
```

The script creates only an optional machine-readable diagnostic report;
it does not change defaults or run the Stage 1b benchmark sweep.
`outputs/` remains ignored. Numerical results below were measured with
Python 3.12.13, NumPy 2.3.1, SciPy 1.16.0 and QuTiP 5.2.0.

| Check | Result | Scope |
| --- | --- | --- |
| Maximum relative total-decay normalization error | 2.22045×10⁻¹⁶ | Every P, A and D sublevel; individual branch sums also tested at relative 10⁻¹⁰ |
| S F=1 Zeeman slope | 1.400872330 MHz/G per m | Linear weak-field model |
| P F=1 Zeeman slope | 0.465622237 MHz/G per m | Includes actual P Hamiltonian term |
| D F=1 Zeeman slope | 1.398993833 MHz/G per m | Includes actual D Hamiltonian term |
| D F=2 Zeeman slope | 0.839096109 MHz/G per m | Includes actual D Hamiltonian term |
| A F=1 Zeeman slope | 0.923376925 MHz/G per m | NIST g_J=1.320, projected with I=1/2; includes actual A term |
| Probe P-line separation | 2.105 GHz | Initial weak-probe spectrum; not a pumped steady state |
| Probe from S F=0 | +14.747812118466 GHz | Relative to the detection carrier |
| Max sampled two-tone population difference | 6.05823×10⁻⁶ | 121 times across 0–30 ns, 369 carrier and +14.748 GHz tone |
| Max sampled density-matrix Frobenius difference | 1.80873×10⁻⁴ | Includes coherences/micromotion omitted by secularization |
| Max trace error through 1 ms | 2.78132×10⁻¹⁰ | Test times 0, 0.1 µs, 10 µs, 1 ms |
| Max Hermiticity error through 1 ms | 9.56362×10⁻¹⁴ | Largest entrywise mismatch with ρ† |
| Minimum density eigenvalue across those times | 0 | Includes the initial pure state; no negativity repair |
| Minimum Choi eigenvalue of the 20 ns channel | +4.17261×10⁻¹³ | An additional finite-time complete-positivity check |

The two-tone test starts in (|S,F=0,m=0⟩+|S,F=1,m=0⟩)/√2, with B=5 G,
θ=0.7 rad and both reduced Ω/(2π)=5 MHz. These Rabi inputs and angle are
illustrative, not a fit. The full integrator uses atol=10⁻¹¹, rtol=10⁻⁹,
maximum step 1/(50×14.748 GHz)=1.35612 ps. Repeating at half that step gives
6.05816×10⁻⁶ and 1.80873×10⁻⁴ respectively. The population-difference
metric changes by 7.10×10⁻¹¹; the density-norm metric by 1.58×10⁻¹⁰.
These are sampled maxima, not bounds between output times. Thirty ns is
about 3.7 P lifetimes, not a detection-window convergence study.

### Zero field versus 5 G

For this check both beams have θ=55°, carrier Ω_red/(2π)=20 MHz, sideband
Ω_red/(2π)=10 MHz, and zero carrier detuning. The 369 sideband is +14.748 GHz
and the 935 sideband is −3.070 GHz. These strengths are **illustrative
validation controls**. They are not Egan's measured intensities. Clock
quadratic correction and laser dephasing are off. All physical decay channels
remain active.

| Magnetic field | Stationary emitted 369 nm photons/s |
| --- | ---: |
| 0 G | 851.431664 |
| 5 G | 5,103,099.706775 |

The ratio is 1.66846×10⁻⁴, about a 5994-fold suppression. The stationary
Liouvillian residual norm is below 9×10⁻⁷ s⁻¹ in both cases, and both
density matrices have nonnegative eigenvalues within numerical precision.
The dominant S F=1→P F=0 coupling has a two-dimensional dark subspace at
zero field: one row of couplings cannot couple three independent ground
states. A test constructs a null vector. Off-resonant coupling to P F=1
and the remaining repump dynamics weakly lift exact darkness, explaining
the small nonzero fluorescence floor. The field splits m levels so the
dark superposition evolves into combinations the laser can excite.
This is a consistency check, not a magnetic-field/polarization optimization
or a fit to Egan's apparatus.

## Approximation inventory and unresolved questions

- **Restricted internal basis:** only these 20 states; no F₇/₂ shelving,
  P₃/₂, collisions, motional Doppler shifts, micromotion or recoil. The
  model describes cooling-light fluorescence, not actual Doppler cooling.
- **Weak field:** linear g_F shifts with fixed F basis, with an optional
  clock-only B² correction. No full Breit–Rabi mixing or complete quadratic
  Hamiltonian. S/P/D electronic LS factors are approximate; A is tabulated.
- **Optical RWA and secular multitone approximation:** exact single-tone
  static Hamiltonian including far-detuned transitions; additional beat
  terms omitted only for multitone runs. Their micromotion and light shifts
  are not captured. The 30 ns check does not certify small leakage rates
  over 100 µs detection windows; that must be revisited for Stage 1b.
- **Coherent ideal linear polarization:** fixed angles and relative phases,
  with no default phase noise, amplitude noise or ellipticity. User Rabi
  frequencies are not intensity-calibrated inputs. The optional pure
  dephasing model is independent white phase noise per beam, not a complete
  model of correlated carrier/sideband linewidth.
- **Markovian resolved-channel vacuum bath:** angularly weighted, normalized
  independent jumps. Cross-channel decay coherences at exact degeneracy
  are ignored. D decay is E2 and is assumed to terminate entirely in S;
  P and A each use only the two retained electronic branches.
- **Adopted spectroscopy:** rounded sideband offsets need not be exactly
  resonant. D splitting **UNVERIFIED precision** remains the only explicitly
  unverified numeric input. Uncertainties of cited measurements and the
  tabulated auxiliary g_J are not propagated.
- **Validation scope:** the susceptibility assumes fixed initial ground
  populations; full weak pulses corroborate its resonances. Numerical
  positivity checks supplement, rather than replace, the Lindblad argument.
  No SPAM fidelity, optimum angle, scattering-budget or detector histogram
  is claimed. Those are Stage 1b work, which has not started.
