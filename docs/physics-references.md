# Physics references and conventions

Foot supplies the atomic-physics background; **Thomas Walker, _Single
Photons from a Trapped-Ion Cavity-QED System_, PhD thesis, University of
Sussex, June 2020**, supplies the calcium ion-cavity model. The earlier
attribution to a different thesis was incorrect. The user's supplied Walker
PDF is a reference, not a repository asset; no source PDF or level diagram
is distributed here. The [validation notes](validation-notes.md) record current numerical checks
and earlier atomic-data correction comparisons.

## Atomic inputs

These are adopted inputs, not fitted experimental parameters. The code
retains the precision requested for this project. A citation to spectroscopic
context does not establish that every retained decimal was independently
verified in that source. In particular, see the wavelength and D hyperfine
qualifications below. Derived quantities inherit their input uncertainties;
this simulator does not propagate those uncertainties.

| Input | Adopted value | Published source / interpretation |
| --- | --- | --- |
| S1/2–P1/2 wavelength | 369.52604 nm (vacuum) | McLoughlin et al., [PRA 83, 013406 (2011)](https://doi.org/10.1103/PhysRevA.83.013406), Table III: S1/2(F=1) ↔ P1/2(F=0) of 171Yb+, 369.52604(6) nm. The wavelength is a vacuum measurement; it is used for the reduced-element frequency, not as a completed hyperfine-resolved detuning map. |
| D3/2–P1/2 vacuum Ritz wavelength | 2438.41775948 nm (about 2.438 µm) | Derived from NIST ASD Yb II fine-structure levels: 27061.82 − 22960.80 = 4101.02 cm⁻¹; see the NIST derivation below. |
| S1/2–D3/2 vacuum Ritz wavelength | 435.524894603 nm | Derived from NIST ASD D level 22960.80 cm⁻¹ above S; unused E2 metadata, not an E1 element. |
| P1/2 lifetime | 8.12(2) ns; code uses central value | Olmschenk et al., [PRA 80, 022502 (2009)](https://doi.org/10.1103/PhysRevA.80.022502), _Measurement of the lifetime of the 6p 2P1/2 level of Yb+_. |
| D3/2 lifetime | 52.7 ms; source uncertainty 2.4 ms | Yu and Maleki, [PRA 61, 022507 (2000)](https://doi.org/10.1103/PhysRevA.61.022507). Unused transition metadata; no D→S radiative collapse is implemented. |
| 3D[3/2]1/2 lifetime | 37.7 ns | Berends et al., [J. Phys. B 26, L701 (1993)](https://doi.org/10.1088/0953-4075/26/20/003), _Beam-laser lifetime measurements for four resonance levels of Yb II_. Retained metadata; this level is absent from the evolved basis. |
| P1/2→D3/2 branching fraction | 0.00501(15); code uses central value | Olmschenk et al., [PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314), Fig. 1 and Appendix A. |
| P1/2→S1/2 branching fraction | `1 - 0.00501` = 0.99499 | Complement of the preceding measurement in the retained two-branch approximation, not a separate measurement. |
| S1/2 hyperfine value | 12.642812118466 GHz | [IEEE Trans. UFFC 44, 344 (1997)](https://doi.org/10.1109/58.585119), _Accurate measurement of the 12.6 GHz clock transition in trapped 171Yb+ ions_. Metadata uses the same S value. |
| P1/2 hyperfine value | 2.105 GHz | [PRA 49, 3351 (1994)](https://doi.org/10.1103/PhysRevA.49.3351), _Isotope shifts and hyperfine structure in the 369.4-nm resonance line of singly ionized ytterbium_. Metadata uses the same P value. |
| D3/2 splitting and A/h | splitting 0.86375 GHz; A/h = 0.431875 GHz | Adopted splitting, [PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314), Fig. 5, which reports 0.86(2) GHz. A/h = splitting/2 for I=1/2,J=3/2; adopted digits exceed the source precision. |
| 3D[3/2]1/2 hyperfine splitting | 2.2095 GHz | [PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314), Fig. 5: 2.2095(11) GHz. Metadata only. |
| Nuclear spin | I=1/2 | [PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314), Sec. I. |
| Manifolds | S1/2, P1/2, D3/2; auxiliary 3D[3/2]1/2 | [PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314), Figs. 1 and 5. F,mF basis from angular-momentum addition with I=1/2 (Foot, Ch. 6). |
| Electron spin g factor | 2.00231930436092 | Kroeze, Kristensen and Pucher, [arXiv:2509.04416](https://arxiv.org/abs/2509.04416), Table 2 and Sec. 4.2. |
| Orbital g factor | 0.99999679 | Same reference, Eq. 17; retained finite-mass approximation. |
| Nuclear g factor | -0.0005362 | Same reference, Eqs. 16–19, in Bohr-magneton units with a **positive** `g_I I·B` term in the magnetic Hamiltonian. Used consistently in active code and level metadata. |
| g_J, g_F | Calculated from L,S,J,I | Foot, §§5.5 and 6.3; [arXiv:2509.04416](https://arxiv.org/abs/2509.04416), Eqs. 21–23. The active code uses corrected spin/orbital factors; legacy metadata uses the elementary LS Landé formula. These are calculated, not measured ion g_J values. |
| Quadrupole hyperfine metadata B | Zero | I=1/2 has no nuclear electric quadrupole contribution (Foot, Ch. 6); the metadata is unused in evolution. |
| SI constants | `scipy.constants` | SciPy's supplied physical constants; no manually substituted numerical values. |

**Hyperfine labeling.** `nu_D32` stores the F=1↔F=2 splitting;
`HFS_A_GHZ['D']` and `LevelData.Ahfs/h` now store half that value, the
magnetic-dipole A/h. S and P have J=1/2, so their A/h equals their splitting.
Their metadata now uses the same S/P values as `HFS_A_GHZ`, rather than
rounded duplicates. These constants still do not enter active diagonal
energies. The auxiliary D3212 object is metadata only; its legacy S,L fields
are not a valid LS configuration for building additional couplings.

## NIST wavelength derivation

Kramida, Ralchenko, Reader and NIST ASD Team (2024), _NIST Atomic Spectra
Database_, version 5.12, [DOI 10.18434/T4W30F](https://doi.org/10.18434/T4W30F),
accessed **2026-11-16**. [Official Yb II level query](https://physics.nist.gov/cgi-bin/ASD/energy1.pl?spectrum=Yb%20II&units=0&format=3&output=0&page_size=500&conf_out=on&term_out=on&level_out=on&j_out=on&biblio=on).
The query returns S1/2 = 0.00, D3/2 = 22960.80, and P1/2 = 27061.82 cm⁻¹.
For energies expressed as wavenumbers, `lambda_vac(nm)=1e7/(E_upper-E_lower)`.
The code retains these input level energies and derives both wavelengths;
extra displayed digits show arithmetic, not extra experimental precision.
These are fine-structure Ritz values, not isotope/hyperfine-resolved
171Yb+ lines. No hyperfine detuning or energy offsets were inferred from them.
The separate 369.52604 nm input is the vacuum hyperfine-component measurement
identified in the table above; its distinct provenance is intentional.

**Neutral versus ion data.** arXiv:2509.04416 concerns **neutral Yb**.
Its magnetic-moment convention and generic angular-momentum formulas are
used here; its neutral-atom level tables are not a source for the ion's
S/P/D hyperfine constants. The orbital factor's suitability at higher
precision in the ion remains a separate approximation.

## Derived inputs and decay conventions

The SP/DP entries in `REDUCED_E1_ELEMENTS` are calculated, in units of ea0,
from the preceding branching fractions, optical angular frequencies, and
P lifetime using `_rme_eq54`. This is the Racah-normalized lifetime relation
in [arXiv:2509.04416, Eq. 54](https://arxiv.org/html/2509.04416v1#S5.SS3).
It is a general E1 relation, not a neutral-atom numerical dipole input.
The unused SD E1 estimate and its unsourced branching factor were removed;
the NIST-derived S–D E2 line has no E1 reduced element in this model.
Einstein-A metadata uses `branch/tau` (SP/DP) and `1/tau_D` (SD). These are
not automatically the Lindblad rates selected by the UI or demo.

The SP/DP lowering amplitudes are `sqrt(2*gamma_pair)*abs(c_ul)` and cavity
losses are `sqrt(2*kappa)*a` and `sqrt(2*kappa)*b`. The weighted coefficients
include reduced dipoles, not normalized angular branching amplitudes alone.
Consequently, **the decay-rate inputs are not normalized physical rates**.
A total rate requires summing `C†C` over the available final states. Equal
SP and DP input values do not implement the measured branching ratio simply
because that ratio also enters the reduced dipoles. The project does not
silently reinterpret or normalize these inputs.

Walker uses the same factor-of-two collapse convention: Eqs. 2.2.6–2.2.7
and 2.4.16–2.4.19. His cavity output equations are 2.6.3–2.6.5. Integrated
output intensity is a mean photon count; it is a single-photon probability
only under an at-most-one-photon protocol. Output-loss separation and
pulse-resolved cavity correlations remain unimplemented.

## Angular phases and amplitude normalization

The reference phase convention for absorption is

```text
<Fp,mp|d_q|F,m> / <Jp||d||J>
 = (-1)^(Fp-mp) * (Fp 1 F; -mp q m)
   * (-1)^(Jp+I+F+1) * sqrt((2Fp+1)(2F+1)) * {Jp Fp I; F J 1}.
```

Round and test the integer exponent itself, not twice the exponent. The
fixed signed cases in [test_angular_momentum.py](../tests/test_angular_momentum.py)
include evaluated 3j and 6j factors and odd exponents. Daniel A. Steck,
[_Rubidium 87 D Line Data_](https://steck.us/alkalidata/rubidium87numbers.pdf),
Sec. 3.3, Eqs. 35–39, supplies the convention check after translation from
his reduced-element normalization and matrix-element ordering. No Rb atomic
values are used. Foot explains the underlying selection rules, but is not
cited as containing an unlocated detailed recoupling identity.

Foot's `P_e(t)=sin^2(Omega*t/2)` is Eq. 7.29. The project retains its existing
extra amplitude half-factor; see the [exact mapping](model-status.md#exact-amplitude-convention-behavior-preserved).
Foot's optical Bloch equations 7.66–7.67 give population decay Gamma and
coherence decay Gamma/2. For a unit-weight Walker collapse channel,
Gamma=2*gamma. All angular weights and branches must still be included.

## Textbook sections

C. J. Foot, _Atomic Physics_, Oxford University Press (2005), is the primary
physics textbook. Printed book pages:

| Topic | Sections / pages |
| --- | --- |
| E1 transitions and selection rules | §2.2, pp. 29–33; §5.4, p. 90 |
| Electronic Zeeman shifts | §5.5, pp. 90–93 |
| Hyperfine structure | §6.1, pp. 97–104 |
| Hyperfine Zeeman shifts and field regimes | §6.3, pp. 108–112 |
| Driving and rotating-wave approximation | §§7.1–7.1.2, pp. 123–126 |
| Rabi oscillations and pulses | §§7.3–7.3.1, pp. 127–128 |
| Spontaneous emission | §1.7, pp. 11–13; §7.2, pp. 126–127 |
| Optical Bloch equations | §7.5.2, pp. 137–138 |
| Raman transitions | §9.8, pp. 208–210; Appendix E.1, pp. 310–312 |
| Quantum jumps | §12.6, pp. 269–271 |

A Lambda Hamiltonian and Gaussian pulses alone do not establish adiabatic
transfer or a far-detuned Raman regime. Detuning signs and reference
transitions must be stated explicitly before comparing equations or sources.
