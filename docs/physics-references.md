# Physics references and conventions

The user established the following reference hierarchy for project cleanup and
future development. The references provide physical background; they do not
define the software task or authorize additional changes.

1. **Primary physics reference:** C. J. Foot, *Atomic Physics*, Oxford
   University Press, 2005. Use this for atomic structure, selection rules,
   hyperfine and Zeeman physics, coherent driving, damping, and Raman processes.
2. **Cavity-generation model:** Travers Ward, *Generation of phase controlled
   single photons*, PhD thesis, University of Sussex, September 2022. Use this
   for the specific ion-cavity scheme, two polarization modes, collapse-rate
   conventions, output observables, and photon-generation analysis.
3. **Yb-specific numerical inputs:** the user's separately computed and
   verified values. Preserve these inputs; applying a reference's example
   numbers to Yb is not part of the cleanup.

When notation differs, translate the units and definitions explicitly before
comparing equations. Use Foot for the underlying physics, then express Ward's
specialized cavity model in a consistent convention. Identify any genuine
disagreement rather than silently switching formulas or numerical values.

## Relevant textbook sections

Page numbers below are the printed book pages. In the supplied PDF, printed
page 128 is PDF page 143; the main text has a 15-page offset.

| Topic in CAIPE-sim | Foot reference | Use in development |
| --- | --- | --- |
| Electric-dipole transitions and selection rules | §2.2, pp. 29–33; §5.4, p. 90 | Check allowed transitions and polarization selection. |
| Electronic Zeeman shifts | §5.5, pp. 90–93 | Check the electronic angular momentum and g-factor convention. |
| Hyperfine structure | §6.1, pp. 97–104 | Explain F-coupled states and relative hyperfine energies. |
| Zeeman shifts in the hyperfine basis | §6.3, pp. 108–112 | Specify weak-, intermediate-, or strong-field assumptions. |
| Electric-field driving and rotating-wave approximation | §§7.1–7.1.2, pp. 123–126 | Relate dipole matrix elements, electric-field amplitudes, Rabi frequencies, and Hamiltonian coefficients. |
| Rabi oscillations and pulse operations | §§7.3–7.3.1, pp. 127–128 | Validate resonant/detuned oscillations and pulse-area limits. |
| Spontaneous emission and transition rates | §1.7, pp. 11–13; §7.2, pp. 126–127 | Relate Einstein coefficients, lifetimes, and transition strengths. |
| Radiative damping and optical Bloch equations | §7.5.2, pp. 137–138 | Check population decay versus coherence decay. |
| Coherent Raman transitions | §9.8, pp. 208–210; Appendix E.1, pp. 310–312 | Check Raman detuning, far-detuned limits, and spontaneous-scattering assumptions. |
| Quantum jumps | §12.6, pp. 269–271 | Background for photon-counting and trajectory descriptions. |

Foot provides the physical basis for angular-momentum selection rules. A
specific Wigner 3j/6j recoupling identity should have its own derivation or
appropriate detailed reference; do not attribute an unlocated identity to
the textbook merely because it discusses hyperfine structure.

## Conventions to keep explicit

- Foot's resonant two-level result is `P_e(t) = sin^2(Omega * t / 2)`
  (Eq. 7.29). For a fixed-phase resonant envelope, the corresponding pulse-area
  result uses the integral of the transition-specific Omega. A field amplitude
  or intermediate helper value is not automatically that Omega.
- Foot defines detuning as radiation frequency minus atomic resonance in
  §7.3. A different Hamiltonian sign convention can be equivalent, but should
  be stated and used consistently across all terms.
- In Foot's optical Bloch equations, an isolated excited population decays at
  Gamma and its coherence decays at Gamma/2 (Eqs. 7.66–7.67).
- Ward uses collapse amplitudes `sqrt(2 * gamma)` and `sqrt(2 * kappa)`.
  For a unit-weight isolated atomic decay channel, Foot's population-decay
  Gamma therefore corresponds to `2 * gamma` in that convention. Additional
  angular weights and branching channels must be accounted for explicitly.
- Ward's output observables follow the cavity annihilation operators, whereas
  an atomic fluorescence operator describes a different emitted field.

The book's Raman discussion also distinguishes a coherent Raman transition
from sequential excitation and decay through a populated intermediate state.
A Lambda Hamiltonian alone does not establish that a chosen pulse sequence
operates in a Raman or adiabatic-transfer regime.

## Source access

The user supplied the local files `Foot C.J. Atomic physics (Oxford,
2005)(ISBN.pdf` and `Ward Travers.pdf`. This repository records bibliographic
and section references; the source PDFs remain in the user's Downloads folder.
Inspect the relevant PDF page directly if text extraction makes a mathematical
symbol or equation ambiguous.
