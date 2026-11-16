# CAIPE-sim — Yb-171 ion–photon emission

Open-system simulation of a driven Yb-171 ion with two optical cavity polarization spaces, built to study cavity-assisted ion–photon emission.

![Resonant Gaussian Rabi pulse: simulation, analytic solution and residual](docs/figures/rabi_validation.png)

Simulation against the analytic Gaussian pulse-area solution, using the same setup as the [Rabi test](tests/portfolio/test_yb171.py); residuals are shown below the population. [Illustrative demo dynamics](docs/figures/demo_dynamics.png).

## What it models

- A 16-state F,mF basis: S1/2 and P1/2 each have F=0,1; D3/2 has F=1,2. Hyperfine splittings are absent from the diagonal energies.
- A 369.52604 nm S–P Gaussian drive and a D3/2→P1/2 **reset coupling**, a simplified stand-in for the 935 nm repump via 3D[3/2]1/2. That intermediate-level pathway is not simulated.
- Two circular-polarization spaces on the **2.438 µm D3/2–P1/2 cavity line**; one polarization couples per run.
- P→S/P→D loss, cavity leakage and S/D dephasing, with explicit but unnormalized decay-input conventions.
- CPU QuTiP master-equation evolution, a Matplotlib UI, and headless population/photon-number exports.

## Parameters

The demonstration uses these inputs. Amplitudes and decay settings are model scales; see the [exact conventions](docs/model-status.md#exact-amplitude-convention-behavior-preserved).

| Parameter | Value | Units | Source |
| --- | --- | --- | --- |
| Drive / reset amplitude input | 5 / 3 | MHz | Illustrative; [demo] |
| Drive / reset pulse centre | 0.6 / 1.1 | µs | Illustrative; [demo] |
| Drive / reset Gaussian sigma | 0.2 / 0.25 | µs | Illustrative; [demo] |
| Drive / reset / cavity detuning; B | 0 / 0 / 0; 0 | MHz; G | Illustrative; [demo] |
| Cavity coupling input; κ/(2π) | 0.5; 0.1 | MHz | Illustrative; [demo] |
| γ_SP/(2π), γ_DP/(2π); S/D dephasing input | 0.01, 0.01; 0.000136 | MHz | Illustrative; [demo] |
| Initial state; coupled mode | S,F=0,mF=0; left | — | Illustrative; [demo] |
| Time window; samples | 0–2; 201 | µs; count | [demo] |
| Fock dimension per mode | 2 | count | [Truncation decision](docs/fock-convergence.md) |
| S–P vacuum wavelength | 369.52604 | nm | PRA 83, 013406, Table III; [sources] |
| D–P vacuum Ritz wavelength | 2438.417759 | nm | NIST ASD level difference; [sources] |
| P1/2 lifetime; P→D branching | 8.12; 0.00501 | ns; fraction | PRA 80, 022502; PRA 76, 052314; [sources] |

The UI starts in the same atomic state, left polarization, coupling and cavity loss, with its own field/pulse settings. Fixed atomic inputs, uncertainties and the NIST derivation are in the [source guide][sources].

## Validation

[Tests](tests/portfolio/) and reproducible scripts check:

- Selection rules and basis completeness.
- Hamiltonian Hermiticity.
- Population conservation and finite, nonnegative observables.
- Analytic resonant Rabi evolution.
- Signed hyperfine coefficients against evaluated Wigner 3j/6j reference values.
- Zeeman slopes against published formulas and rounded reference values.
- Fock truncation at dimensions 2, 3 and 4: **3 versus 4 exceeds 1e-3**, so the default stays 2; this does not establish convergence.

## Quickstart

In a checkout of this branch, with Python 3.12+ and uv installed:

```bash
uv sync --locked
uv run pytest
uv run caipe-sim --demo
uv run python scripts/make_readme_figures.py
uv run python scripts/check_fock_truncation.py
```

The demo writes PNG/NPZ files to `outputs/`; the figure script regenerates `docs/figures/`. Use `uv run caipe-sim` for the UI on a graphical desktop.

## Status and limitations

Inputs are illustrative and not fitted to experiment. The model uses the RWA, linear weak-field S/D Zeeman shifts, Markovian losses and a finite Fock basis. Missing diagonal hyperfine/P Zeeman terms, unnormalized decay scales and truncation sensitivity limit quantitative interpretation. See [model status](docs/model-status.md) and the [review report](docs/portfolio-review.md).

### Next steps

- Complete the detuning/energy map, physical decay calibration and truncation study.
- Evolve both cavity polarizations simultaneously for the intended protocol.
- Add pulse-resolved cavity-output correlations.
- Separate useful output flux from other losses.

Current plots do not establish emission probability, indistinguishability or polarization entanglement.

## Origin

Grew out of a collaborative three-level model, preserved with the historical calcium examples in [archive/](archive/README.md). Licensed under [MIT](LICENSE).

## References

C. J. Foot, *Atomic Physics* (Oxford, 2005). Thomas Walker, *Single Photons from a Trapped-Ion Cavity-QED System* (Sussex PhD, June 2020).

Atomic data: [wavelengths, PRA 83, 013406](https://doi.org/10.1103/PhysRevA.83.013406); [P lifetime, PRA 80, 022502](https://doi.org/10.1103/PhysRevA.80.022502); [D lifetime, PRA 61, 022507](https://doi.org/10.1103/PhysRevA.61.022507); [auxiliary lifetime, J. Phys. B 26, L701](https://doi.org/10.1088/0953-4075/26/20/003); [branching and HFS, PRA 76, 052314](https://doi.org/10.1103/PhysRevA.76.052314); [S HFS, IEEE UFFC 44, 344](https://doi.org/10.1109/58.585119); [P HFS, PRA 49, 3351](https://doi.org/10.1103/PhysRevA.49.3351); [magnetic conventions, arXiv:2509.04416](https://arxiv.org/abs/2509.04416); [NIST ASD](https://doi.org/10.18434/T4W30F). Scope and source-precision qualifications are in [sources].

[demo]: src/caipe_sim/demo.py
[sources]: docs/physics-references.md
