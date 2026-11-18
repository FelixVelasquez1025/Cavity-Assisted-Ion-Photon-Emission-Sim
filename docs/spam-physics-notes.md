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
