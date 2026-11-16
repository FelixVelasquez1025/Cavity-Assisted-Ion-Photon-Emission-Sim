# Hyperfine and coupling notes for 171Yb+

The active code uses an F,mF basis with I=1/2: S1/2 and P1/2 each have F=0,1;
D3/2 has F=1,2. See `HF_BASIS_ORDER` in
[src/caipe_sim/yb171_simulator.py](../src/caipe_sim/yb171_simulator.py).
These labels and angular couplings do not mean the model includes
hyperfine energy splittings in its diagonal Hamiltonian.

| Stored input | Value (GHz) | Source |
| --- | --- | --- |
| S value | 12.642812118466 | [IEEE Trans. UFFC 44, 344 (1997)](https://doi.org/10.1109/58.585119) |
| P value | 2.105 | [PRA 49, 3351 (1994)](https://doi.org/10.1103/PhysRevA.49.3351) |
| D value | 0.86375 | Phase 1b adopted value; [PRA 76, 052314 (2007)](https://doi.org/10.1103/PhysRevA.76.052314) reports a splitting 0.86(2), not this precision or an A constant |

Before adding diagonal hyperfine energies, resolve the D value's A-versus-
splitting interpretation and level ordering. The formula `E_F=A*K/2`, with
`K=F(F+1)-I(I+1)-J(J+1)`, gives a D splitting of `2*A` for these quantum
numbers. Foot, Ch. 6, supplies the angular-momentum background. Do not apply
a quadrupole formula with a denominator that vanishes for I=1/2; there is
no nuclear electric quadrupole term in this isotope.

SP/DP reduced E1 elements are derived from published lifetimes, branching
fractions and adopted wavelengths; they are no longer placeholders. The
signed 3j/6j recoupling convention is checked in
[tests/portfolio/test_angular_momentum.py](../tests/portfolio/test_angular_momentum.py).
Source details and qualifications are in [physics-references.md](physics-references.md).
Linear Zeeman shifts are included only for S and D; quadratic shifts,
P shifts, and field-induced mixing are absent. See [model-status.md](model-status.md).
