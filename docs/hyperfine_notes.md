# Hyperfine and Coupling Notes (Yb‑171, fermionic I=1/2)

## Constants (energies at B=0 unless stated)
- 2S1/2: A = 12.642812118466 GHz, plus quadratic Zeeman correction +310.6 Hz · B² (B in gauss). No B term; S1/2 has no electric quadrupole (B constant).
- 2P1/2: A = 2.105 GHz. (B = 0 for P1/2.)
- 2D3/2: A = 0.86372 GHz. (B typically small/neglected for D3/2 here.)

## Basis (implemented in code)
- Full 16-state hyperfine ordering:
  - S1/2: F=0 (mF=0), F=1 (mF=-1,0,+1)
  - P1/2: F=0 (mF=0), F=1 (mF=-1,0,+1)
  - D3/2: F=1 (mF=-1,0,+1), F=2 (mF=-2,-1,0,+1,+2)
- Helper: `HF_BASIS_ORDER` and `HF_INDEX[(level,F,mF)]` in `examples/rabi_flopping_Yb171.py` map these to indices; `HF_BASIS_DIM = 16`.

## Usage roadmap
- Diagonal energies: use hyperfine formula \(E_F = \frac{A}{2}K + \frac{B}{8}\frac{3K(K+1)-4I(I+1)J(J+1)}{I(2I-1)J(2J-1)}\) with \(K = F(F+1)-I(I+1)-J(J+1)\). For S1/2, add the quadratic Zeeman term +310.6 Hz · B² on top of the linear Zeeman shift \(g_F \mu_B m_F B\).
- Basis expansion target: full 16-state hyperfine manifold for S1/2 (F=0,1), P1/2 (F=0,1), D3/2 (F=1,2) with all m_F sublevels.
- Couplings: E1 angular factors use 3j (m_F) and 6j (J,I,F) with the same reduced matrix elements as the bosonic case; placeholders remain in `REDUCED_E1_ELEMENTS` until real values are provided.
- Detunings: reference laser detunings to the hyperfine-resolved transition frequencies once the diagonal energies are updated.
- Collapse operators: decay branching should respect the hyperfine level ordering once energies are in place (even if total Γ is unchanged).

## Open items
- Import the actual reduced E1 values when available; current values are placeholders.
- Decide ordering/indexing of the 16 basis states (e.g., block by manifold then m_F increasing) and update all mapping helpers accordingly.
- Confirm if any electric-quadrupole B constants are to be included for D3/2; currently only A is specified.
