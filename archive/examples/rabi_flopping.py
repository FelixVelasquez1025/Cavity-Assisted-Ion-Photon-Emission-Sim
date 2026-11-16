"""
Rabi flopping on a two-level system using QuTiP.

Paper mapping (examples/rabi_hamiltonian.qmd):
- Implements the section "Hamiltonian for the Two Level System".
- H0: `wc * a^\dag a` (cavity) and `(wa/2) * sz` (atom) correspond to the
  free/cavity and atomic inversion energies described as H0 in the doc.
- Interaction (RWA): `g * (a * sigma_+ + a^\dag * sigma_-)` is the
  Jaynes–Cummings Hamiltonian in the doc.
- Interaction (non‑RWA): `g * (a + a^\dag) * (sigma_- + sigma_+)` matches the
  pre‑RWA light–matter coupling form.
- Collapse operators: cavity damping (kappa, n_th) and atomic decay (gamma)
  implement the Lindblad terms used in the master‑equation `mesolve` solver.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from qutip import about, basis, destroy, mesolve, qeye, sigmaz, sigmam, sigmap, tensor



# Define the sample parameters (goal is to use real Yb and cavity parameters later but this will do for now)
N = 15                # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005         # cavity dissipation rate
gamma = 0.05          # atom dissipation rate
n_th_a = 2            # temperature in frequency units
use_rwa = True

tlist = np.linspace(0, 40, 100)

# initial state:
psi0 = tensor(basis(N, 0), basis(2, 0))

# collapse operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), sigmam())          # sigma_- (lowering)
sp = tensor(qeye(N), sigmap())          # sigma_+ (raising)
sz = tensor(qeye(N), sigmaz())

# Hamiltonian with and without RWA
# Paper: Two-level Hamiltonian (RWA vs full). See the two-level section
# in examples/rabi_hamiltonian.qmd for the explicit forms and symbols.
if use_rwa:
    # Jaynes–Cummings Hamiltonian under RWA: g (a σ+ + a† σ−)
    H = wc * a.dag() * a + (wa / 2) * sz + g * (a * sp + a.dag() * sm)
else:
    # Full (non-RWA) light–matter coupling: g (a + a†)(σ− + σ+)
    H = wc * a.dag() * a + (wa / 2) * sz + g * (a.dag() + a) * (sm + sp)

# Create a list of collapse operators
c_op_list = []
# Paper: Master equation dissipators. Cavity bath via kappa and n_th,
# and atomic spontaneous emission via gamma, as discussed alongside the
# two-level Hamiltonian and the use of `mesolve`.

# Photon annihilation
rate = kappa * (1 + n_th_a)
if rate > 0:
    c_op_list.append(np.sqrt(rate) * a)

# Photon creation
rate = kappa * n_th_a
if rate > 0:
    c_op_list.append(np.sqrt(rate) * a.dag())

# Atomic annihliation
if gamma > 0:
    c_op_list.append(np.sqrt(gamma) * sm)

# Use Mesolver to evolve the system by solving the master equation
output = mesolve(
    H,
    psi0,
    tlist,
    c_ops=c_op_list,
    e_ops=[a.dag() * a, sp * sm],  # sp*sm = |1><1| projector
)

#Plot the results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist, output.expect[0], label="Cavity")
ax.plot(tlist, output.expect[1], label="Atom excited state")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability")
ax.set_title("Vacuum Rabi oscillations at T={}".format(n_th_a));
plt.show()
