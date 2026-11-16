"""Reproducible resonant pulse experiment shared by the test and README figure."""

import numpy as np
from qutip import basis, mesolve, qeye, tensor
from scipy.special import erf

from . import yb171_simulator as model
from .config import DEFAULT_FOCK_DIMENSION


def run_rabi_validation() -> dict:
    """Original analytic test setup, with two uncoupled vacuum cavity spaces.

    No physical settings were retuned for the figure. The cavity, decay,
    reset coupling, and bare terms are off, isolating S(F=0,mF=0) and
    P(F=1,mF=0). The cavity truncation follows the shared application default.
    """
    # Tight integration isolates numerical error in the analytic comparison.
    n = DEFAULT_FOCK_DIMENSION
    ground = model.hf_state_index("S", 0.0, 0.0)
    excited = model.hf_state_index("P", 1.0, 0.0)
    atom_h = model._build_systems_hf("SP", 1.0, 16)["pi"]
    h = tensor(qeye(n), qeye(n), atom_h)
    times = np.linspace(0.0, 2e-6, 201)
    args = dict(drive_amp=2*np.pi*1e6, drive_t0=1e-6, drive_sigma=.2e-6)
    psi0 = tensor(basis(n, 0), basis(n, 0), basis(16, ground))
    p_e = tensor(qeye(n), qeye(n), basis(16, excited).proj())
    result = mesolve(
        [h*0, [h, model.omega_drive]], psi0, times,
        e_ops=[p_e], args=args, options={**model.solver_options(), "atol": 1e-10, "rtol": 1e-8},
    )
    sigma, t0 = args["drive_sigma"], args["drive_t0"]
    area = args["drive_amp"]*sigma*np.sqrt(np.pi/2)*(
        erf((times-t0)/(np.sqrt(2)*sigma))-erf(-t0/(np.sqrt(2)*sigma))
    )
    analytic = np.sin(abs(atom_h[excited, ground])*area)**2
    simulated = np.asarray(result.expect[0])
    return dict(time_us=times*1e6, simulated=simulated, analytic=analytic,
                residual=simulated-analytic, fock_dimension=n)
