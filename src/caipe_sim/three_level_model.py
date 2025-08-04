"""
three_level_model.py

Simulates a three-level atom in a cavity using QuTiP.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip import steadystate


def run_sim(
    t_start=0.0,
    t_end=10.0,
    t_steps=200,
    Ncav=4,
    detune_c=1.0,
    detune_l=1.0,
    g_val=1.0,
    rabi=2.0,
    kappa=0.2,
    gamma=0.05,
    init_cav=0,
    init_sys=2,
):
    """
    Simulate the time evolution of a three-level atom in a cavity using QuTiP.

    This function models the dynamics of a Λ-type atom interacting with a single-mode cavity field
    and a classical drive, including dissipation due to spontaneous emission and cavity decay.
    It solves the Lindblad master equation and returns the time evolution of photon number and
    atomic populations.

    Parameters
    ----------
    t_start : float, optional
        Initial time of the simulation (default is 0.0).
    t_end : float, optional
        Final time of the simulation (default is 10.0).
    t_steps : int, optional
        Number of time steps (default is 200).
    Ncav : int, optional
        Hilbert space truncation for the cavity photon number (default is 4).
    detune_c : float, optional
        Cavity detuning relative to the atomic |g⟩ ↔ |e⟩ transition (default is 1.0).
    detune_l : float, optional
        Laser detuning relative to the atomic |s⟩ ↔ |e⟩ transition (default is 1.0).
    g_val : float, optional
        Coupling strength (vacuum Rabi frequency) between the atom and cavity (default is 1.0).
    rabi : float, optional
        Rabi frequency of the classical drive on the |s⟩ ↔ |e⟩ transition (default is 2.0).
    kappa : float, optional
        Cavity decay rate (default is 0.2).
    gamma : float, optional
        Spontaneous emission rate from |e⟩ to |g⟩ (default is 0.05).
    init_cav : int, optional
        Initial photon number in the cavity (default is 0).
    init_sys : int, optional
        Initial atomic state index: 0 = |g⟩, 1 = |e⟩, 2 = |s⟩ (default is 2).

    Returns
    -------
    tuple of list of float
        - times : list
            List of steps where differential equation is solved.
        - expect_photon : list
            Expectation value of the cavity photon number ⟨a†a⟩ over time.
        - expect_e : list
            Population in the excited state |e⟩ over time.
        - expect_g : list
            Population in the ground state |g⟩ over time.
        - expect_s : list
            Population in the metastable state |s⟩ over time.

    Notes
    -----
    The system is described in the rotating frame and solved using `qutip.mesolve`
    with collapse operators for cavity decay and spontaneous emission.
    """
    # Parameters
    times = np.linspace(t_start, t_end, t_steps)

    Ncav = Ncav  # photon truncation
    Natom = 3  # three-level atom

    # Define basis states
    g = basis(3, 0)  # ground state
    e = basis(3, 1)  # excited state
    s = basis(3, 2)  # metastable state

    # Operators on atom
    sigma_ge = g * e.dag()
    sigma_se = s * e.dag()
    I_atom = qeye(3)

    # Operators on cavity
    a = destroy(Ncav)
    I_cav = qeye(Ncav)

    # Tensor operators
    a = tensor(a, I_atom)
    sigma_ge = tensor(I_cav, sigma_ge)
    sigma_se = tensor(I_cav, sigma_se)

    proj_g = basis(3, 0) * basis(3, 0).dag()  # |g⟩⟨g|
    proj_s = basis(3, 2) * basis(3, 2).dag()  # |s⟩⟨s|
    proj_e = basis(3, 1) * basis(3, 1).dag()

    # Tensor with identity on cavity space
    P_g = tensor(qeye(Ncav), proj_g)
    P_s = tensor(qeye(Ncav), proj_s)
    P_e = tensor(qeye(Ncav), proj_e)

    # Parameters
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = rabi  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition

    # Hamiltonian
    H = (
        detune_c * a.dag() * a
        + detune_l * sigma_se.dag() * sigma_se
        + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
        + rabi * (sigma_se + sigma_se.dag())
    )
    kappa = kappa  # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)

    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    psi = tensor(basis(Ncav, init_cav), basis(3, init_sys))

    result = mesolve(
        H, psi, times, c_ops, [a.dag() * a, sigma_ge.dag() * sigma_ge, P_g, P_s]
    )
    return times, result.expect[0], result.expect[1], result.expect[2], result.expect[3]


def define_basis_operators(Ncav=4):
    """
    Defines the key operators for a three-level atom coupled to an optical cavity.

    Parameters
    ----------
    Ncav : int, optional
        maximum amount of photons in optical cavity, default = 4

    Returns
    -------
    tuple
        - sigma_ge : Qobj
            Lowering operator from |e⟩ to |g⟩.
        - sigma_se : Qobj
            Lowering operator from |e⟩ to |s⟩.
        - a : Qobj
            Cavity annihilation operator.
        - P_g : Qobj
            Projector onto the ground state |g⟩.
        - P_s : Qobj
            Projector onto the metastable state |s⟩.
        - P_e : Qobj
            Projector onto the excited state |e⟩.
    """  
    g = basis(3, 0)
    e = basis(3, 1)
    s = basis(3, 2)

    sigma_ge = tensor(qeye(Ncav), g * e.dag())
    sigma_se = tensor(qeye(Ncav), s * e.dag())
    a = tensor(destroy(Ncav), qeye(3))

    P_g = tensor(qeye(Ncav), g * g.dag())
    P_s = tensor(qeye(Ncav), s * s.dag())
    P_e = tensor(qeye(Ncav), e * e.dag())

    return sigma_ge, sigma_se, a, P_g, P_s, P_e


def sweep_kappa(
    kappa_values_start,
    kappa_values_end,
    kappa_values_steps,
    gamma=0.05,
    Ncav=4,
    detune_c=1.0,
    detune_l=1.0,
    g_val=1.0,
    rabi=2.0,
):
    """
    Sweep over different kappa values and compute steady-state observables.

    Parameters
    ----------
    kappa_values_start : integer
        Start of kappa (cavity decay rates) range to sweep
    kappa_values_end : integer
        End of kappa range to sweep
    kappa_values_steps : integer
        Number of kappa values to sweep through
    gamma : float
        Fixed spontaneous emission rate.

    Returns
    -------
    tuple of lists
        Populations in |g>, |e>, |s> and photon number over all kappa values.
    """
    g = basis(3, 0)
    e = basis(3, 1)
    s = basis(3, 2)

    sigma_ge = tensor(qeye(Ncav), g * e.dag())
    sigma_se = tensor(qeye(Ncav), s * e.dag())
    a = tensor(destroy(Ncav), qeye(3))

    P_g = tensor(qeye(Ncav), g * g.dag())
    P_s = tensor(qeye(Ncav), s * s.dag())
    P_e = tensor(qeye(Ncav), e * e.dag())

    H = (
        detune_c * a.dag() * a
        + detune_l * sigma_se.dag() * sigma_se
        + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
        + rabi * (sigma_se + sigma_se.dag())
    )

    pg, pe, ps, photons = [], [], [], []
    kappa_values = np.linspace(kappa_values_start, kappa_values_end, kappa_values_steps)
    for kappa in kappa_values:
        c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
        rho_ss = steadystate(H, c_ops)
        pg.append(expect(P_g, rho_ss))
        pe.append(expect(P_e, rho_ss))
        ps.append(expect(P_s, rho_ss))
        photons.append(expect(a.dag() * a, rho_ss))
    return kappa_values, pg, pe, ps, photons


def sweep_gamma(
    gamma_values_start,
    gamma_values_end,
    gamma_values_steps,
    kappa=0.2,
    Ncav=4,
    detune_c=1.0,
    detune_l=1.0,
    g_val=1.0,
    rabi=2.0,
):
    """
    Sweep over different gamma values and compute steady-state observables.

    Parameters
    ----------
    gamma_values_start : integer
        Start of gamma range to sweep
    gamma_values_end : integer
        End of gamma range to sweep
    gamma_values_steps : integer
        Number of gamma values to sweep through
    gamma : float
        Fixed cavity decay rate.

    Returns
    -------
    tuple of lists
        Populations in |g>, |e>, |s> and photon number over all gamma values.
    """
    g = basis(3, 0)
    e = basis(3, 1)
    s = basis(3, 2)

    sigma_ge = tensor(qeye(Ncav), g * e.dag())
    sigma_se = tensor(qeye(Ncav), s * e.dag())
    a = tensor(destroy(Ncav), qeye(3))

    P_g = tensor(qeye(Ncav), g * g.dag())
    P_s = tensor(qeye(Ncav), s * s.dag())
    P_e = tensor(qeye(Ncav), e * e.dag())

    H = (
        detune_c * a.dag() * a
        + detune_l * sigma_se.dag() * sigma_se
        + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
        + rabi * (sigma_se + sigma_se.dag())
    )

    pg, pe, ps, photons = [], [], [], []

    gamma_values = np.linspace(gamma_values_start, gamma_values_end, gamma_values_steps)

    for gamma in gamma_values:
        c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
        rho_ss = steadystate(H, c_ops)
        pg.append(expect(P_g, rho_ss))
        pe.append(expect(P_e, rho_ss))
        ps.append(expect(P_s, rho_ss))
        photons.append(expect(a.dag() * a, rho_ss))
    return gamma_values, pg, pe, ps, photons


def sweep_rabi(
    rabi_vals_start,
    rabi_vals_end,
    rabi_vals_step,
    gamma=0.05,
    Ncav=4,
    detune_c=1.0,
    detune_l=1.0,
    g_val=1.0,
    kappa=0.2,
):
    """
    Sweep over different Rabi frequencies, gives the laser strength, and compute steady-state observables.

    Parameters
    ----------
    rabi_vals_start : float
        Starting value of the Rabi frequency to sweep.
    rabi_vals_end : float
        Final value of the Rabi frequency to sweep.
    rabi_vals_step : int
        Number of Rabi frequency steps to evaluate.

    Returns
    -------
    tuple of lists
        - rabi_values : list
            Rabi frequencies swept.
        - pg : list
            Population in state |g⟩ for each Rabi frequency.
        - pe : list
            Population in state |e⟩ for each Rabi frequency.
        - ps : list
            Population in state |s⟩ for each Rabi frequency.
        - photons : list
            Mean photon number ⟨a†a⟩ for each Rabi frequency.
    """

    rabi_values = np.linspace(rabi_vals_start, rabi_vals_end, rabi_vals_step)
    sigma_ge, sigma_se, a, P_g, P_s, P_e = define_basis_operators(Ncav)
    pg, pe, ps, photons = [], [], [], []

    for rabi in rabi_values:
        H = (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi * (sigma_se + sigma_se.dag())
        )

        c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
        rho_ss = steadystate(H, c_ops)
        pg.append(expect(P_g, rho_ss))
        pe.append(expect(P_e, rho_ss))
        ps.append(expect(P_s, rho_ss))
        photons.append(expect(a.dag() * a, rho_ss))
    return rabi_values, pg, pe, ps, photons


def sweep_gval(
    gval_vals_start,
    gval_vals_end,
    gval_vals_step,
    gamma=0.05,
    Ncav=4,
    detune_c=1.0,
    detune_l=1.0,
    kappa=0.2,
    rabi=2.0,
):
    """
    Sweep over different atom-cavity coupling strengths (g) and compute steady-state observables.

    Parameters
    ----------
    gval_vals_start : float
        Starting value of g to sweep.
    gval_vals_end : float
        Final value of g to sweep.
    gval_vals_step : int
        Number of g values to evaluate.

    Returns
    -------
    tuple of lists
        - gval_values : list
            Coupling strengths swept.
        - pg : list
            Population in state |g⟩ for each g value.
        - pe : list
            Population in state |e⟩ for each g value.
        - ps : list
            Population in state |s⟩ for each g value.
        - photons : list
            Mean photon number ⟨a†a⟩ for each g value.
    """

    gval_values = np.linspace(gval_vals_start, gval_vals_end, gval_vals_step)
    sigma_ge, sigma_se, a, P_g, P_s, P_e = define_basis_operators(Ncav)
    pg, pe, ps, photons = [], [], [], []

    for gval_value in gval_values:
        H = (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + gval_value * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi * (sigma_se + sigma_se.dag())
        )

        c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
        rho_ss = steadystate(H, c_ops)
        pg.append(expect(P_g, rho_ss))
        pe.append(expect(P_e, rho_ss))
        ps.append(expect(P_s, rho_ss))
        photons.append(expect(a.dag() * a, rho_ss))
    return gval_values, pg, pe, ps, photons


if __name__ == "__main__":
    times, n_pop, g_pop, e_pop, s_pop = run_sim()
    plt.plot(times, n_pop, label="Photon numbers")
    plt.plot(times, g_pop, label="|e⟩ population")
    plt.plot(times, e_pop, label="|g⟩ population")
    plt.plot(times, s_pop, label="|s⟩ population")
    plt.plot(times, g_pop + e_pop + s_pop, label="Total population")
    plt.legend()
    plt.show()