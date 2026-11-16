"""
three_level_model.py

Simulates a three-level atom in a cavity using QuTiP.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip import steadystate
from tqdm import tqdm


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
    The system is described in the rotating frame and solved using qutip.mesolve
    with collapse operators for cavity decay and spontaneous emission.
    """
    # Parameters
    times = np.linspace(t_start, t_end, t_steps)

    Ncav = Ncav  # photon truncation

    sigma_ge, sigma_se, a, P_g, P_s, P_e = define_basis_operators(Ncav)

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


def H_op(sigma_ge, sigma_se, a, const):
    """
    Construct the Hamiltonian for the three-level atom-cavity system given environment constants.

    TBD
    """
    H = (
        const["detune_c"] * a.dag() * a
        + const["detune_l"] * sigma_se.dag() * sigma_se
        + const["gval"] * (a.dag() * sigma_ge + a * sigma_ge.dag())
        + const["rabi"] * (sigma_se + sigma_se.dag())
    )
    return H


def sweep_parameters(parameter, parameter_range, constants):
    """
    Sweep a given system parameter and compute steady-state observables.

    Parameters
    ----------
    parameter : str
        Name of the parameter to sweep. Must be one of:
        'gval', 'rabi', 'kappa', or 'gamma'.
    parameter_range : array-like
        Range of values for the chosen parameter.
    constants : dict
        Dictionary of constant system parameters, including:
        'gamma', 'kappa', 'rabi', 'gval', 'detune_c', 'detune_l', 'Ncav'.

    Returns
    -------
    tuple
        - parameter_range : np.ndarray
            Values of the swept parameter.
        - pg : list of float
            Population in ground state |g⟩.
        - pe : list of float
            Population in excited state |e⟩.
        - ps : list of float
            Population in shelving state |s⟩.
        - photons : list of float
            Mean photon number ⟨a†a⟩ in the cavity.
    """
    sigma_ge, sigma_se, a, P_g, P_s, P_e = define_basis_operators(constants["Ncav"])
    pg, pe, ps, photons, eff, coop = [], [], [], [], [], []

    for value in tqdm(parameter_range):
        constants[parameter] = value  # update sweep parameter
        H = H_op(sigma_ge, sigma_se, a, constants)  # update Hamiltonian

        c_ops = [
            np.sqrt(constants["kappa"]) * a,
            np.sqrt(constants["gamma"]) * sigma_ge,
        ]

        rho_ss = steadystate(H, c_ops)
        pg.append(expect(P_g, rho_ss))
        pe.append(expect(P_e, rho_ss))
        ps.append(expect(P_s, rho_ss))
        photons.append(expect(a.dag() * a, rho_ss))
        efficiency = (constants["kappa"]* expect(a * a.dag(), rho_ss)) / (
                    constants["gamma"]* expect(sigma_ge * sigma_ge.dag(), rho_ss)
                    + constants["kappa"]* expect(a * a.dag(), rho_ss)
                )
        eff.append(efficiency)
        coop.append((constants["gval"] * constants["gval"]) / (constants["kappa"] * constants["gamma"]))

    return pg, pe, ps, photons, eff, coop


if __name__ == "__main__":
    constants = {
        "gamma": 0.05,
        "kappa": 0.2,
        "rabi": 2.0,
        "gval": 1.0,
        "detune_c": 1.0,
        "detune_l": 1.0,
        "Ncav": 4,
    }
    times, n_pop, g_pop, e_pop, s_pop = run_sim()
    plt.plot(times, n_pop, label="Photon numbers")
    plt.plot(times, g_pop, label="|e⟩ population")
    plt.plot(times, e_pop, label="|g⟩ population")
    plt.plot(times, s_pop, label="|s⟩ population")
    plt.plot(times, g_pop + e_pop + s_pop, label="Total population")
    plt.xlabel("Time")
    plt.legend()
    plt.grid()
    plt.show()

    constants_gval = {
        "gamma": 0.05,
        "kappa": 0.2,
        "rabi": 2.0,
        "detune_c": 1.0,
        "detune_l": 1.0,
        "Ncav": 4,
    }
    g_val_vals = np.linspace(0.01, 1.5, 100)
    pg_gval,pe_gval,ps_gval,photons_gval, eff_gval, coop_gval = sweep_parameters('gval', g_val_vals, constants_gval)
    plt.plot(g_val_vals, pg_gval, label='|g>')
    plt.plot(g_val_vals, pe_gval, label='|e>')
    plt.plot(g_val_vals, ps_gval, label='|s>')
    plt.plot(g_val_vals, photons_gval, label='photons')
    plt.plot(g_val_vals, eff_gval, label='efficiency')
    plt.xlabel('g_val')
    plt.ylabel('Population')
    plt.title('g_val Sweep')
    plt.legend()
    plt.grid()
    plt.show()
    
    constants_kappa = {
        "gamma": 0.05,
        "rabi": 2.0,
        "gval": 1.0,
        "detune_c": 1.0,
        "detune_l": 1.0,
        "Ncav": 4,
    }

    kappa_vals = np.linspace(0.01, 1, 1000)
    pg_kappa,pe_kappa,ps_kappa,photons_kappa, eff_kappa, coop_kappa = sweep_parameters('kappa', kappa_vals, constants_kappa)
    plt.plot(kappa_vals, pg_kappa, label='|g>')
    plt.plot(kappa_vals, pe_kappa, label='|e>')
    plt.plot(kappa_vals, ps_kappa, label='|s>')
    plt.plot(kappa_vals, photons_kappa, label='photons')
    plt.plot(kappa_vals, eff_kappa, label='effiency')
    plt.xlabel('Kappa')
    plt.ylabel('Population')
    plt.title('Kappa Sweep')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(coop_kappa, eff_kappa)
    plt.xlabel('Cooperativity')
    plt.ylabel('Efficiency')
    plt.title('Copperativity Sweep (ΔKappa)')
    plt.grid()
    plt.show()

    constants_gamma = {
        "kappa": 0.2,
        "rabi": 2.0,
        "gval": 1.0,
        "detune_c": 1.0,
        "detune_l": 1.0,
        "Ncav": 4,
    }

    gamma_vals = np.linspace(0.01, 1, 1000)
    pg_gamma,pe_gamma,ps_gamma,photons_gamma, eff_gamma, coop_gamma = sweep_parameters('gamma', gamma_vals, constants_gamma)
    plt.plot(gamma_vals, pg_gamma, label='|g>')
    plt.plot(gamma_vals, pe_gamma, label='|e>')
    plt.plot(gamma_vals, ps_gamma, label='|s>')
    plt.plot(gamma_vals, photons_gamma, label='photons')
    plt.plot(gamma_vals, eff_gamma, label='efficiency')
    plt.xlabel('Gamma')
    plt.ylabel('Population')
    plt.title('Gamma Sweep')
    plt.legend()
    plt.grid()
    plt.show()
    plt.plot(coop_gamma, eff_gamma)
    plt.xlabel('Cooperativity')
    plt.ylabel('Efficiency')
    plt.title('Copperativity Sweep (ΔGamma)')
    plt.grid()
    plt.show()
    print("Finished")