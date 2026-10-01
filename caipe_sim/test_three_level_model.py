"""
three_level_model.py

Simulates a three-level atom in a cavity using QuTiP.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip import steadystate
from tqdm import tqdm


def run_sim(
    t_start=0.0,
    t_end=100.0,
    t_steps=2000,
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


# times, n_pop, g_pop, e_pop, s_pop = run_sim()
# plt.plot(times, n_pop, label="photon numbers")
# plt.plot(times, g_pop, label=" |e> pop")
# plt.plot(times, e_pop, label="|g⟩ pop")
# plt.plot(times, s_pop, label="|s⟩ pop")
# plt.plot(times, g_pop + e_pop + s_pop, label="total pop")
# plt.legend()
# plt.show()


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


# kap_vals, kap_g, kap_e, kap_s, kap_photon = sweep_kappa(0.01, 1, 100)
# gam_vals, gam_g, gam_e, gam_s, gam_photon = sweep_gamma(0.01, 1, 100)

# plt.plot(kap_vals, kap_g, label="|g> population")
# plt.plot(kap_vals, kap_e, label="|e> poopulation")
# plt.plot(kap_vals, kap_s, label="|s> population")
# plt.plot(kap_vals, kap_photon, label="Photon population")
# plt.xlabel("Kappa")
# plt.title("Steady State Kappa Sweep")
# plt.legend()
# plt.show()

# plt.plot(gam_vals, gam_g, label="|g> population")
# plt.plot(gam_vals, gam_e, label="|e> poopulation")
# plt.plot(gam_vals, gam_s, label="|s> population")
# plt.plot(gam_vals, gam_photon, label="Photon population")
# plt.legend()
# plt.show()


def sweep_coop(
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
    coop = []
    for i in range(0, len(gamma_values)):
        coop.append((g_val * g_val) / (kappa * gamma_values[i]))
    coop = coop.sort()
    return coop, pg, pe, ps, photons


# plt.plot()


def sweep_coop_eff(Ncav=4, detune_c=1.0, detune_l=1.0, rabi=2.0):
    """
    Sweeps over g, kappa, and gamma to compute the cooperativity and photon generation efficiency,
    categorizing the results into four cavity QED regimes: strong, intermediate, Purcell, and other.

    Parameters
    ----------
    Ncav : int, optional
        Truncation level of the cavity photon number Hilbert space.
    detune_c : float, optional
        Detuning between cavity mode and atomic |g⟩↔|e⟩ transition.
    detune_l : float, optional
        Detuning between classical laser and atomic |s⟩↔|e⟩ transition.
    rabi : float, optional
        Rabi frequency of the classical drive on the |s⟩↔|e⟩ transition.

    Returns
    -------
    tuple of lists
        intermediate : list of float
            Cooperativity values in the intermediate regime.
        eff_i : list of float
            Efficiencies in the intermediate regime.
        strong : list of float
            Cooperativity values in the strong coupling regime.
        eff_s : list of float
            Efficiencies in the strong coupling regime.
        purcell : list of float
            Cooperativity values in the Purcell regime.
        eff_p : list of float
            Efficiencies in the Purcell regime.
        other : list of float
            Cooperativity values not falling into the above regimes.
        eff_o : list of float
            Efficiencies corresponding to `other` regime.
    """
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
    g_val = np.linspace(
        0.01, 10, 1000
    )  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = rabi  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition
    kappa = np.linspace(
        0.01, 10, 1000
    )  # Cavity field decay rate (photon leakage through mirrors)
    gamma = np.linspace(
        0.01, 10, 1000
    )  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # coop = []
    eff_i = []
    eff_s = []
    eff_p = []
    eff_o = []
    intermediate = []
    strong = []
    purcell = []
    other = []
    # Hamiltonian

    for i in range(0, len(g_val)):
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", i)
        for j in range(0, len(kappa)):
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", j)
            for k in range(0, len(gamma)):
                cooperativ = (g_val[i] * g_val[i]) / (kappa[j] * gamma[k])
                # coop.append(cooperativ)
                H = (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val[i] * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi * (sigma_se + sigma_se.dag())
                )

                c_ops = [np.sqrt(kappa[j]) * a, np.sqrt(gamma[k]) * sigma_ge]
                rho_ss = steadystate(H, c_ops)
                efficiency = (kappa[j] * expect(a * a.dag(), rho_ss)) / (
                    gamma[k] * expect(sigma_ge * sigma_ge.dag(), rho_ss)
                    + kappa[j] * expect(a * a.dag(), rho_ss)
                )
                g_over_kappa = g_val[i] / kappa[j]
                g_over_gamma = g_val[i] / gamma[k]
                kappa_over_g = kappa[j] / g_val[i]

                if g_over_kappa > 5 and g_over_gamma > 10:
                    strong.append(cooperativ)
                    eff_s.append(efficiency)
                elif 0.5 < kappa_over_g < 2 and gamma[k] < 0.1 * g_val[i]:
                    intermediate.append(cooperativ)
                    eff_i.append(efficiency)
                elif kappa_over_g > 5 and g_over_gamma > 5:
                    purcell.append(cooperativ)
                    eff_p.append(efficiency)
                else:
                    other.append(cooperativ)
                    eff_o.append(efficiency)
                print(k)

    return intermediate, eff_i, strong, eff_s, purcell, eff_p, other, eff_o


# intermediate, eff_i, strong, eff_s, purcell, eff_p, other, eff_o = sweep_coop_eff()
# plt.scatter(strong, eff_s, color="red", label="strong")
# plt.scatter(intermediate, eff_i, color="blue", label="intermediate")
# plt.scatter(purcell, eff_p, color="green", label="purcell")
# plt.scatter(other, eff_o, color="black", label="other")
# plt.xlabel("Cooperativity")
# plt.ylabel("Efficiency")
# plt.legend()
# plt.show()


def sweep_kappa_eff(
    Ncav=4, detune_c=1.0, detune_l=1.0, g_val=1.0, rabi=2.0, gamma=0.005
):
    """
    Sweeps over cavity decay rate `kappa` while holding other parameters fixed to study
    photon generation efficiency as a function of `kappa`.

    Parameters
    ----------
    Ncav : int, optional
        Truncation level of the cavity photon number Hilbert space.
    detune_c : float, optional
        Detuning between cavity mode and atomic |g⟩↔|e⟩ transition.
    detune_l : float, optional
        Detuning between classical laser and atomic |s⟩↔|e⟩ transition.
    g_val : float, optional
        Atom–cavity coupling strength.
    rabi : float, optional
        Rabi frequency of the classical drive on the |s⟩↔|e⟩ transition.
    gamma : float, optional
        Spontaneous decay rate from |e⟩ to |g⟩.

    Returns
    -------
    tuple
        kappa : np.ndarray
            Array of kappa values used in the sweep.
        eff : list of float
            Photon generation efficiency corresponding to each kappa.
    """
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
    kappa = np.linspace(
        0.01, 3, 3000
    )  # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = (
        detune_c * a.dag() * a
        + detune_l * sigma_se.dag() * sigma_se
        + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
        + rabi * (sigma_se + sigma_se.dag())
    )
    eff = []
    for i in range(0, len(kappa)):
        c_ops = [np.sqrt(kappa[i]) * a, np.sqrt(gamma) * sigma_ge]
        rho_ss = steadystate(H, c_ops)
        effiency = (kappa[i] * expect(a * a.dag(), rho_ss)) / (
            gamma * expect(sigma_ge * sigma_ge.dag(), rho_ss)
            + kappa[i] * expect(a * a.dag(), rho_ss)
        )
        eff.append(effiency)
    return kappa, eff


# kappa, eff = sweep_kappa_eff()
# plt.plot(kappa, eff)
# plt.plot(0.2, max(eff), linestyle="dotted")
# plt.plot(2.0, max(eff), linestyle="dotted")
# plt.ylabel("Efficiency")
# plt.xlabel("κ")
# plt.show()

def pulse_shape(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, rabi=0.5, gamma=0.05, kappa = 0.01, init_cav = 0, init_sys = 2):
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
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = (
        detune_c * a.dag() * a
        + detune_l * sigma_se.dag() * sigma_se
        + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
        + rabi * (sigma_se + sigma_se.dag())
    )
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    times = np.linspace(0,300,1000)
    rho = [rho_init]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse2 = [kappa*expect(a.dag()*a, rho_init)]
    for i_time in tqdm(range(1, len(times))):
        solver = MESolver(H, c_ops = c_ops)
        solver.start(rho[i_time-1], times[i_time-1])
        rho_t = solver.step(times[i_time])
        # pulse1.append(rho_t.expect[0])
        pulse2.append(expect(a.dag()*a, rho_t))
        rho.append(rho_t)


    
    return pulse2, times 
pulse2,times = pulse_shape()
# plt.plot(times, pulse1)
# plt.xlabel('Time /s')
# plt.ylabel('Pulse Amp')
# plt.title('Pulse1')
# plt.show()
plt.plot(times, pulse2)
plt.xlabel('Time /s')
plt.ylabel('Pulse Amp')
plt.title('Pulse2')
plt.show()


def pulse_shape_strong_bad(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.01, kappa = 0.2, init_cav = 0, init_sys = 2):
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
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi = []  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition
    for i in range(0, len(ratios)):
        rabi.append(ratios[i]*g_val)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = []
    for i in range(0,len(rabi)):
        H .append( (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi[i] * (sigma_se + sigma_se.dag())
        ))
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    times = np.linspace(0,300,1000)
    rho = [rho_init]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    all_pulse = []
    for j in tqdm(range(0,len(H))):
        rho = tensor(basis(Ncav, init_cav), basis(3, init_sys)).proj()
        solver = MESolver(H[j], c_ops=c_ops)
        solver.start(rho, times[0])
        pulse = [kappa * expect(a.dag()*a, rho)]
        for i_time in tqdm(range(1, len(times))):
            rho = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            amp = (expect(a.dag()*a, rho))
            pulse.append(kappa*amp)
        all_pulse.append(np.array(pulse))


    
    return rabi, times, all_pulse
rabi_list_s_bad, times_s_bad, all_pulse_s_bad = pulse_shape_strong_bad()

for rabi, pulse in zip(rabi_list_s_bad, all_pulse_s_bad):
    plt.plot(times_s_bad, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Bad)")
plt.legend()
plt.tight_layout()
plt.show()


def pulse_shape_strong_good(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.1, kappa = 0.01, init_cav = 0, init_sys = 2):
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
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi = []  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition
    for i in range(0, len(ratios)):
        rabi.append(ratios[i]*g_val)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = []
    for i in range(0,len(rabi)):
        H .append( (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi[i] * (sigma_se + sigma_se.dag())
        ))
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    times = np.linspace(0,300,1000)
    rho = [rho_init]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    all_pulse = []
    for j in tqdm(range(0,len(H))):
        rho = tensor(basis(Ncav, init_cav), basis(3, init_sys)).proj()
        solver = MESolver(H[j], c_ops=c_ops)
        solver.start(rho, times[0])
        pulse = [kappa * expect(a.dag()*a, rho)]
        for i_time in tqdm(range(1, len(times))):
            rho = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            amp = (expect(a.dag()*a, rho))
            pulse.append(kappa*amp)
        all_pulse.append(np.array(pulse))


    
    return rabi, times, all_pulse
rabi_list_s_good, times_s_good, all_pulse_s_good = pulse_shape_strong_good()
for rabi, pulse in zip(rabi_list_s_good, all_pulse_s_good):
    plt.plot(times, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Good)")
plt.legend()
plt.tight_layout()
plt.show()


def pulse_shape_inter_bad(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.05, kappa = 0.5, init_cav = 0, init_sys = 2):
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
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi = []  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition
    for i in range(0, len(ratios)):
        rabi.append(ratios[i]*g_val)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = []
    for i in range(0,len(rabi)):
        H .append( (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi[i] * (sigma_se + sigma_se.dag())
        ))
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    times = np.linspace(0,300,1000)
    rho = [rho_init]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    all_pulse = []
    for j in tqdm(range(0,len(H))):
        rho = tensor(basis(Ncav, init_cav), basis(3, init_sys)).proj()
        solver = MESolver(H[j], c_ops=c_ops)
        solver.start(rho, times[0])
        pulse = [kappa * expect(a.dag()*a, rho)]
        for i_time in tqdm(range(1, len(times))):
            rho = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            amp = (expect(a.dag()*a, rho))
            pulse.append(kappa*amp)
        all_pulse.append(np.array(pulse))


    
    return rabi, times, all_pulse
rabi_list_int_bad, times_int_bad, all_pulse_int_bad = pulse_shape_inter_bad()
for rabi, pulse in zip(rabi_list_int_bad, all_pulse_int_bad):
    plt.plot(times, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title('Intermediate (Bad)')
plt.legend()
plt.tight_layout()
plt.show()


def pulse_shape_inter_good(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.5, kappa = 0.05, init_cav = 0, init_sys = 2):
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
    ratios = [0.05,0.15,0.3,0.5,1.0,2.5]
    rabi = []  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition
    for i in range(0, len(ratios)):
        rabi.append(ratios[i]*g_val)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = []
    for i in range(0,len(rabi)):
        H .append( (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi[i] * (sigma_se + sigma_se.dag())
        ))
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    times = np.linspace(0,300,1000)
    rho = [rho_init]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    all_pulse = []
    for j in tqdm(range(0,len(H))):
        rho = tensor(basis(Ncav, init_cav), basis(3, init_sys)).proj()
        solver = MESolver(H[j], c_ops=c_ops)
        solver.start(rho, times[0])
        pulse = [kappa * expect(a.dag()*a, rho)]
        for i_time in tqdm(range(1, len(times))):
            rho = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            amp = (expect(a.dag()*a, rho))
            pulse.append(kappa*amp)
        all_pulse.append(np.array(pulse))


    
    return rabi, times, all_pulse
rabi_list_int_good, times_int_good, all_pulse_int_good = pulse_shape_inter_good()
for rabi, pulse in zip(rabi_list_int_good, all_pulse_int_good):
    plt.plot(times, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title('Intermediate (Strong)')
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_purcell(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.2, gamma=0.01, kappa = 2.0, init_cav = 0, init_sys = 2):
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
    ratios = [0.1,0.3,0.6,1.0,2.0,5.0]
    rabi = []  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transition
    for i in range(0, len(ratios)):
        rabi.append(ratios[i]*kappa)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    H = []
    for i in range(0,len(rabi)):
        H .append( (
            detune_c * a.dag() * a
            + detune_l * sigma_se.dag() * sigma_se
            + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
            + rabi[i] * (sigma_se + sigma_se.dag())
        ))
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    times = np.linspace(0,300,1000)
    rho = [rho_init]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    all_pulse = []
    for j in tqdm(range(0,len(H))):
        rho = tensor(basis(Ncav, init_cav), basis(3, init_sys)).proj()
        solver = MESolver(H[j], c_ops=c_ops)
        solver.start(rho, times[0])
        pulse = [kappa * expect(a.dag()*a, rho)]
        for i_time in tqdm(range(1, len(times))):
            rho = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            amp = (expect(a.dag()*a, rho))
            pulse.append(kappa*amp)
        all_pulse.append(np.array(pulse))


    
    return rabi, times, all_pulse
rabi_list_purr, times_purr, all_pulse_purr= pulse_shape_purcell()
for rabi, pulse in zip(rabi_list_purr, all_pulse_purr):
    plt.plot(times, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title('Purcell')
plt.legend()
plt.tight_layout()
plt.show()  


def pulse_shape_strong_bad_linear(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.01, kappa = 0.2, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = np.linspace(0, 5, len(times))  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    rho = [rho_init]
    pulse2 = [expect(a.dag()*a, rho_init)]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]

    for i_time in tqdm(range(1, len(times))):
        H = ( (
                detune_c * a.dag() * a
                + detune_l * sigma_se.dag() * sigma_se
                + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                + rabi[i_time] * (sigma_se + sigma_se.dag())
            ))
        solver = MESolver(H, c_ops = c_ops)
        solver.start(rho[i_time-1], times[i_time-1])
        rho_t = solver.step(times[i_time])
        # pulse1.append(rho_t.expect[0])
        pulse2.append(expect(a.dag()*a, rho_t))
        rho.append(rho_t)
        


    
    return rabi, times, pulse2
rabi_list_s_bad_lin, times_s_bad_lin, all_pulse_s_bad_lin = pulse_shape_strong_bad_linear()
plt.plot(times_s_bad_lin, all_pulse_s_bad_lin)
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Bad), Linear Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_bad_square(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.01, kappa = 0.2, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
    rabi_before = np.zeros(int((1/3)*len(times))+1)  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi_after = np.zeros(int((1/3)*len(times))) 
    rabi = []
    for i in range(0,len(rabi_vals)):
        rabi_during = np.ones(int(((1/3)*len(times))))*rabi_vals[i]
        rabi.append(np.concatenate((rabi_before, rabi_during, rabi_after), axis = None))

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_s_bad_sq, times_s_bad_sq, all_pulse_s_bad_sq = pulse_shape_strong_bad_square()

for rabi, pulse in zip(rabi_list_s_bad_sq, all_pulse_s_bad_sq):
    plt.plot(times_s_bad_sq, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Bad), Sqaure Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_bad_gauss(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.01, kappa = 0.2, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    t0 = times[int(len(times)/2)]
    std = len(times)/10
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.exp(-((times[j]-t0)**2)/(2*std**2))
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
       
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_s_bad_gauss, times_s_bad_gauss, all_pulse_s_bad_gauss = pulse_shape_strong_bad_gauss()

for rabi, pulse in zip(rabi_list_s_bad_gauss, all_pulse_s_bad_gauss):
    plt.plot(times_s_bad_gauss, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Bad), Gaussian Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_bad_sin2(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.01, kappa = 0.2, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    T = times[-1]
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.sin(np.pi*times[j]/T)**2
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]

        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_s_bad_sin2, times_s_bad_sin2, all_pulse_s_bad_sin2 = pulse_shape_strong_bad_sin2()

for rabi, pulse in zip(rabi_list_s_bad_sin2, all_pulse_s_bad_sin2):
    plt.plot(times_s_bad_sin2, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Bad), Sin^2 Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_good_linear(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.1, kappa = 0.01, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = np.linspace(0, 5, len(times))  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    rho = [rho_init]
    pulse2 = [expect(a.dag()*a, rho_init)]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]

    for i_time in tqdm(range(1, len(times))):
        H = ( (
                detune_c * a.dag() * a
                + detune_l * sigma_se.dag() * sigma_se
                + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                + rabi[i_time] * (sigma_se + sigma_se.dag())
            ))
        solver = MESolver(H, c_ops = c_ops)
        solver.start(rho[i_time-1], times[i_time-1])
        rho_t = solver.step(times[i_time])
        # pulse1.append(rho_t.expect[0])
        pulse2.append(expect(a.dag()*a, rho_t))
        rho.append(rho_t)
        


    
    return rabi, times, pulse2
rabi_list_s_good_lin, times_s_good_lin, all_pulse_s_good_lin = pulse_shape_strong_good_linear()
plt.plot(times_s_good_lin, all_pulse_s_good_lin)
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Good), Linear Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_good_square(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.1, kappa = 0.01, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
    rabi_before = np.zeros(int((1/3)*len(times))+1)  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi_after = np.zeros(int((1/3)*len(times))) 
    rabi = []
    for i in range(0,len(rabi_vals)):
        rabi_during = np.ones(int(((1/3)*len(times))))*rabi_vals[i]
        rabi.append(np.concatenate((rabi_before, rabi_during, rabi_after), axis = None))

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]

        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_s_good_sq, times_s_good_sq, all_pulse_s_good_sq = pulse_shape_strong_good_square()

for rabi, pulse in zip(rabi_list_s_good_sq, all_pulse_s_good_sq):
    plt.plot(times_s_good_sq, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Good), Sqaure Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_good_gauss(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.1, kappa = 0.01, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    t0 = times[int(len(times)/2)]
    std = len(times)/10
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.exp(-((times[j]-t0)**2)/(2*std**2))
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]

        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_s_good_gauss, times_s_good_gauss, all_pulse_s_good_gauss = pulse_shape_strong_good_gauss()

for rabi, pulse in zip(rabi_list_s_good_gauss, all_pulse_s_good_gauss):
    plt.plot(times_s_good_gauss, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Good), Gaussian Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_strong_good_sin2(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=1.0, gamma=0.1, kappa = 0.01, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    T = times[-1]
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.sin(np.pi*times[j]/T)**2
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_s_good_sin2, times_s_good_sin2, all_pulse_s_good_sin2 = pulse_shape_strong_good_square()

for rabi, pulse in zip(rabi_list_s_good_sin2, all_pulse_s_good_sin2):
    plt.plot(times_s_good_sin2, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Strong (Bad), Sin^2 Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_bad_linear(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.05, kappa = 0.5, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = np.linspace(0, 5, len(times))  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    rho = [rho_init]
    pulse2 = [expect(a.dag()*a, rho_init)]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]

    for i_time in tqdm(range(1, len(times))):
        H = ( (
                detune_c * a.dag() * a
                + detune_l * sigma_se.dag() * sigma_se
                + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                + rabi[i_time] * (sigma_se + sigma_se.dag())
            ))
        solver = MESolver(H, c_ops = c_ops)
        solver.start(rho[i_time-1], times[i_time-1])
        rho_t = solver.step(times[i_time])
        # pulse1.append(rho_t.expect[0])
        pulse2.append(expect(a.dag()*a, rho_t))
        rho.append(rho_t)
        


    
    return rabi, times, pulse2
rabi_list_int_bad_lin, times_int_bad_lin, all_pulse_int_bad_lin = pulse_shape_inter_bad_linear()
plt.plot(times_int_bad_lin, all_pulse_int_bad_lin)
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Inter (Bad), Linear Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_bad_square(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.05, kappa = 0.5, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
    rabi_before = np.zeros(int((1/3)*len(times))+1)  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi_after = np.zeros(int((1/3)*len(times))) 
    rabi = []
    for i in range(0,len(rabi_vals)):
        rabi_during = np.ones(int(((1/3)*len(times))))*rabi_vals[i]
        rabi.append(np.concatenate((rabi_before, rabi_during, rabi_after), axis = None))

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_int_bad_sq, times_int_bad_sq, all_pulse_int_bad_sq = pulse_shape_inter_bad_square()

for rabi, pulse in zip(rabi_list_int_bad_sq, all_pulse_int_bad_sq):
    plt.plot(times_int_bad_sq, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Bad), Sqaure Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_bad_gauss(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.05, kappa = 0.5, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    t0 = times[int(len(times)/2)]
    std = len(times)/10
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.exp(-((times[j]-t0)**2)/(2*std**2))
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_int_bad_gauss, times_int_bad_gauss, all_pulse_int_bad_gauss = pulse_shape_inter_bad_gauss()

for rabi, pulse in zip(rabi_list_int_bad_gauss, all_pulse_int_bad_gauss):
    plt.plot(times_int_bad_gauss, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Bad), Gaussian Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_bad_sin2(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.05, kappa = 0.5, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    T = times[-1]
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.sin(np.pi*times[j]/T)**2
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_int_bad_sin2, times_int_bad_sin2, all_pulse_int_bad_sin2 = pulse_shape_inter_bad_sin2()

for rabi, pulse in zip(rabi_list_int_bad_sin2, all_pulse_int_bad_sin2):
    plt.plot(times_int_bad_sin2, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Bad), Sin^2 Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_good_linear(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.5, kappa = 0.05, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = np.linspace(0, 5, len(times))  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    rho = [rho_init]
    pulse2 = [expect(a.dag()*a, rho_init)]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]

    for i_time in tqdm(range(1, len(times))):
        H = ( (
                detune_c * a.dag() * a
                + detune_l * sigma_se.dag() * sigma_se
                + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                + rabi[i_time] * (sigma_se + sigma_se.dag())
            ))
        solver = MESolver(H, c_ops = c_ops)
        solver.start(rho[i_time-1], times[i_time-1])
        rho_t = solver.step(times[i_time])
        # pulse1.append(rho_t.expect[0])
        pulse2.append(expect(a.dag()*a, rho_t))
        rho.append(rho_t)
        


    
    return rabi, times, pulse2
rabi_list_int_good_lin, times_int_good_lin, all_pulse_int_good_lin = pulse_shape_inter_good_linear()
plt.plot(times_int_good_lin, all_pulse_int_good_lin)
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Good), Linear Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_good_square(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.5, kappa = 0.05, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05,0.15,0.3,0.5,1.0,2.5]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
    rabi_before = np.zeros(int((1/3)*len(times))+1)  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi_after = np.zeros(int((1/3)*len(times))) 
    rabi = []
    for i in range(0,len(rabi_vals)):
        rabi_during = np.ones(int(((1/3)*len(times))))*rabi_vals[i]
        rabi.append(np.concatenate((rabi_before, rabi_during, rabi_after), axis = None))

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in range(1, len(times)):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_int_good_sq, times_int_good_sq, all_pulse_int_good_sq = pulse_shape_inter_good_square()

for rabi, pulse in zip(rabi_list_int_good_sq, all_pulse_int_good_sq):
    plt.plot(times_int_good_sq, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Good), Sqaure Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_good_gauss(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.5, kappa = 0.05, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05,0.15,0.3,0.5,1.0,2.5]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    t0 = times[int(len(times)/2)]
    std = len(times)/10
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.exp(-((times[j]-t0)**2)/(2*std**2))
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_int_good_gauss, times_int_good_gauss, all_pulse_int_good_gauss = pulse_shape_inter_good_gauss()

for rabi, pulse in zip(rabi_list_int_good_gauss, all_pulse_int_good_gauss):
    plt.plot(times_int_good_gauss, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Good), Gaussian Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_inter_good_sin2(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.5, gamma=0.5, kappa = 0.05, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    ratios = [0.05,0.15,0.3,0.5,1.0,2.5]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*g_val)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    T = times[-1]
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.sin(np.pi*times[j]/T)**2
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_int_good_sin2, times_int_good_sin2, all_pulse_int_good_sin2 = pulse_shape_inter_good_square()

for rabi, pulse in zip(rabi_list_int_good_sin2, all_pulse_int_good_sin2):
    plt.plot(times_int_good_sin2, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Intermediate (Good), Sin^2 Rabi")
plt.legend()
plt.tight_layout()
plt.show()


def pulse_shape_purcell_linear(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.2, gamma=0.01, kappa = 2.0, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    rabi = np.linspace(0, 5, len(times))  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    rho = [rho_init]
    pulse2 = [expect(a.dag()*a, rho_init)]
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]

    for i_time in tqdm(range(1, len(times))):
        H = ( (
                detune_c * a.dag() * a
                + detune_l * sigma_se.dag() * sigma_se
                + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                + rabi[i_time] * (sigma_se + sigma_se.dag())
            ))
        solver = MESolver(H, c_ops = c_ops)
        solver.start(rho[i_time-1], times[i_time-1])
        rho_t = solver.step(times[i_time])
        # pulse1.append(rho_t.expect[0])
        pulse2.append(expect(a.dag()*a, rho_t))
        rho.append(rho_t)
        


    
    return rabi, times, pulse2
rabi_list_purr_lin, times_purr_lin, all_pulse_purr_lin = pulse_shape_purcell_linear()
plt.plot(times_purr_lin, all_pulse_purr_lin)
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Purcell, Linear Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_purcell_square(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.2, gamma=0.01, kappa = 2.0, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    ratios = [0.1,0.3,0.6,1.0,2.0,5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*kappa)
    rabi_before = np.zeros(int((1/3)*len(times))+1)  # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi_after = np.zeros(int((1/3)*len(times))) 
    rabi = []
    for i in range(0,len(rabi_vals)):
        rabi_during = np.ones(int(((1/3)*len(times))))*rabi_vals[i]
        rabi.append(np.concatenate((rabi_before, rabi_during, rabi_after), axis = None))

    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in range(1, len(times)):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_purcell_sq, times_purcell_sq, all_pulse_purcell_sq = pulse_shape_purcell_square()

for rabi, pulse in zip(rabi_list_purcell_sq, all_pulse_purcell_sq):
    plt.plot(times_purcell_sq, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Purcell, Sqaure Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_purcell_gauss(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.2, gamma=0.01, kappa = 2.0, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    ratios = [0.1,0.3,0.6,1.0,2.0,5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*kappa)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    t0 = times[int(len(times)/2)]
    std = len(times)/10
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.exp(-((times[j]-t0)**2)/(2*std**2))
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_purcell_gauss, times_purcell_gauss, all_pulse_purcell_gauss = pulse_shape_purcell_gauss()

for rabi, pulse in zip(rabi_list_purcell_gauss, all_pulse_purcell_gauss):
    plt.plot(times_purcell_gauss, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Purcell, Gaussian Rabi")
plt.legend()
plt.tight_layout()
plt.show()

def pulse_shape_purcell_sin2(Ncav=4, detune_c=0.0, detune_l=0.0, g_val=0.2, gamma=0.01, kappa = 2.0, init_cav = 0, init_sys = 2):
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
    times = np.linspace(0,300,1000)
    detune_c = detune_c  # Detuning between cavity resonance and atomic |g⟩↔|e⟩ transition frequency
    detune_l = detune_l  # Detuning between classical laser and atomic |s⟩↔|e⟩ transition frequency
    g_val = g_val  # Atom–cavity coupling strength (vacuum Rabi frequency) for |g⟩↔|e⟩ transition
    kappa = kappa # Cavity field decay rate (photon leakage through mirrors)
    gamma = gamma  # Spontaneous emission rate from |e⟩ to |g⟩ (atomic decay into free space)
    ratios = [0.1,0.3,0.6,1.0,2.0,5.0]
    rabi_vals = []
    for i in range(0,len(ratios)):
        rabi_vals.append(ratios[i]*kappa)
          # Rabi frequency of classical laser driving |s⟩↔|e⟩ transitional)
    rabi = []
    T = times[-1]
    for i in range(0,len(rabi_vals)):
        rabi_mod_2 = []
        for j in range(0,len(times)):
            rabi_mod = rabi_vals[i]*np.sin(np.pi*times[j]/T)**2
            rabi_mod_2.append(rabi_mod)
        rabi.append(rabi_mod_2)
        

    
    # Hamiltonian
    c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sigma_ge]
    rho_init = tensor(basis(Ncav, init_cav), basis(3, init_sys))
    
    # pulse1 = [kappa*expect(a.dag()*a, rho_init)]
    pulse_all = []
    for k in range(0, len(rabi)):
        rho = [rho_init]
        pulse2 = [expect(a.dag()*a, rho_init)]
        
        for i_time in tqdm(range(1, len(times))):
            H = ( (
                    detune_c * a.dag() * a
                    + detune_l * sigma_se.dag() * sigma_se
                    + g_val * (a.dag() * sigma_ge + a * sigma_ge.dag())
                    + rabi[k][i_time] * (sigma_se + sigma_se.dag())
                ))
            solver = MESolver(H, c_ops = c_ops)
            solver.start(rho[i_time-1], times[i_time-1])
            rho_t = solver.step(times[i_time])
            # pulse1.append(rho_t.expect[0])
            pulse2.append(expect(a.dag()*a, rho_t))
            rho.append(rho_t)
        pulse_all.append(pulse2)
            
        


    
    return rabi_vals, times, pulse_all
rabi_list_purcell_sin2, times_purcell_sin2, all_pulse_purcell_sin2 = pulse_shape_purcell_sin2()

for rabi, pulse in zip(rabi_list_purcell_sin2, all_pulse_purcell_sin2):
    plt.plot(times_purcell_sin2, pulse, label=f"Rabi ={rabi:g}")
plt.xlabel('Time')
plt.ylabel('Pulse Amp')
plt.title("Purcell, Sin^2 Rabi")
plt.legend()
plt.tight_layout()
plt.show()
