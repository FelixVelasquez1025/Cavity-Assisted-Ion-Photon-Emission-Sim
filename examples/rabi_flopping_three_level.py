"""
Three-level atom with shelving (e -> s), with two modes:

1) Atom-only drive (recommended; mirrors the classic three_level_atom example)
   - Coherent Rabi drive on g<->e
   - Spontaneous emission from e->g and e->s (shelving/dark)

2) JC + shelving (optional)
   - Cavity exchanges excitations with the g<->e transition
   - Same decay channels from e

Default run uses (1), which is closest to the QuTiP three_level_atom examples.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from qutip import basis, destroy, mesolve, qeye, tensor


def three_level_rabi(
    N: int = 10,
    g_cav: float = 0.05 * 2 * np.pi,
    kappa: float = 0.0,
    n_th: float = 0.0,
    gamma_eg: float = 0.0,
    gamma_es: float = 0.05,  # shelving to |s>
    tmax: float = 40.0,
    steps: int = 400,
    start: str = "e",  # e or g or photon
    pump: bool = False,
):
    # Hilbert spaces
    a = tensor(destroy(N), qeye(3))

    # Atomic projectors and transition operators
    g = basis(3, 0)
    e = basis(3, 1)
    s = basis(3, 2)
    P_g = tensor(qeye(N), g * g.dag())
    P_e = tensor(qeye(N), e * e.dag())
    P_s = tensor(qeye(N), s * s.dag())

    # Transitions: |g><e| and |s><e|
    sigma_ge = tensor(qeye(N), g * e.dag())  # e -> g
    sigma_eg = sigma_ge.dag()
    sigma_se = tensor(qeye(N), s * e.dag())  # e -> s (shelving)

    # Rotating-frame resonant JC: only coupling term (set detunings to zero)
    H = g_cav * (a * sigma_eg + a.dag() * sigma_ge)

    # Initial state
    if start == "e":
        psi0 = tensor(basis(N, 0), e)
    elif start == "g":
        psi0 = tensor(basis(N, 0), g)
    elif start == "photon":
        psi0 = tensor(basis(N, 1), g)
    else:
        raise ValueError("start must be 'e', 'g', or 'photon'")

    # Collapse operators
    c_ops = []
    
    # Cavity thermal bath
    rate = kappa * (1 + n_th)
    if rate > 0:
        c_ops.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    if rate > 0:
        c_ops.append(np.sqrt(rate) * a.dag())
    # Atomic decays
    if gamma_eg > 0:
        c_ops.append(np.sqrt(gamma_eg) * sigma_ge)  # e->g
    if gamma_es > 0:
        c_ops.append(np.sqrt(gamma_es) * sigma_se)  # e->s (shelving)

    tlist = np.linspace(0.0, tmax, steps)

    result = mesolve(
        H,
        psi0,
        tlist,
        c_ops=c_ops,
        e_ops=[a.dag() * a, P_g, P_e, P_s],
    )

    nbar = np.array(result.expect[0])
    Pg = np.array(result.expect[1])
    Pe = np.array(result.expect[2])
    Ps = np.array(result.expect[3])
    return tlist, nbar, Pg, Pe, Ps


def three_level_atom_driven(

    delta_drive: float = 0.0,
    delta_rep: float = 0.0,
    rabi_drive: float = 1.0,
    rabi_repump: float = 0.0,
    gamma_eg: float = 0.05,
    gamma_es: float = 0.05,
    tmax: float = 40.0,
    steps: int = 400,
    start: str = "e",
    pump: bool = False,
):
    """
    Three-level atom (no cavity): g<->e coherently driven, e decays to g and s.

    Hamiltonian (rotating frame): H = Delta_ge |e><e| + (Omega_ge/2)(|e><g| + |g><e|)
    Collapse: sqrt(gamma_eg) |g><e|, sqrt(gamma_es) |s><e|
    """
    # Use operators mirroring the QuTiP three_level_atom helpers
    def three_level_basis():
        return basis(3, 0), basis(3, 1), basis(3, 2)

    def three_level_ops():
        g, e, s = three_level_basis()
        sig_gg = g * g.dag()
        sig_ee = e * e.dag()
        sig_ss = s * s.dag()
        sig_ge = g * e.dag()
        sig_se = s * e.dag()
        return np.array([sig_gg, sig_ee, sig_ss, sig_ge, sig_se], dtype=object)

    sig_gg, sig_ee, sig_ss, sig_ge, sig_se = three_level_ops()

    # Atom-only Hamiltonian in rotating frame (no cavity here)
    # H = Delta_ge |e><e| + Delta_se |s><s| + (Omega_ge/2)(|e><g| + |g><e|) [+ repump term]
    H_0 = delta_drive * sig_ee + delta_rep * sig_ss
    H_drive = 0.5 * rabi_drive * (sig_ge + sig_ge.dag())
    H_rep = 0.5 * rabi_repump * (sig_se + sig_se.dag())

    H = H_0 + H_drive + (H_rep if pump else 0)

    # Initial state
    if start == "e":
        psi0 = basis(3, 1)
    elif start == "g":
        psi0 = basis(3, 0)
    else:
        raise ValueError("start must be 'e' or 'g'")

    # Collapse operators
    c_ops = []
    if gamma_eg > 0:
        c_ops.append(np.sqrt(gamma_eg) * sig_ge)  # e->g
    if gamma_es > 0:
        c_ops.append(np.sqrt(gamma_es) * sig_se)  # e->s 

    tlist = np.linspace(0.0, tmax, steps)
    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[sig_gg, sig_ee, sig_ss])
    Pg = np.array(result.expect[0])
    Pe = np.array(result.expect[1])
    Ps = np.array(result.expect[2])
    return tlist, Pg, Pe, Ps


def main():
    # Atom-only driven case 
    t, Pg, Pe, Ps = three_level_atom_driven(
        delta_drive=1.0,
        delta_rep=1.0,
        rabi_drive=1.0,
        rabi_repump=1.0,
        gamma_eg=0.05,
        gamma_es=0.05,
        tmax=100.0,
        steps=400,
        start="e",
        pump=True,
    )
        


    plt.figure(figsize=(8, 5))
    plt.plot(t, Pe, label="|e> population")
    plt.plot(t, Pg, label="|g> population")
    plt.plot(t, Ps, label="|s> population (dark)")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Driven three-level atom with e→s")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.title("Driven three-level atom with shelving")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Three-level Rabi flopping example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive sliders to tweak parameters",
    )
    parser.set_defaults(interactive=True)
    args = parser.parse_args()

    if args.interactive:
        # Lazily define and run an interactive UI using Matplotlib widgets.
        def interactive_three_level():
            # Defaults matching main()
            params = dict(
                delta_drive=0.0,
                delta_rep=0.0,
                rabi_drive=1.0,
                rabi_repump=0.0,
                gamma_eg=0.05,
                gamma_es=0.05,
                tmax=100.0,
                steps=400,
                start="e",
                pump=False,
            )

            t, Pg, Pe, Ps = three_level_atom_driven(**params)

            fig, ax = plt.subplots(figsize=(9, 6))
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.40)
            ln_e, = ax.plot(t, Pe, label="|e> population")
            ln_g, = ax.plot(t, Pg, label="|g> population")
            ln_s, = ax.plot(t, Ps, label="|s> population (dark)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            ax.set_ylim(0, 1.05)
            ax.set_title("Driven three-level atom with shelving (interactive)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # Slider axes
            axcolor = 'lightgoldenrodyellow'
            ax_dd = plt.axes([0.10, 0.32, 0.75, 0.03], facecolor=axcolor)
            ax_dr = plt.axes([0.10, 0.28, 0.75, 0.03], facecolor=axcolor)
            ax_rp = plt.axes([0.10, 0.24, 0.75, 0.03], facecolor=axcolor)
            ax_geg = plt.axes([0.10, 0.20, 0.75, 0.03], facecolor=axcolor)
            ax_ges = plt.axes([0.10, 0.16, 0.75, 0.03], facecolor=axcolor)
            ax_drep = plt.axes([0.10, 0.12, 0.75, 0.03], facecolor=axcolor)

            s_dd = Slider(ax_dd, 'Δ_ge', -3.0, 3.0, valinit=params['delta_drive'])
            s_dr = Slider(ax_dr, 'Ω_ge', 0.0, 5.0, valinit=params['rabi_drive'])
            s_rp = Slider(ax_rp, 'Ω_rep', 0.0, 5.0, valinit=params['rabi_repump'])
            s_geg = Slider(ax_geg, 'γ_eg', 0.0, 1.0, valinit=params['gamma_eg'])
            s_ges = Slider(ax_ges, 'γ_es', 0.0, 1.0, valinit=params['gamma_es'])
            s_drep = Slider(ax_drep, 'Δ_se', -3.0, 3.0, valinit=params['delta_rep'])

            # Radio buttons for start state and a checkbox for repump enable
            ax_radio = plt.axes([0.10, 0.02, 0.18, 0.08], facecolor=axcolor)
            radio = RadioButtons(ax_radio, ('e', 'g'), active=0)

            ax_chk = plt.axes([0.32, 0.02, 0.18, 0.08], facecolor=axcolor)
            chk = CheckButtons(ax_chk, ['Repump'], [params['pump']])

            # Reset button
            ax_reset = plt.axes([0.54, 0.02, 0.12, 0.06])
            btn_reset = Button(ax_reset, 'Reset')

            def recompute():
                params.update(
                    delta_drive=s_dd.val,
                    delta_rep=s_drep.val,
                    rabi_drive=s_dr.val,
                    rabi_repump=s_rp.val,
                    gamma_eg=s_geg.val,
                    gamma_es=s_ges.val,
                )
                # pump checkbox
                params['pump'] = chk.get_status()[0]
                # start radio
                params['start'] = radio.value_selected
                t2, Pg2, Pe2, Ps2 = three_level_atom_driven(**params)
                ln_e.set_ydata(Pe2)
                ln_g.set_ydata(Pg2)
                ln_s.set_ydata(Ps2)
                fig.canvas.draw_idle()

            for sl in (s_dd, s_dr, s_rp, s_geg, s_ges, s_drep):
                sl.on_changed(lambda _val: recompute())
            radio.on_clicked(lambda _label: recompute())
            chk.on_clicked(lambda _label: recompute())
            btn_reset.on_clicked(lambda _evt: (
                s_dd.reset(), s_dr.reset(), s_rp.reset(), s_geg.reset(), s_ges.reset(), s_drep.reset(),
                radio.set_active(0),
                chk.set_active(0) if chk.get_status()[0] else None,
                recompute()
            ))

            plt.show()

        interactive_three_level()
    else:
        main()
