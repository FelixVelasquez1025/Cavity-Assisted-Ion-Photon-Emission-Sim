"""
Three-level atom with shelving (e -> s), with two modes:

1) Atom-only drive (recommended; mirrors the classic three_level_atom example)
   - Coherent Rabi drive on g<->e
   - Spontaneous emission from e->g and e->s (shelving/dark)

2) JC + shelving (optional)
   - Cavity exchanges excitations with the g<->e transition
   - Same decay channels from e

Default run uses (1), which is closest to the QuTiP three_level_atom examples.

Paper mapping (examples/rabi_hamiltonian.qmd):
- Implements the section "Hamiltonian for the Three Level System".
- H_0 in code corresponds to the rotating-frame detuning terms
  `Δ_driv σ_gg + Δ_rep σ_ss + (Δ_c - Δ_rep) a^† a`.
- H_drive corresponds to `(Ω_driv/2)(σ_ge + σ_eg)`.
- H_rep corresponds to `(Ω_repump/2)(σ_se + σ_es)`.
- H_cavity (optional) corresponds to `g (a^† σ_se + a σ_es)`.
- Collapse operators implement `√γ_eg σ_g e` and `√γ_es σ_s e` as described.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from scipy import constants as consts
from qutip import basis, destroy, mesolve, qeye, tensor

# Hbar in qutip is set to 1, using the real value throws off the dynamics of the system. For simplicity we take hbar = 1. 
hbar = 1


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
    # Paper mapping: This is the JC part of the interaction ("coupling" term)
    # for a three-level system when the cavity couples the g<->e branch.
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
    # Paper mapping: Lindblad terms for cavity (kappa, n_th) and atomic
    # spontaneous emission e->g (γ_eg) and e->s (γ_es / shelving).
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
    delta_c: float = 0.0,
    delta_rep: float = 0.0,
    drive_amp: float = 1.0,
    drive_t0: float = 5.0,
    drive_sigma: float = 1.0,
    rep_amp: float = 0.0,
    rep_t0: float = 5.0,
    rep_sigma: float = 1.0,
    gamma_eg: float = 0.05,
    gamma_es: float = 0.05,
    g_cav: float = 0.0,
    tmax: float = 40.0,
    steps: int = 400,
    start: str = "e",
    pump: bool = False,
):
    """
    Three-level atom (no cavity): g<->e coherently driven, e decays to g and s.

    Paper mapping (examples/rabi_hamiltonian.qmd): implements
    - H_0 (rotating frame detunings), H_drive, and H_rep terms from the
      three-level Hamiltonian section, and the corresponding collapse terms.
    """
    # Use operators mirroring the QuTiP three_level_atom helpers
    def three_level_basis():
        return basis(3, 0), basis(3, 1), basis(3, 2)

    def three_level_ops():
        one, two, three = three_level_basis()
        sig11 = one * one.dag()
        sig22 = two * two.dag()
        sig33 = three * three.dag()
        sig12 = one * two.dag()
        sig32 = three * two.dag()
        return np.array([sig11, sig22, sig33, sig12, sig32], dtype=object)

    sig11, sig22, sig33, sig12, sig32 = three_level_ops()

    # Build the Hamiltonian components as originally structured
    # Paper mapping: H_0, H_drive, H_rep, H_cavity correspond directly to the
    # three-level Hamiltonian components defined in the document.
    # Use a strict zero operator for the (optional) cavity placeholder so it
    # doesn't add spurious diagonal terms if g_cav or detunings are nonzero.
    from qutip import Qobj
    a = Qobj(np.zeros((3, 3)))
    H_0 = hbar * (
        delta_drive * sig11 + delta_rep * sig33 + (delta_c - delta_rep) * a.dag() * a
    )
    H_drive = 0.5 * hbar * (sig12 + sig12.dag())
    H_rep = 0.5 * hbar * (sig32 + sig32.dag())
    H_cavity = hbar * g_cav * (a * sig32.dag() + a.dag() * sig32)

    def omega_drive(t, args):
        amp = float(args.get("drive_amp", 0.0))
        t0 = float(args.get("drive_t0", 0.0))
        sigma = float(args.get("drive_sigma", 0.0))
        if sigma <= 0.0:
            return 0.0
        return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    def omega_rep(t, args):
        amp = float(args.get("rep_amp", 0.0))
        t0 = float(args.get("rep_t0", 0.0))
        sigma = float(args.get("rep_sigma", 0.0))
        if sigma <= 0.0:
            return 0.0
        return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

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
        c_ops.append(np.sqrt(gamma_eg) * sig12)  # e->g
    if gamma_es > 0:
        c_ops.append(np.sqrt(gamma_es) * sig32)  # e->s 

    tlist = np.linspace(0.0, tmax, steps)
    args = {
        "drive_amp": drive_amp,
        "drive_t0": drive_t0,
        "drive_sigma": drive_sigma,
        "rep_amp": rep_amp,
        "rep_t0": rep_t0,
        "rep_sigma": rep_sigma,
    }
    if pump is False:
        H = [H_0 + H_cavity, [H_drive, omega_drive]]
    else:
        H = [H_0 + H_cavity, [H_drive, omega_drive], [H_rep, omega_rep]]
    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[sig11, sig22, sig33], args=args)
    Pg = np.array(result.expect[0])
    Pe = np.array(result.expect[1])
    Ps = np.array(result.expect[2])
    return tlist, Pg, Pe, Ps


def main():
    t, Pg, Pe, Ps = three_level_atom_driven(
        delta_drive=1.0,
        delta_rep=1.0,
        drive_amp=1.0,
        drive_t0=10.0,
        drive_sigma=2.0,
        rep_amp=1.0,
        rep_t0=10.0,
        rep_sigma=2.0,
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
                drive_amp=1.0,
                drive_t0=10.0,
                drive_sigma=2.0,
                rep_amp=0.0,
                rep_t0=10.0,
                rep_sigma=2.0,
                gamma_eg=0.05,
                gamma_es=0.05,
                tmax=100.0,
                steps=400,
                start="e",
                pump=False,
            )

            t, Pg, Pe, Ps = three_level_atom_driven(**params)

            fig, ax = plt.subplots(figsize=(9, 6))
            plt.subplots_adjust(left=0.1, right=0.95, bottom=0.58)
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
            ax_dd = plt.axes([0.10, 0.48, 0.75, 0.03], facecolor=axcolor)
            ax_dr = plt.axes([0.10, 0.44, 0.75, 0.03], facecolor=axcolor)
            ax_dt0 = plt.axes([0.10, 0.40, 0.75, 0.03], facecolor=axcolor)
            ax_dsig = plt.axes([0.10, 0.36, 0.75, 0.03], facecolor=axcolor)
            ax_rp = plt.axes([0.10, 0.32, 0.75, 0.03], facecolor=axcolor)
            ax_rt0 = plt.axes([0.10, 0.28, 0.75, 0.03], facecolor=axcolor)
            ax_rsig = plt.axes([0.10, 0.24, 0.75, 0.03], facecolor=axcolor)
            ax_geg = plt.axes([0.10, 0.20, 0.75, 0.03], facecolor=axcolor)
            ax_ges = plt.axes([0.10, 0.16, 0.75, 0.03], facecolor=axcolor)
            ax_drep = plt.axes([0.10, 0.12, 0.75, 0.03], facecolor=axcolor)

            s_dd = Slider(ax_dd, 'Δ_ge', -3.0, 3.0, valinit=params['delta_drive'])
            s_dr = Slider(ax_dr, 'Ω_ge (amp)', 0.0, 5.0, valinit=params['drive_amp'])
            s_rp = Slider(ax_rp, 'Ω_rep (amp)', 0.0, 5.0, valinit=params['rep_amp'])
            s_dt0 = Slider(ax_dt0, 't0 drive', 0.0, params['tmax'], valinit=params['drive_t0'])
            s_dsig = Slider(ax_dsig, 'σ drive', 0.1, params['tmax'], valinit=params['drive_sigma'])
            s_rt0 = Slider(ax_rt0, 't0 rep', 0.0, params['tmax'], valinit=params['rep_t0'])
            s_rsig = Slider(ax_rsig, 'σ rep', 0.1, params['tmax'], valinit=params['rep_sigma'])
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
                    drive_amp=s_dr.val,
                    rep_amp=s_rp.val,
                    drive_t0=s_dt0.val,
                    drive_sigma=s_dsig.val,
                    rep_t0=s_rt0.val,
                    rep_sigma=s_rsig.val,
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

            for sl in (s_dd, s_dr, s_rp, s_dt0, s_dsig, s_rt0, s_rsig, s_geg, s_ges, s_drep):
                sl.on_changed(lambda _val: recompute())
            radio.on_clicked(lambda _label: recompute())
            chk.on_clicked(lambda _label: recompute())
            btn_reset.on_clicked(lambda _evt: (
                s_dd.reset(), s_dr.reset(), s_rp.reset(), s_dt0.reset(), s_dsig.reset(), s_rt0.reset(), s_rsig.reset(), s_geg.reset(), s_ges.reset(), s_drep.reset(),
                radio.set_active(0),
                chk.set_active(0) if chk.get_status()[0] else None,
                recompute()
            ))

            plt.show()

        interactive_three_level()
    else:
        main()
