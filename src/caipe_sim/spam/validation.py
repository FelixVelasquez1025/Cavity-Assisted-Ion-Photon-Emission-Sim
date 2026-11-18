"""Reproducible Stage 1a checks, not the Stage 1b experimental benchmarks.

All Rabi magnitudes below are illustrative test inputs, not atomic constants
or fitted experimental values. See docs/spam-physics-notes.md for scope.
"""
import numpy as np
from qutip import Qobj, liouvillian, to_choi
from scipy.linalg import expm
from scipy.signal import find_peaks

from . import atomic as a
from .model import Beam, Tone, build_model, fluorescence_369_s


def validation_beams():
    # 55 degrees and 5 G protocol context: Egan 2021 §2.2.2.
    # Reduced Rabi frequencies 20/10 MHz are arbitrary validation controls.
    theta = np.deg2rad(55)
    return (Beam('369', 2*np.pi*20e6, theta,
                 sidebands=(Tone(14.748e9, 2*np.pi*10e6),)),
            Beam('935', 2*np.pi*20e6, theta,
                 sidebands=(Tone(-3.07e9, 2*np.pi*10e6),)))


def weak_probe_spectrum(beam, ground_F, detunings_hz):
    """Initial weak-probe fluorescence susceptibility in photons/s.

    Ground populations are fixed, incoherent and uniform within S F.
    Solve optical coherences to leading order Omega²: rate =
    Gamma_PS*|H_ul|² / ((Gamma_P/2)²+delta_ul²), summed with ground weights.
    This is a weak-probe spectrum BEFORE optical pumping, not the eventual
    steady-state fluorescence of an ion pumped into a dark superposition.
    It uses the actual single-tone Hamiltonian's couplings and detunings.
    """
    if beam.line != '369' or beam.sidebands or beam.linewidth_dephasing_s:
        raise ValueError("Probe must be a single 369 nm tone with no added dephasing")
    if ground_F not in (0, 1):
        raise ValueError("S ground_F must be 0 or 1")
    model = build_model([Beam('369', beam.rabi_rad_s, beam.theta_rad)])
    h = model.hamiltonian.full()
    rates = np.zeros_like(np.asarray(detunings_hz, dtype=float))
    gamma = 1/a.MANIFOLDS['P'].lifetime_s
    for l in [s for s in a.STATES if s[0:2] == ('S', ground_F)]:
        for u in [s for s in a.STATES if s[0] == 'P']:
            il, iu = a.INDEX[l], a.INDEX[u]
            delta = (h[iu,iu]-h[il,il]).real-2*np.pi*np.asarray(detunings_hz)
            rates += a.BRANCHES['P']['S']*gamma*abs(h[iu,il])**2 / ((gamma/2)**2+delta**2)/(2*ground_F+1)
    return rates


def hyperfine_probe_check():
    # Coarse full scan plus 0.1 MHz sampling around each expected feature.
    targets = [0.0, a.MANIFOLDS['P'].splitting_hz,
               a.MANIFOLDS['S'].splitting_hz+a.MANIFOLDS['P'].splitting_hz]
    grid = np.unique(np.concatenate([np.linspace(-40e6, 14.8e9, 1201),
           *[np.linspace(t-30e6,t+30e6,601) for t in targets]]))
    beam = Beam('369',2*np.pi*50e3,np.deg2rad(55))
    s1 = weak_probe_spectrum(beam,1,grid)
    s0 = weak_probe_spectrum(beam,0,grid)
    peaks = find_peaks(s1)[0]
    # Two highest peaks, sorted in detuning; S0 has one allowed F line.
    s1_peaks = np.sort(grid[peaks[np.argsort(s1[peaks])[-2:]]])
    s0_peak = float(grid[np.argmax(s0)])
    return {'S_F1_peak_detunings_hz': s1_peaks.tolist(),
            'P_peak_separation_hz': float(np.diff(s1_peaks)[0]),
            'S_F0_peak_detuning_hz': s0_peak,
            'fine_grid_spacing_hz': 1e5,
            'interpretation': 'Leading-order fixed-ground-population susceptibility; not an optically pumped steady state'}


def finite_weak_probe_check():
    """Corroborate the susceptibility peaks with the full static master equation.

    A 100 ns weak pulse lets optical coherences settle while scattering
    probabilities remain tiny; all off-resonant couplings and decay are kept.
    """
    records = []
    for F, center in [(1,0), (1,2.105e9), (0,14.747812118466e9)]:
        rho0 = sum(a.ket('S',F,m).proj() for m in range(-F,F+1))/(2*F+1)
        populations = []
        for offset in (-80e6,0,80e6):
            model = build_model([Beam('369',2*np.pi*50e3,np.deg2rad(55),detuning_hz=center+offset)])
            rho = model.evolve(rho0,[100e-9])[-1]
            populations.append(float(sum(rho[a.INDEX[s],a.INDEX[s]].real for s in a.STATES if s[0]=='P')))
        records.append({'S_F': F, 'center_hz': center,
                        'detuning_offsets_hz': [-80e6,0,80e6],
                        'P_population_after_100ns': populations})
    return records


def two_tone_comparison(*, step_divisor=50):
    beam = Beam('369',2*np.pi*5e6,.7,
                sidebands=(Tone(14.748e9,2*np.pi*5e6),))
    model = build_model([beam],B_gauss=5)
    times = np.linspace(0,30e-9,121)
    initial = (a.ket('S',0,0)+a.ket('S',1,0)).unit()
    secular = model.evolve(initial,times)
    full = model.full_evolution(initial,times,max_step_s=1/(step_divisor*14.748e9))
    return {'window_s': float(times[-1]), 'samples': len(times),
            'max_population_difference': float(max(np.max(abs(x.diag()-y.diag())) for x,y in zip(secular,full))),
            'max_density_frobenius_difference': float(max(np.linalg.norm(x.full()-y.full()) for x,y in zip(secular,full))),
            'max_full_reference_step_s': 1/(step_divisor*14.748e9)}


def dark_state_check():
    rates, residuals, mins = {}, {}, {}
    for B in (0,5):
        model = build_model(validation_beams(),B_gauss=B)
        rho = model.stationary_state()
        rates[str(B)] = fluorescence_369_s(rho)
        residuals[str(B)] = float(np.linalg.norm(liouvillian(model.hamiltonian,model.collapses).full()@rho.full().reshape(-1,order='F')))
        mins[str(B)] = float(np.linalg.eigvalsh(rho.full()).min())
    return {'B_gauss': [0,5], 'fluorescence_369_per_s': rates,
            'zero_to_five_ratio': rates['0']/rates['5'],
            'stationary_residual_norm_per_s': residuals,
            'minimum_density_eigenvalues': mins}


def long_run_check():
    model = build_model(validation_beams(), B_gauss=5)
    initial = (a.ket('S',0,0)+1j*a.ket('S',1,1)).unit()
    times = [0,1e-7,1e-5,1e-3]
    states = model.evolve(initial,times)
    generator = liouvillian(model.hamiltonian,model.collapses)
    channel = Qobj(expm(generator.full()*20e-9),dims=generator.dims,superrep='super')
    choi = to_choi(channel).full()
    return {'times_s': times,
            'maximum_trace_error': float(max(abs(r.tr()-1) for r in states)),
            'maximum_hermiticity_error': float(max(np.max(abs(r.full()-r.full().conj().T)) for r in states)),
            'minimum_density_eigenvalue': float(min(np.linalg.eigvalsh(r.full()).min() for r in states)),
            'choi_check_time_s': 20e-9,
            'minimum_choi_eigenvalue': float(np.linalg.eigvalsh((choi+choi.conj().T)/2).min()),
            'choi_hermiticity_error': float(np.max(abs(choi-choi.conj().T)))}


def stage1a_report():
    normalization = {}
    for u in a.STATES:
        if u[0] not in a.BRANCHES:
            continue
        actual = sum(rate for upper,_,rate in a.decay_channels() if upper == u)
        normalization[str(u)] = abs(actual*a.MANIFOLDS[u[0]].lifetime_s-1)
    slopes = {f'{level},F={F}': a.g_f(level,F)*a.MU_B_OVER_HBAR*1e-4/(2*np.pi*1e6)
              for level,m in a.MANIFOLDS.items() for F in m.Fs if F}
    return {'scope': 'Stage 1a atomic checks only; no Stage 1b benchmarks',
            'unverified': list(a.UNVERIFIED),
            'max_relative_decay_normalization_error': max(normalization.values()),
            'zeeman_mhz_per_gauss_per_m': slopes,
            'weak_probe_spectrum': hyperfine_probe_check(),
            'finite_weak_probe': finite_weak_probe_check(),
            'two_tone': two_tone_comparison(),
            'two_tone_half_reference_step': two_tone_comparison(step_divisor=100),
            'long_run': long_run_check(),
            'dark_state': dark_state_check()}
