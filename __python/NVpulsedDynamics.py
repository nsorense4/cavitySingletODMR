# -*- coding: utf-8 -*-
"""
NV center 11-level model: time dynamics under pulsed green pumping.

Starting point for pulsed-protocol optimization. Solves the Lindblad master
equation (same Liouvillian as nv_steady_state in
linbladMESolverFunctionChargeConversionLiouvillian.py) for a square green
pulse of intensity I_p and duration tPulse, followed by a dark period tDark.
The IR probe (I_s) and MW drive (OmegaR) are held constant throughout.

Because the pulse is piecewise constant, each segment is evolved exactly with
the propagator expm(L*dt) acting on the vectorized density matrix, so there
are no ODE-solver issues with the ~1 ns singlet lifetime or the pulse edges.

@author: nicho
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.linalg import expm
from functools import partial
import os
import time
import optuna

from linbladMESolverFunctionChargeConversionLiouvillian import (
    nv_steady_state, DIM, BASIS,
    g0, gp, gm, e0, ep, em, s1, s2, gNV0, mNV0, eNV0,
)
from commonlabcode.plot_custom import plot_custom
from nvConstants import *

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIGURES_DIR = os.path.join(_THIS_DIR, "..", "__figures")

# Same fixed constants as nv_steady_state_default in NVpopulationDynamics.py,
# but returning the Liouvillian instead of solving it.
nv_liouvillian = partial(
    nv_steady_state,
    gamma_eg=gamma_eg,
    gamma_isc_0=gamma_isc_0,
    gamma_isc_1=gamma_isc_1,
    gamma_isc_g0=gamma_isc_g0,
    gamma_isc_g1=gamma_isc_g1,
    gamma_s=gamma_s,
    gamma_NV0_eg=gamma_NV0_eg,
    gamma_NV0_em=gamma_NV0_em,
    gamma_NV0_mg=gamma_NV0_mg,
    sigma_p=sigma_p,
    sigma_s=sigma_s,
    lamb_s=lamb_s,
    lamb_p=lamb_p,
    c=c_const,
    h=h,
    gamma_phi=gamma_phi,
    paramsRates=paramsRatesOnes,
    paramsRatesReal=paramsRatesReal,
    returnLiouvillian=True,
)

# Vectorized-density-matrix index of each population rho_ii (qutip stacks
# columns, so rho_ij -> i + j*DIM).
_POP_IDX = np.arange(DIM) * (DIM + 1)

STATE_LABELS = [r'$|0\rangle$', r'$|1\rangle$', r'$|2\rangle$', r'$|3\rangle$',
                r'$|4\rangle$', r'$|5\rangle$', r'$|6\rangle$', r'$|7\rangle$',
                r'$|8\rangle$', r'$|9\rangle$', r'$|10\rangle$']
STATE_COLORS = ['#3dfe00ff', '#35db00ff', '#218800ff',
                '#fe7900ff', '#fe0000ff', '#fe0076ff',
                '#190cffff', '#800000ff', '#800cffff', '#ff0cf4ff', '#feb400ff']


def initialState(kind='mixedGround'):
    """Initial density matrix (vectorized).

    'mixedGround' : equal mixture of the NV- ground triplet (thermal, no prior pumping)
    'g0'          : fully polarized into NV- ms=0
    """
    if kind == 'mixedGround':
        rho0 = sum(BASIS[i] * BASIS[i].dag() for i in [g0, gp, gm]) / 3
    elif kind == 'g0':
        rho0 = BASIS[g0] * BASIS[g0].dag()
    elif kind == 'mixedGroundNonZero':
        rho0 = sum(BASIS[i] * BASIS[i].dag() for i in [gp, gm]) / 2
    else:
        raise ValueError(f"unknown initial state '{kind}'")
    return qt.operator_to_vector(rho0).full().ravel()


def evolveSegment(L, rhoVec0, tSeg, dt):
    """Evolve rhoVec0 under constant Liouvillian L for tSeg, sampled every dt.

    Returns (t, rhoVecs) with t starting at 0 and rhoVecs[k] the state at t[k].
    """
    nSteps = max(int(round(tSeg/dt)), 1)
    dtSeg = tSeg/nSteps
    U = expm(L.full()*dtSeg)
    rhoVecs = np.empty((nSteps + 1, rhoVec0.size), dtype=complex)
    rhoVecs[0] = rhoVec0
    for k in range(nSteps):
        rhoVecs[k + 1] = U @ rhoVecs[k]
    return np.arange(nSteps + 1)*dtSeg, rhoVecs


def solvePulsedPopulation(I_p, tPulse, tDark,
                          I_s=0,
                          OmegaR=0,
                          detuning=detuning,
                          zeeman=zeeman,
                          rho0='mixedGround',
                          dt=1e-9):
    """Populations for a square green pulse of intensity I_p [W/m^2] and
    duration tPulse [s], followed by tDark [s] with the green off.

    Returns
    -------
    t : (N,) array of times [s], t=0 at the pulse turn-on
    pops : (N, 11) array of populations p_i(t)
    """
    common = dict(Omega=OmegaR, I_s=I_s, detuning=detuning, zeeman=zeeman)
    L_on = nv_liouvillian(I_p=I_p, **common)
    L_off = nv_liouvillian(I_p=0, **common)

    rhoVec0 = initialState(rho0) if isinstance(rho0, str) else rho0

    tOn, rhoOn = evolveSegment(L_on, rhoVec0, tPulse, dt)
    tList, rhoList = [tOn], [rhoOn]
    if tDark > 0:
        tOff, rhoOff = evolveSegment(L_off, rhoOn[-1], tDark, dt)
        # drop the duplicated boundary point
        tList.append(tOff[1:] + tPulse)
        rhoList.append(rhoOff[1:])

    t = np.concatenate(tList)
    pops = np.real(np.concatenate(rhoList)[:, _POP_IDX])
    return t, pops


def derivedSignals(pops):
    """Experimentally relevant combinations of the populations.

    PL rates are emitted photons per NV per second (no collection efficiency).
    """
    return {
        'NVminus': pops[:, :8].sum(axis=1),
        'NV0': pops[:, 8:].sum(axis=1),
        'PL_NVminus': gamma_eg*(pops[:, e0] + pops[:, ep] + pops[:, em]),
        'PL_NV0': gamma_NV0_eg*pops[:, eNV0],
        'singlet_s2': pops[:, s2],  # IR-absorbing metastable singlet
        'ms0': pops[:, g0]/pops[:, [g0, gp, gm]].sum(axis=1),  # spin polarization within NV- ground
    }


def _windowActive(t, tOn, tOff):
    """Whether a pulse window [tOn, tOff) is on at time t (zero/negative duration -> never on)."""
    return (tOff > tOn) and (tOn <= t < tOff)


def solvePulsedSequence(I_p, OmegaR, tGreen1On, tGreen1Off, tMWOn, tMWOff,
                         tGreen2On, tGreen2Off,
                         tIROn=None, tIROff=None,
                         I_s=0,
                         detuning=detuning,
                         zeeman=zeeman,
                         rho0='mixedGround',
                         dt=1e-9,
                         tEnd=None):
    """Populations for an arbitrary green1 / MW / green2 / IR pulse sequence.

    Green intensity is I_p whenever t falls in [tGreen1On, tGreen1Off) or
    [tGreen2On, tGreen2Off), and the MW Rabi rate is OmegaR whenever t falls
    in [tMWOn, tMWOff); both are 0 otherwise. If tIROn/tIROff are given, the
    IR probe I_s is likewise pulsed on only within [tIROn, tIROff) -- e.g. a
    readout IR pulse immediately after green2, via tIROn=tGreen2Off -- and 0
    outside it; if either is omitted, I_s is held constant for the whole
    sequence instead (the original behaviour). The timeline runs from t=0
    to tEnd (default: the latest of all the pulse-off times in use).

    Returns
    -------
    t : (N,) array of times [s], t=0 at the start of the sequence
    pops : (N, 11) array of populations p_i(t)
    """
    irPulsed = tIROn is not None and tIROff is not None

    if tEnd is None:
        tEnd = max(tGreen1Off, tMWOff, tGreen2Off, tIROff if irPulsed else 0.0)

    bpSet = {0.0, tGreen1On, tGreen1Off, tMWOn, tMWOff, tGreen2On, tGreen2Off, tEnd}
    if irPulsed:
        bpSet |= {tIROn, tIROff}
    breakpoints = sorted(t for t in bpSet if 0.0 <= t <= tEnd)

    common = dict(detuning=detuning, zeeman=zeeman)
    Lcache = {}

    def getL(I_p_seg, Omega_seg, I_s_seg):
        key = (I_p_seg, Omega_seg, I_s_seg)
        if key not in Lcache:
            Lcache[key] = nv_liouvillian(I_p=I_p_seg, Omega=Omega_seg, I_s=I_s_seg, **common)
        return Lcache[key]

    rhoVec = initialState(rho0) if isinstance(rho0, str) else rho0
    tList, rhoList = [], []
    for tA, tB in zip(breakpoints[:-1], breakpoints[1:]):
        tSeg = tB - tA
        if tSeg <= 0:
            continue
        tMid = 0.5*(tA + tB)
        greenOn = (_windowActive(tMid, tGreen1On, tGreen1Off)
                   or _windowActive(tMid, tGreen2On, tGreen2Off))
        mwOn = _windowActive(tMid, tMWOn, tMWOff)
        I_s_seg = (I_s if _windowActive(tMid, tIROn, tIROff) else 0.0) if irPulsed else I_s
        L = getL(I_p if greenOn else 0.0, OmegaR if mwOn else 0.0, I_s_seg)

        tSegArr, rhoSegArr = evolveSegment(L, rhoVec, tSeg, dt)
        if tList:
            # drop the duplicated boundary point shared with the previous segment
            tList.append(tSegArr[1:] + tA)
            rhoList.append(rhoSegArr[1:])
        else:
            tList.append(tSegArr + tA)
            rhoList.append(rhoSegArr)
        rhoVec = rhoSegArr[-1]

    t = np.concatenate(tList)
    pops = np.real(np.concatenate(rhoList)[:, _POP_IDX])
    return t, pops


def _integratedSignal(t, pops, key, tWinOn, tWinOff):
    """derivedSignals(pops)[key] integrated (trapezoidal) over [tWinOn, tWinOff];
    falls back to the nearest sample if the window is narrower than the
    sampling grid."""
    sig = derivedSignals(pops)[key]
    mask = (t >= tWinOn) & (t <= tWinOff)
    if mask.sum() < 2:
        idx = min(np.searchsorted(t, tWinOn), len(sig) - 1)
        return sig[idx]
    return np.trapz(sig[mask], t[mask])


def _steadyStateSignal(I_p, OmegaR, I_s, detuning, zeeman, key):
    """derivedSignals(...)[key] for the CW steady state at the given I_p/OmegaR
    (used to compare a pulsed trace against the continuous-illumination limit)."""
    popSS = np.real(np.array(nv_liouvillian(
        I_p=I_p, Omega=OmegaR, I_s=I_s, detuning=detuning, zeeman=zeeman,
        returnLiouvillian=False, steadystatebool=True,
    )))
    return derivedSignals(popSS[None, :])[key][0]


def pulsedContrast(I_p, OmegaR, tGreen1On, tGreen1Off, tMWOn, tMWOff,
                    tGreen2On, tGreen2Off,
                    tIROn=None, tIROff=None,
                    I_s=0,
                    detuning=detuning,
                    zeeman=zeeman,
                    rho0='mixedGround',
                    dt=1e-9,
                    tEnd=None):
    """ODMR contrast of a green1 -> MW -> green2 (readout) pulse sequence.

    contrast = (PL_ref - PL_mw) / PL_ref, where PL_mw and PL_ref are the NV-
    PL integrated over the readout window [tGreen2On, tGreen2Off], run with
    the MW on (Rabi rate OmegaR during [tMWOn, tMWOff)) and with the MW
    replaced by OmegaR=0 (same timing otherwise) as the reference run.
    tIROn/tIROff optionally pulse the IR probe I_s on only within that
    window (e.g. immediately after green2); see solvePulsedSequence.

    Returns
    -------
    contrast : float
    traces : dict with the two full time traces (t_mw, pops_mw, t_ref,
        pops_ref) and the two integrated readout PLs (PL_mw, PL_ref)
    """
    kwargs = dict(tGreen1On=tGreen1On, tGreen1Off=tGreen1Off,
                  tMWOn=tMWOn, tMWOff=tMWOff,
                  tGreen2On=tGreen2On, tGreen2Off=tGreen2Off,
                  tIROn=tIROn, tIROff=tIROff,
                  I_s=I_s, detuning=detuning, zeeman=zeeman,
                  rho0=rho0, dt=dt, tEnd=tEnd)
    t_mw, pops_mw = solvePulsedSequence(I_p, OmegaR, **kwargs)
    t_ref, pops_ref = solvePulsedSequence(I_p, 0.0, **kwargs)

    PL_mw = _integratedSignal(t_mw, pops_mw, 'PL_NVminus', tGreen2On, tGreen2Off)
    PL_ref = _integratedSignal(t_ref, pops_ref, 'PL_NVminus', tGreen2On, tGreen2Off)
    contrast = (PL_ref - PL_mw) / PL_ref

    traces = dict(t_mw=t_mw, pops_mw=pops_mw, t_ref=t_ref, pops_ref=pops_ref,
                  PL_mw=PL_mw, PL_ref=PL_ref)
    return contrast, traces


def singletContrast(I_p, OmegaR, tGreen1On, tGreen1Off, tMWOn, tMWOff,
                     tGreen2On, tGreen2Off,
                     tIROn=None, tIROff=None,
                     I_s=0,
                     detuning=detuning,
                     zeeman=zeeman,
                     rho0='mixedGround',
                     dt=1e-9,
                     tEnd=None):
    """IR-absorption-readout analog of pulsedContrast.

    Only the metastable singlet |7> (s2) absorbs the IR probe, so its
    population during the readout window is a proxy for the IR absorption
    an actual IR probe pulse would see. contrast = (pop_mw - pop_ref) /
    pop_mw, where pop_mw and pop_ref are the |7> population integrated over
    the readout window [tGreen2On, tGreen2Off], run with the MW on (Rabi
    rate OmegaR during [tMWOn, tMWOff)) and with the MW replaced by
    OmegaR=0 (same timing otherwise) as the reference run. MW on resonance
    shelves more population into the singlet than the reference, so this
    comes out positive. tIROn/tIROff optionally pulse the IR probe I_s on
    only within that window (e.g. immediately after green2); see
    solvePulsedSequence.

    Returns
    -------
    contrast : float
    traces : dict with the two full time traces (t_mw, pops_mw, t_ref,
        pops_ref) and the two integrated readout |7> populations
        (pop_s2_mw, pop_s2_ref)
    """
    kwargs = dict(tGreen1On=tGreen1On, tGreen1Off=tGreen1Off,
                  tMWOn=tMWOn, tMWOff=tMWOff,
                  tGreen2On=tGreen2On, tGreen2Off=tGreen2Off,
                  tIROn=tIROn, tIROff=tIROff,
                  I_s=I_s, detuning=detuning, zeeman=zeeman,
                  rho0=rho0, dt=dt, tEnd=tEnd)
    t_mw, pops_mw = solvePulsedSequence(I_p, OmegaR, **kwargs)
    t_ref, pops_ref = solvePulsedSequence(I_p, 0.0, **kwargs)

    pop_s2_mw = _integratedSignal(t_mw, pops_mw, 'singlet_s2', tGreen2On, tGreen2Off)
    pop_s2_ref = _integratedSignal(t_ref, pops_ref, 'singlet_s2', tGreen2On, tGreen2Off)
    contrast = (pop_s2_mw - pop_s2_ref) / pop_s2_mw

    traces = dict(t_mw=t_mw, pops_mw=pops_mw, t_ref=t_ref, pops_ref=pops_ref,
                  pop_s2_mw=pop_s2_mw, pop_s2_ref=pop_s2_ref)
    return contrast, traces


def plotPulsedSequence(I_p=I_pOpt, OmegaR=0, tGreen1On=0, tGreen1Off=2e-6,
                        tMWOn=2e-6, tMWOff=4e-6, tGreen2On=4e-6, tGreen2Off=6e-6,
                        tIROn=None, tIROff=None,
                        I_s=0, rho0='mixedGroundNonZero', I_pCompare=None,
                        saveBool=False):
    """Plot the MW-on vs MW-off (reference) traces for a full green1/MW/green2
    sequence, and report the resulting contrast.

    tIROn/tIROff : optional window in which the IR probe I_s is pulsed on
        (e.g. a readout IR pulse immediately after green2, via
        tIROn=tGreen2Off); if omitted, I_s is held constant throughout, as
        before. Shaded in the plot if given.
    I_pCompare : optional sequence of up to two other green intensities
        [W/m^2]; the CW steady-state NV- PL at each (same OmegaR, I_s,
        detuning, zeeman) is drawn as a horizontal line in the lower plot.
    """
    contrast, traces = pulsedContrast(I_p, OmegaR, tGreen1On, tGreen1Off,
                                       tMWOn, tMWOff, tGreen2On, tGreen2Off,
                                       tIROn=tIROn, tIROff=tIROff,
                                       I_s=I_s, rho0=rho0)
    t = traces['t_mw']*1e6
    sig_mw = derivedSignals(traces['pops_mw'])
    sig_ref = derivedSignals(traces['pops_ref'])

    fig, axArr = plot_custom(
        PRQuantumWidthHalf, 0.9*PRQuantumWidthHalf,
        r"time ($\upmu$s)", [r"$p_i$ (MW on)", r"NV$^-$ PL (au)"],
        labelPadX=3, labelPadY=10, commonY=False, commonX=True, hSpaceSetting=0.1,
        nrow=2, ncol=1, fontType='BeraSans', axefont=labelfontsize, numsize=numfontsize,
        widthBool=False
    )

    ax = axArr[0]
    for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
        ax.plot(t, traces['pops_mw'][:, i], label=STATE_LABELS[i], color=STATE_COLORS[i], lw=lw)
    ax.set_yscale('log')
    ax.set_ylim([1e-4, 1.1])
    ax.legend(fontsize=labelfontsize - 2, ncol=3, frameon=False, loc='lower right')

    ax = axArr[1]
    ax.plot(t, sig_ref['PL_NVminus'], color=colorBlue, lw=lw, label='MW off (ref)')
    ax.plot(t, sig_mw['PL_NVminus'], color=colorRed, lw=lw, label='MW on')
    if I_pCompare is not None:
        for I_pC, colorC in zip(I_pCompare, [colorDarkGreen, colorGreen]):
            PL_ss = _steadyStateSignal(I_pC, OmegaR, I_s, detuning, zeeman, 'PL_NVminus')
            ax.axhline(PL_ss, ls='--', lw=lw, color=colorC,
                       label=fr'CW steady state, $I_p$={I_pC:.2g}')
    ax.legend(fontsize=labelfontsize - 2, frameon=False, loc='lower right')
    ax.set_title(f"contrast = {contrast:.3%}", fontsize=labelfontsize)

    for ax in axArr:
        ax.axvspan(tGreen1On*1e6, tGreen1Off*1e6, color=colorLightGreen, alpha=0.15, lw=0)
        ax.axvspan(tGreen2On*1e6, tGreen2Off*1e6, color=colorLightGreen, alpha=0.15, lw=0)
        ax.axvspan(tMWOn*1e6, tMWOff*1e6, color=colorRed, alpha=0.08, lw=0)
        if tIROn is not None and tIROff is not None:
            ax.axvspan(tIROn*1e6, tIROff*1e6, color=colorLightBlue, alpha=0.15, lw=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    if saveBool:
        fig.savefig(os.path.join(_FIGURES_DIR, "NVPulsedSequence.svg"),
                    dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    return contrast, traces


def plotSingletSequence(I_p=I_pOpt, OmegaR=0, tGreen1On=0, tGreen1Off=2e-6,
                         tMWOn=2e-6, tMWOff=4e-6, tGreen2On=4e-6, tGreen2Off=6e-6,
                         tIROn=None, tIROff=None,
                         I_s=0, rho0='mixedGroundNonZero', OmegaRCompare=None,
                         saveBool=False):
    """Plot the MW-on vs MW-off (reference) |7> (singlet s2) population
    traces for a full green1/MW/green2 sequence -- the IR-absorption-readout
    analog of plotPulsedSequence -- and report the resulting contrast.

    tIROn/tIROff : optional window in which the IR probe I_s is pulsed on
        (e.g. a readout IR pulse immediately after green2, via
        tIROn=tGreen2Off); if omitted, I_s is held constant throughout, as
        before. Shaded in the plot if given.
    I_pCompare : optional sequence of up to two other green intensities
        [W/m^2]; the CW steady-state |7> population at each (same OmegaR,
        I_s, detuning, zeeman) is drawn as a horizontal line in the lower
        plot.
    """
    contrast, traces = singletContrast(I_p, OmegaR, tGreen1On, tGreen1Off,
                                        tMWOn, tMWOff, tGreen2On, tGreen2Off,
                                        tIROn=tIROn, tIROff=tIROff,
                                        I_s=I_s, rho0=rho0)
    t = traces['t_mw']*1e6
    sig_mw = derivedSignals(traces['pops_mw'])
    sig_ref = derivedSignals(traces['pops_ref'])

    fig, axArr = plot_custom(
        PRQuantumWidthHalf, 0.9*PRQuantumWidthHalf,
        r"time ($\upmu$s)", [r"$p_i$ (MW on)", r"$p_{|7\rangle}$ (au)"],
        labelPadX=3, labelPadY=10, commonY=False, commonX=True, hSpaceSetting=0.1,
        nrow=2, ncol=1, fontType='BeraSans', axefont=labelfontsize, numsize=numfontsize,
        widthBool=False
    )

    ax = axArr[0]
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        ax.plot(t, traces['pops_mw'][:, i], label=STATE_LABELS[i], color=STATE_COLORS[i], lw=lw)
    ax.set_yscale('log')
    ax.set_ylim([1e-4, 1.1])
    ax.legend(fontsize=labelfontsize - 2, ncol=3, frameon=False, loc='lower right')

    ax = axArr[1]
    ax.plot(t, sig_ref['singlet_s2'], color=colorBlue, lw=lw, label='MW off (ref)')
    ax.plot(t, sig_mw['singlet_s2'], color=colorRed, lw=lw, label='MW on')
    deltap7 = np.abs(sig_ref['singlet_s2'][-1] - sig_mw['singlet_s2'][-1])
    if OmegaRCompare is not None:
        pop_s2_ss = 0
        for OmegaR, colorC in zip(OmegaRCompare, [colorRed, colorBlue]):
            pop_s2_ssOld = pop_s2_ss
            pop_s2_ss = _steadyStateSignal(I_pOpt, OmegaR, I_sOpt, detuning, zeeman, 'singlet_s2')
            ax.axhline(pop_s2_ss, ls='--', lw=lw, color=colorC,
                       label=fr'CW steady state, $\Omega_R$={OmegaR:.2g}')
    ax.legend(fontsize=labelfontsize - 2, frameon=False, loc='lower right')
    deltap7ss = np.abs(pop_s2_ss.mean() - pop_s2_ssOld.mean())
    ax.set_title(f"Delta p7 percent = {deltap7*100:.2f}, {deltap7ss*100:.2f}", fontsize=labelfontsize)

    for ax in axArr:
        ax.axvspan(tGreen1On*1e6, tGreen1Off*1e6, color=colorLightGreen, alpha=0.15, lw=0)
        ax.axvspan(tGreen2On*1e6, tGreen2Off*1e6, color=colorLightGreen, alpha=0.15, lw=0)
        ax.axvspan(tMWOn*1e6, tMWOff*1e6, color=colorRed, alpha=0.08, lw=0)
        if tIROn is not None and tIROff is not None:
            ax.axvspan(tIROn*1e6, tIROff*1e6, color=colorLightBlue, alpha=0.15, lw=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    if saveBool:
        fig.savefig(os.path.join(_FIGURES_DIR, "NVSingletSequence.svg"),
                    dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    return contrast, traces


def rabiFloppingCurve(tMWArr, tWait, I_p=I_pOpt, OmegaR=2*np.pi*6e6,
                       tPulse=1e-6, tReadout=2e-6, I_s=0,
                       rho0='mixedGround', dt=2e-9, useSinglet=False):
    """Predicted Rabi-flopping contrast vs MW pulse duration, at one delay
    time tWait between the green1 init pulse and the MW pulse.

    For each tMW in tMWArr: tGreen1On=0, tGreen1Off=tPulse,
    tMWOn=tGreen1Off+tWait, tMWOff=tMWOn+tMW, tGreen2On=tMWOff (readout
    starts immediately when the MW pulse ends), tGreen2Off=tGreen2On+tReadout.
    Uses the NV- PL contrast (pulsedContrast) by default, or the |7>
    IR-absorption-readout contrast (singletContrast) if useSinglet=True.

    Returns contrastArr, the same shape as tMWArr.
    """
    contrastFn = singletContrast if useSinglet else pulsedContrast
    contrastArr = np.empty_like(tMWArr)
    for k, tMW in enumerate(tMWArr):
        tGreen1On = 0.0
        tGreen1Off = tGreen1On + tPulse
        tMWOn = tGreen1Off + tWait
        tMWOff = tMWOn + tMW
        tGreen2On = tMWOff
        tGreen2Off = tGreen2On + tReadout
        contrastArr[k], _ = contrastFn(I_p, OmegaR, tGreen1On, tGreen1Off,
                                        tMWOn, tMWOff, tGreen2On, tGreen2Off,
                                        I_s=I_s, rho0=rho0, dt=dt)
    return contrastArr


def plotRabiFlopping(I_p=I_pOpt, OmegaR=2*np.pi*6e6,
                      tPulse=1e-6, tWaitList=(0.1e-6, 1e-6, 5e-6),
                      tMWMax=None, nMW=151,
                      tReadout=2e-6, I_s=0,
                      rho0='mixedGround', dt=2e-9, useSinglet=False,
                      saveBool=False):
    """Plot predicted Rabi-flopping contrast vs MW pulse duration, for
    several different delay times (tWaitList) between the green1 init pulse
    and the MW pulse -- showing how spin re-polarization/relaxation during
    the delay affects the Rabi contrast and decay envelope.

    tMWMax defaults to 3 Rabi periods (2*pi/OmegaR) if not given.
    """
    if tMWMax is None:
        tMWMax = 3*(2*np.pi/OmegaR)
    tMWArr = np.linspace(0, tMWMax, nMW)

    ylabel = r"$|7\rangle$ contrast (\%)" if useSinglet else r"NV$^-$ PL contrast (\%)"
    fig, ax = plot_custom(
        PRQuantumWidthHalf, 0.7*PRQuantumWidthHalf,
        r"MW pulse duration ($\upmu$s)", ylabel,
        labelPadX=3, labelPadY=10, commonY=False, commonX=True,
        nrow=1, ncol=1, fontType='BeraSans', axefont=labelfontsize, numsize=numfontsize,
        widthBool=False
    )

    colors = [colorBlue, colorGreen, colorRed, colorDarkBlue, colorDarkGreen, colorDarkRed]
    results = {}
    for i, tWait in enumerate(tWaitList):
        contrastArr = rabiFloppingCurve(tMWArr, tWait, I_p=I_p, OmegaR=OmegaR,
                                         tPulse=tPulse, tReadout=tReadout, I_s=I_s,
                                         rho0=rho0, dt=dt, useSinglet=useSinglet)
        results[tWait] = contrastArr
        ax.plot(tMWArr*1e6, contrastArr*100, color=colors[i % len(colors)], lw=lw,
                label=fr'$t_{{wait}}$ = {tWait*1e6:.2g} $\upmu$s')

    ax.legend(fontsize=labelfontsize - 2, frameon=False, loc='best')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if saveBool:
        fig.savefig(os.path.join(_FIGURES_DIR, "NVRabiFlopping.svg"),
                    dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    return tMWArr, results


def plotPulsedPopulation(I_p=I_pOpt, tPulse=2e-6, tDark=2e-6, I_s=0, OmegaR=0,
                         rho0='mixedGroundNonZero', saveBool=False):
    """Plot populations (log scale) and derived signals for one green pulse."""
    t, pops = solvePulsedPopulation(I_p, tPulse, tDark, I_s=I_s, OmegaR=OmegaR, rho0=rho0)
    sig = derivedSignals(pops)
    tus = t*1e6

    fig, axArr = plot_custom(
        PRQuantumWidthHalf, 0.9*PRQuantumWidthHalf,
        r"time ($\upmu$s)", [r"$p_i$", r"signal (au)"],
        labelPadX=3, labelPadY=10, commonY=False, commonX=True, hSpaceSetting=0.1,
        nrow=2, ncol=1, fontType='BeraSans', axefont=labelfontsize, numsize=numfontsize,
        widthBool=False
    )

    ax = axArr[0]
    for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
        ax.plot(tus, pops[:, i], label=STATE_LABELS[i], color=STATE_COLORS[i], lw=lw)
    ax.set_yscale('log')
    ax.set_ylim([1e-4, 1.1])
    ax.legend(fontsize=labelfontsize - 2, ncol=3, frameon=False, loc='lower right')

    ax = axArr[1]
    ax.plot(tus, sig['ms0'], color=colorGreen, lw=lw, label=r'$m_s{=}0$ polarization')
    ax.plot(tus, sig['NVminus'], color=colorRed, lw=lw, label=r'NV$^-$ fraction')
    ax.plot(tus, sig['PL_NVminus']/sig['PL_NVminus'].max(), color=colorBlue, lw=lw,
            label=r'NV$^-$ PL (norm.)')
    ax.legend(fontsize=labelfontsize - 2, frameon=False, loc='lower right')

    for ax in axArr:
        ax.axvspan(0, tPulse*1e6, color=colorLightGreen, alpha=0.15, lw=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    if saveBool:
        fig.savefig(os.path.join(_FIGURES_DIR, "NVPulsedDynamics.svg"),
                    dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    return t, pops, sig


def pulsedSensitivitySingle(I_s, tIRPulse, tPulse2,
                             I_p=I_pOpt*3, OmegaR=2*np.pi*6e6,
                             tPulse=2e-6, tWait=1e-6, tMW=None,
                             tWait2=0.0,
                             detuning=detuning, zeeman=zeeman,
                             rho0='mixedGround',
                             returnAllParams=False):
    """Sensitivity of the pulsed green1 -> MW -> green2 -> IR readout at one
    (I_s, tIRPulse, tWait2) point; everything else held at its current value.

    sensitivity = sqrt(total time) / (|Delta p7| * sqrt(I_s * tIRPulse)),
    where Delta p7 is the |7> population reached by the end of the final IR
    pulse, MW-on vs MW-off (same sign convention as singletContrast), and
    total time is the full single-shot sequence duration (through the end
    of the IR pulse). This is the pulsed analog of solveSensitivitySingle in
    NVpopulationDynamics.py (the CW IR-steady-state scheme): the finite-time
    |7> populations from solvePulsedSequence stand in for the CW steady-state
    populations, and I_s*tIRPulse (the IR pulse's photon fluence) stands in
    for the CW photon count rate in the shot-noise denominator.

    tGreen2On = tMWOff + tPulse2 and tIROn = tGreen2Off + tWait3, exactly as
    in the __main__ pulse-sequence setup; tWait3 is fixed (default 0, i.e.
    the IR pulse starts immediately when green2 turns off) rather than
    optimized over.

    dt=1.0 is passed to solvePulsedSequence deliberately: each pulse segment
    has a constant drive, so propagating it in a single expm(L*tSeg) step
    (rather than many fine steps for a smooth trajectory, which is all that
    dt controls) is exact, not an approximation -- and far cheaper for an
    optimization inner loop than the fine sampling the plotting functions use.
    """
    if tMW is None:
        tMW = np.pi/OmegaR if OmegaR > 0 else 0.0  # default: a pi-pulse

    tGreen1On = 0.0
    tGreen1Off = tGreen1On + tPulse
    tMWOn = tGreen1Off + tWait
    tMWOff = tMWOn + tMW
    tGreen2On = tMWOff
    tGreen2Off = tGreen2On + tPulse2
    tIROn = tGreen2Off
    tIROff = tIROn + tIRPulse

    kwargs = dict(tGreen1On=tGreen1On, tGreen1Off=tGreen1Off,
                  tMWOn=tMWOn, tMWOff=tMWOff,
                  tGreen2On=tGreen2On, tGreen2Off=tGreen2Off,
                  tIROn=tIROn, tIROff=tIROff,
                  I_s=I_s, detuning=detuning, zeeman=zeeman,
                  rho0=rho0, dt=1.0)
    _, pops_mw = solvePulsedSequence(I_p, OmegaR, **kwargs)
    t, pops_ref = solvePulsedSequence(I_p, 0.0, **kwargs)

    deltaP7 = abs(np.mean(pops_mw[-1, 7]) - np.mean(pops_ref[-1, 7])) #this is an underestimate
    totalTime = tIROff

    sens = np.sqrt(totalTime) / (deltaP7 * np.sqrt(I_s*tIRPulse))

    if returnAllParams:
        return sens, dict(deltaP7=deltaP7, totalTime=totalTime,
                           tGreen2On=tGreen2On, tIROn=tIROn, tIROff=tIROff)
    return sens


def objectivePulsedIR(trial: optuna.Trial, I_p=I_pOpt*3, OmegaR=2*np.pi*6e6):
    """Optuna objective: pulsed IR-absorption-readout sensitivity (via
    pulsedSensitivitySingle) to minimize over I_s, tIRPulse, tPulse2, tWait."""
    log_Is = trial.suggest_float("log10_Is", 9, 12)
    log_tIRPulse = trial.suggest_float("log10_tIRPulse", -7, -5)
    tPulse2 = trial.suggest_float("tPulse2", 0.0, 1e-6)
    tWait = trial.suggest_float("tWait", 0.0, 2e-6)

    I_s = 10**log_Is
    tIRPulse = 10**log_tIRPulse

    sens = pulsedSensitivitySingle(I_s=I_s, tIRPulse=tIRPulse, tPulse2=tPulse2,
                                    tWait=tWait, I_p=I_p, OmegaR=OmegaR)

    trial.report(sens, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return sens


def _cwDeltaP7(I_s, I_p, OmegaR, detuning, zeeman):
    """Steady-state |7> population difference, MW-on vs MW-off -- the CW
    (continuous-illumination) analog of pulsedSensitivitySingle's Delta p7."""
    p7_on = _steadyStateSignal(I_p, OmegaR, I_s, detuning, zeeman, 'singlet_s2')
    p7_off = _steadyStateSignal(I_p, 0.0, I_s, detuning, zeeman, 'singlet_s2')
    return abs(p7_on - p7_off)


def cwSensitivityBestIs(I_p, OmegaR, detuning=detuning, zeeman=zeeman,
                         I_s = I_sOpt, nGrid=300):
    """Best CW IR-absorption-readout sensitivity at fixed I_p/OmegaR, scanning
    I_s on a log grid.

    For continuous (CW) illumination the IR probe runs for the whole
    measurement, so in the pulsed formula's sqrt(total time)/(Delta p7 *
    sqrt(I_s * probe time)) the probe time equals the total time and the
    sqrt(total time) factors cancel: sensitivity = 1/(Delta p7 * sqrt(I_s)),
    independent of total averaging time (the standard CW convention).

    Returns
    -------
    sens
    """
    deltaP7 = _cwDeltaP7(I_s, I_p, OmegaR, detuning, zeeman)
    sens = 1.0/(deltaP7*np.sqrt(I_s))
    return deltaP7, sens


def performPulsedIROptimization(n_trials=60, I_p=I_pOpt*3, OmegaR=2*np.pi*6e6):
    """Run an Optuna study minimizing objectivePulsedIR and print the best
    I_s/tIRPulse/tPulse2/tWait (and the resulting tGreen2On/tIROn/tIROff)
    found, plus the improvement factor over the best CW (continuous-
    illumination) sensitivity at the same I_p/OmegaR.
    """
    start_time = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=3,   # random exploration first
            multivariate=True,
            seed=123,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=30,
            n_warmup_steps=0
        ),
    )

    study.optimize(
        lambda trial: objectivePulsedIR(trial, I_p=I_p, OmegaR=OmegaR),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1                   # >1 only if solver is thread-safe
    )

    elapsed = time.time() - start_time

    print("\nOptimization finished")
    print(f"Elapsed time: {elapsed/60:.1f} min")
    print("Best sensitivity:", study.best_value)

    best = study.best_params
    I_s_best = 10**best["log10_Is"]
    tIRPulse_best = 10**best["log10_tIRPulse"]
    tPulse2_best = best["tPulse2"]
    tWait_best = best["tWait"]
    _, infoBest = pulsedSensitivitySingle(I_s=I_s_best, tIRPulse=tIRPulse_best,
                                           tPulse2=tPulse2_best, tWait=tWait_best,
                                           I_p=I_p, OmegaR=OmegaR, returnAllParams=True)
    print(
        "Best parameters:\n"
        f"  I_s        = {I_s_best:.3e}\n"
        f"  tIRPulse   = {tIRPulse_best:.3e}\n"
        f"  tPulse2    = {tPulse2_best:.3e}\n"
        f"  tWait      = {tWait_best:.3e}\n"
        f"  tGreen2On  = {infoBest['tGreen2On']:.3e}\n"
        f"  tIROn      = {infoBest['tIROn']:.3e}\n"
        f"  tIROff     = {infoBest['tIROff']:.3e}\n"
        f"  Delta p7   = {infoBest['deltaP7']:.3e}"
    )

    # --- Compare against the best CW (continuous-illumination) sensitivity ---
    deltap7CW, sensCW = cwSensitivityBestIs(I_p=I_pOpt, OmegaR=OmegaROptRadps)
    improvementFactor = (
        1/(infoBest['deltaP7']*np.sqrt(I_s_best*tIRPulse_best/infoBest['tIROff']))/sensCW
    )
    # improvementFactor = sens_pulsed / sens_CW (both "lower is better"), so
    # a value BELOW 1 means the pulsed scheme is more sensitive than CW.
    print(
        "\nCW comparison (same I_p/OmegaR, best I_s found by a log grid scan):\n"
        f"  I_sOptCW   = {sensCW:.3e}\n"
        f"  Delta p7CW = {deltap7CW:.3e}\n"
        f"Sensitivity ratio (pulsed/CW) "
        f"(Delta p7CW*sqrt(tTotal*I_sOptCW))/(Delta p7*sqrt(I_s*tIRPulse)) = "
        f"{improvementFactor:.3g}  ({'pulsed better' if improvementFactor < 1 else 'CW better'}; "
        f"pulsed is {1/improvementFactor:.3g}x more sensitive than CW)"
    )
    return study


if __name__ == "__main__":
    saveBool = False

    # plotPulsedPopulation(I_p=I_pOpt, tPulse=4e-6, tDark=2e-6, rho0 = 'mixedGround', saveBool=saveBool)

    # general init-pulse -> MW -> readout-pulse sequence, with ODMR contrast
    tPulse = 2e-6      # green1 (init) pulse duration
    tWait = 2e-6          # dark wait between green1 and the MW pulse
    tWait2 = 0.2e-6          # dark wait between green1 and the MW pulse
    tWait3 = 0e-6          # dark wait between green2 and the IR pulse
    OmegaR = 2*np.pi*6e6
    tMW = 2*np.pi/OmegaR/1.6 #0.09e-6       # MW pulse duration
    tReadout = 2e-6    # green2 (readout) pulse duration
    tIRPulse = 0e-6    # final IR probe pulse duration
    tGreen1On = 0
    tGreen1Off = tGreen1On + tPulse
    tMWOn = tGreen1Off + tWait
    tMWOff = tMWOn + tMW
    tGreen2On = tMWOff+tWait2
    tGreen2Off = tGreen2On + tReadout
    tIROn = tGreen2Off  + tWait3         # IR pulse starts immediately after green2
    tIROff = tIROn + tIRPulse

    # plotPulsedSequence(I_p=I_pOpt*2.5, OmegaR=2*np.pi*5e6,
    #                     tGreen1On=tGreen1On, tGreen1Off=tGreen1Off,
    #                     tMWOn=tMWOn, tMWOff=tMWOff,
    #                     tGreen2On=tGreen2On, tGreen2Off=tGreen2Off,
    #                     tIROn=tIROn, tIROff=tIROn,
    #                     saveBool=saveBool)
    
    # general init-pulse -> MW -> readout-pulse sequence, with ODMR contrast
    tPulse = 1e-6      # green1 (init) pulse duration
    tWait = 9.807e-08          # dark wait between green1 and the MW pulse
    tWait2 = 0.0e-6          # dark wait between green1 and the MW pulse
    tWait3 = 0e-6          # dark wait between green2 and the IR pulse
    OmegaR = 2*np.pi*1e6
    tMW = 2*np.pi/OmegaR/2 #0.09e-6       # MW pulse duration
    tPulse2 = 2.749e-07    # green2 (readout) pulse duration
    tIRPulse = 3.917e-07    # final IR probe pulse duration
    tGreen1On = 0
    tGreen1Off = tGreen1On + tPulse
    tMWOn = tGreen1Off + tWait
    tMWOff = tMWOn + tMW
    tGreen2On = tMWOff+tWait2
    tGreen2Off = tGreen2On + tPulse2
    tIROn = tGreen2Off  + tWait3         # IR pulse starts immediately after green2
    tIROff = tIROn + tIRPulse
    I_sOptPulsed = 9.888e+11

    # same sequence, but monitoring |7> (singlet s2) as an IR-absorption-readout analog
    # plotSingletSequence(I_p=I_pOpt*3, OmegaR=OmegaR,
    #                       tGreen1On=tGreen1On, tGreen1Off=tGreen1Off,
    #                       tMWOn=tMWOn, tMWOff=tMWOff,
    #                       tGreen2On=tGreen2On, tGreen2Off=tGreen2Off,
    #                       tIROn=tIROn, tIROff=tIROff, I_s=I_sOptPulsed,
    #                       OmegaRCompare=[OmegaROptRadps, 0],
    #                       saveBool=saveBool, rho0 = 'mixedGround')

    # optimize the pulsed IR-absorption readout sensitivity over I_s,
    # tIRPulse (final IR pulse duration), tPulse2 (green2/readout duration),
    # and tWait (gap between green1 and the MW pulse); prints the best
    # params and the improvement factor over the best CW sensitivity
    performPulsedIROptimization(n_trials=200)

    # predicted Rabi flopping: NV- PL contrast vs MW pulse duration, for a
    # few different delay times between the green1 init pulse and the MW
    # pulse -- shows how much spin polarization is lost (contrast decays)
    # the longer you wait before applying the MW pulse
    tPulse = 2e-6      # green1 (init) pulse duration
    tWait = 0.1e-6          # dark wait between green1 and the MW pulse
    tWait2 = 0.0e-6          # dark wait between green1 and the MW pulse
    tWait3 = 0e-6          # dark wait between green2 and the IR pulse
    OmegaR = 2*np.pi*2e6
    tMW = 2*np.pi/OmegaR/2 #0.09e-6       # MW pulse duration
    tReadout = 1e-6    # green2 (readout) pulse duration
    tIRPulse = 0e-6    # final IR probe pulse duration
    tGreen1On = 0
    tGreen1Off = tGreen1On + tPulse
    tMWOn = tGreen1Off + tWait
    tMWOff = tMWOn + tMW
    tGreen2On = tMWOff+tWait2
    tGreen2Off = tGreen2On + tReadout
    tIROn = tGreen2Off  + tWait3         # IR pulse starts immediately after green2
    tIROff = tIROn + tIRPulse
    
    # plotPulsedSequence(I_p=I_pOpt*2.5, OmegaR=OmegaR,
    #                     tGreen1On=tGreen1On, tGreen1Off=tGreen1Off,
    #                     tMWOn=tMWOn, tMWOff=tMWOff,
    #                     tGreen2On=tGreen2On, tGreen2Off=tGreen2Off,
    #                     tIROn=tIROn, tIROff=tIROn,
    #                     saveBool=saveBool)
    
    # plotRabiFlopping(I_p=I_pOpt, OmegaR=OmegaR, tPulse=tPulse,
    #                   tWaitList=[0.1e-6, 2e-6], tReadout=tReadout,
    #                   saveBool=saveBool,nMW = 30)
# 