# -*- coding: utf-8 -*-
"""
NV center 11-level model in a cavity (microdisk)
@author: nicho (refactor)
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from linbladMESolverFunctionChargeConversionLiouvillian import nv_steady_state, nv_steady_state_fast#, nv_steady_state_IRRate
from commonlabcode.plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible

from scipy.io import loadmat
from tqdm import tqdm
import optuna
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIGURES_DIR = os.path.join(_THIS_DIR, "..", "__figures")
_REFFILES_DIR = os.path.join(_THIS_DIR, "..", "__refFiles")

counter = mp.Value('i', 0)

from nvConstants import *
from functools import partial

# Nearly every nv_steady_state(...) call in this file passes the same ~14 rate/
# material constants; only Omega/I_s/I_p/detuning/zeeman/steadystatebool/tEnd vary
# between call sites. Binding the fixed ones here removes ~20 lines of repetition
# per call site (previously copy-pasted at 68 call sites).
nv_steady_state_default = partial(
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
)

# Several functions adaptively grow/shrink a detuning bandwidth `bw` in a loop
# until the resulting FWHM lands in a target range (too wide -> bw *= 2; too
# narrow -> bw is divided by an ad-hoc step factor, e.g. 1.6 or 3, that just
# controls convergence speed and isn't otherwise meaningful). That loop used
# to be `while True:`, which could hang forever if a sweep point never
# converges; it's now capped at this many attempts.
MAX_BISECTION_ITERS = 25

# angleNV = 35 #deg
# alpha0 = 12.6/4 * np.cos(np.pi/180 * angleNV)**2
# alpha1 = 19.3/4 * np.cos(np.pi/180 * angleNV)**2
# alpha1 = 15.5/4 * np.cos(np.pi/180 * angleNV)**2

# alpha0 = 2.44
# alpha1 = alpha0 + 0.11


        
def solveTimeDependentPopulation(saveBool = False,
                                    OmegaR = 2*np.pi*1e6, #rad/s
                                    I_p = 1e8,
                                    I_s = 1e8, bool1042 = False, detuning = detuning):
    """Time-evolve the NV 11-level populations for one set of drive parameters and plot them on a log scale."""

    result, tlist = nv_steady_state_default(
        Omega = OmegaR,
        I_s = I_s,
        I_p = I_p,
        detuning = detuning,
        zeeman = zeeman,
        steadystatebool=False,
        tEnd = 5e-6,
        bool1042 = bool1042,
    )
    
    # ---------------------------
    # Plot populations
    # ---------------------------
    fig, ax = plot_custom(
        PRQuantumWidthFull*0.3, 0.53*PRQuantumWidthFull, 
        r"$\lambda$ (nm)", 
        r'$T$ (au)',
        labelPadX = 3, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
              r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
    colors = ['#3dfe00ff','#35db00ff','#218800ff',
              '#fe7900ff','#fe0000ff','#fe0076ff',
              '#190cffff','#800000ff','#800cffff', '#ff0cf4ff', '#feb400ff'];
    for i in range(11):
        if i in [0, 1, 2, 3, 4, 5, 7]:
            ax.plot(tlist*1e6, result.expect[i], label=labels[i], color = colors[i], lw = 0.2/0.3528)
    
    # P = -np.cos(np.sqrt(OmegaR**2 + (detuning-zeeman)**2)*tlist)/2+0.5
    # ax.plot(tlist*1e6, P, 'k-')
    # ax.plot(tlist*1e6, np.exp(-gamma_phi * tlist)*0.7+0.3, lw = 0.5)
    # ax.plot(tlist*1e6, (1-(np.exp(-gamma_phi * tlist)*0.7+0.3))*0.45, lw = 0.5)
    
    ax.set_yscale('log'); 
    # plt.yscale('log');
    ax.set_xlabel(r"time ($\upmu$s)")
    ax.set_ylabel(r"population in $|i\rangle$, $p_i$ (au)")
    # plt.legend(fontsize = labelfontsize, loc = 'lower right')
    # plt.tight_layout()
    
    tmin = 0
    tmax = 1
    ax.set_ylim([8e-4, 1.1e0])
    ax.spines['bottom'].set_bounds(tmin, tmax)
    xtick = ax.get_xticks()
    ax.set_xticklabels([rf"${val:.1f}$" for val in xtick])
    
    ax.spines['left'].set_bounds(1e-3,1)
    ytick = ax.get_yticks(minor = True)
    filt = np.logical_and(ytick > 1e-3, ytick < 1) 
    ytick = ytick[filt]
    ax.set_yticks(ytick,minor = True)
    ax.spines['left'].set_bounds(1e-3,1)
    
    ax.set_xticklabels([rf"${val:.1f}$" for val in xtick])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVTimeSolution.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    
    plt.show()

def solvePeakWidthAndHeight(x, y):
    """Return the peak height and FWHM (in x-units) of the tallest peak in y(x)."""
    peaks, _ = find_peaks(y)
    FWHM, _, _, _  = peak_widths(y, peaks, rel_height=0.5)
    try:
        maxSignal = y[peaks[0]]
    except:
        maxSignal= np.max(y)
    FWHM = FWHM*(x[1] - x[0])
    return maxSignal, FWHM
    
def calcSensitivityCW(nu, C, P, lamb):
    """CW magnetometry sensitivity from resonance frequency nu, contrast C, optical power P, and wavelength lamb."""
    return 4/(3*np.sqrt(3)) * 1/gamma * nu/C * np.sqrt(h*c_const/lamb/P)
def calcSensitivityCW_countrate(nu, C,R):
    """CW magnetometry sensitivity from resonance frequency nu, contrast C, and photon count rate R."""
    return 4/(3*np.sqrt(3)) * 1/gamma * nu/C * np.sqrt(1/R)

def solveSensitivity(saveBool = False,
                    N_green = 5,
                    N_IR = 5,
                    N_OmegaR = 5,
                    I_smin = 5,
                    I_smax = 10,
                    I_pmin = 5,
                    I_pmax = 10,
                    OmegaRmin = 4,       
                    OmegaRmax = 7,           
                    I_p = 1e7,
                    I_s = 1e12,
                    bw = 0.005e9*2*np.pi,
                    Npoints = 51,
                    h = h,
                    c = c_const,
                    bool1042 = True):
    """Grid sensitivity over green power, IR power, and Rabi frequency, each fit via a detuning scan."""
    # detuning
    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
    
    I_green = np.logspace(I_pmin, I_pmax, N_IR)
    I_IR = np.logspace(I_smin, I_smax, N_green)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    
    
    mOnArr = np.zeros([N_IR,N_green,N_OmegaR])
    mOffArr = np.zeros([N_IR,N_green,N_OmegaR])
    nu = np.zeros([N_IR,N_green,N_OmegaR])
    C = np.zeros([N_IR,N_green,N_OmegaR])
    
    total = N_IR*N_green*N_OmegaR
    with tqdm(total=total, desc="Running simulation") as pbar:
        for l, OmegaR in enumerate(OmegaRArr):
            for i, I_p in enumerate(I_green):
                for j, I_s in enumerate(I_IR):
                    delpop = np.zeros(Npoints)
                    mOn = np.zeros(Npoints)
                    mOff = np.zeros(Npoints)
                    for k, det in enumerate(detunings):
                    
                        popOn = nv_steady_state_default(
                            Omega = OmegaR,
                            I_s = I_s,
                            I_p = I_p,
                            detuning = (det),
                            zeeman = zeeman,
                            steadystatebool=True,
                            bool1042 = bool1042,
                        )
                        popOff = nv_steady_state_default(
                            Omega = OmegaR*0,
                            I_s = I_s,
                            I_p = I_p,
                            detuning = (det),
                            zeeman = zeeman,
                            steadystatebool=True,
                            tEnd = 2e-6,
                            bool1042 = bool1042,
                        )
                        delpop[k] = (popOn[7] - popOff[7])
                        mOn[k] = (popOn[7])
                        mOff[k] = (popOff[7])
                    C[i,j,l] = np.max(delpop)
                    maxSignal, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), mOn)
                    nu[i,j,l] = FWHM[0]
                    mOnArr[i,j,l] = maxSignal
                    mOffArr[i,j,l] = np.max(mOff)
                    pbar.update(1)

    
    return C, nu, mOnArr, mOffArr, I_IR, I_green, OmegaRArr
    
    
def _solveSensitivityVisSingle_worker(det, OmegaR, I_p, ROff):
    """Compute (Cdet[k], ROn[k]) for one detuning point of solveSensitivityVisSingle's inner scan."""
    popDet = nv_steady_state_default(
        Omega=OmegaR,
        I_s=0,
        I_p=I_p,
        detuning=det,
        zeeman=zeeman,
    )
    ROn_k = (popDet[3] + popDet[4] + popDet[5])*gamma_eg
    Cdet_k = (ROff - ROn_k)/ROff
    return Cdet_k, ROn_k


def solveSensitivityVisSingle(OmegaR = gamma_phi,
                            I_p = 1e7,
                            bw = 0.04e9*2*np.pi,
                            Npoints = 101,
                            lamb = 532e-9,
                            bool1042 = True,
                            returnAllParams = False,
                            parallelBool = False,
                            maxWorkers = None):
    """Sensitivity at one (OmegaR, I_p) point using the green(NV0)-PL contrast, via an adaptive detuning-bandwidth fit.

    The inner detuning scan (Npoints points, repeated up to
    MAX_BISECTION_ITERS times until it converges) is independent per point
    and can be dispatched across a ProcessPoolExecutor via parallelBool=True
    (pool created once per call, reused across bisection attempts within
    that call) -- but default OFF: measured slower than serial at the
    default Npoints=101 (a single solve here is ~12ms; per-call pool
    startup, which re-imports qutip/numba in each worker, costs more than
    that saves). A threaded version was also tested and gave no speedup
    either -- the solve holds the GIL throughout. Only worth enabling if
    Npoints is increased substantially, or if you switch to a persistent
    pool shared across many calls (e.g. across an optimization run) rather
    than one created fresh per call.
    """

    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)

    delpop = np.zeros(Npoints)
    mOn = np.zeros(Npoints)


    popOff = nv_steady_state_default(
        Omega = OmegaR*0,
        I_s = 0,
        I_p = I_p,
        detuning = zeeman,
        zeeman = zeeman,
    )
    ROff = (popOff[3] + popOff[4] + popOff[5])*gamma_eg

    executor_cm = ProcessPoolExecutor(max_workers=maxWorkers) if parallelBool else nullcontext()
    with executor_cm as executor:
        for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
            detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
            Cdet = np.zeros(Npoints)
            ROn = np.zeros(Npoints)
            if parallelBool:
                futures = [executor.submit(_solveSensitivityVisSingle_worker, det, OmegaR, I_p, ROff) for det in detunings]
                for k, future in enumerate(futures):
                    Cdet[k], ROn[k] = future.result()
            else:
                for k, det in enumerate(detunings):
                    Cdet[k], ROn[k] = _solveSensitivityVisSingle_worker(det, OmegaR, I_p, ROff)

            peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
            # plt.plot(detunings/1e9/(2*np.pi),Cdet)
            # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
            # plt.show()
            if np.size(FWHM) < 1:
                print('FWHM fitting didnt work')
            # elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.01:
            #     bw = bw*2
            #     continue
            elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/6/1e9/(2*np.pi):
                nu = FWHM[0]*1e6
                C = np.max(Cdet)
                break
            elif FWHM >= bw/3/1e9/2/np.pi:
                bw = bw*2
                continue
            else:
                bw = bw/1.6
                continue

    maxSignal, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), mOn)

    # print(C/nu)
    sens = calcSensitivityCW_countrate(nu*1e3, C, ROn.max())
    if returnAllParams:
        return sens, C, nu, ROn.max()
    else:
        return sens

def _solveSensitivitySingle_worker(det, OmegaR, I_s, I_p, p7Off):
    """Compute Cdet[k] for one detuning point of solveSensitivitySingle's inner scan."""
    popDet = nv_steady_state_default(
        Omega=OmegaR,
        I_s=I_s,
        I_p=I_p,
        detuning=det,
        zeeman=zeeman,
    )
    return popDet[7] - p7Off


def solveSensitivitySingle(OmegaR = gamma_phi,
                            I_p = 1e7,
                            I_s = 1e12,
                            bw = 0.04e9*2*np.pi,
                            Npoints = 101,
                            lamb = 1042e-9,
                            bool1042 = True,
                            parallelBool = False,
                            maxWorkers = None):
    """Sensitivity at one (OmegaR, I_s, I_p) point using the IR-absorption contrast, via an adaptive detuning-bandwidth fit.

    The inner detuning scan (Npoints points, repeated up to
    MAX_BISECTION_ITERS times until it converges) is independent per point
    and can be dispatched across a ProcessPoolExecutor via parallelBool=True
    (pool created once per call, reused across bisection attempts within
    that call) -- but default OFF: measured slower than serial at the
    default Npoints=101 (a single solve here is ~12ms; per-call pool
    startup, which re-imports qutip/numba in each worker, costs more than
    that saves). A threaded version was also tested and gave no speedup
    either -- the solve holds the GIL throughout. Only worth enabling if
    Npoints is increased substantially, or if you switch to a persistent
    pool shared across many calls (e.g. across an optimization run) rather
    than one created fresh per call.
    """

    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)

    delpop = np.zeros(Npoints)
    mOn = np.zeros(Npoints)


    popOff = nv_steady_state_default(
        Omega = OmegaR*0,
        I_s = I_s,
        I_p = I_p,
        detuning = zeeman,
        zeeman = zeeman,
    )
    p7Off = popOff[7]
    # p7Off = popOff[3] + popOff[4] + popOff[5]

    executor_cm = ProcessPoolExecutor(max_workers=maxWorkers) if parallelBool else nullcontext()
    with executor_cm as executor:
        for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
            detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
            Cdet = np.zeros(Npoints)
            if parallelBool:
                futures = [executor.submit(_solveSensitivitySingle_worker, det, OmegaR, I_s, I_p, p7Off) for det in detunings]
                for k, future in enumerate(futures):
                    Cdet[k] = future.result()
            else:
                for k, det in enumerate(detunings):
                    Cdet[k] = _solveSensitivitySingle_worker(det, OmegaR, I_s, I_p, p7Off)

            peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
            # plt.plot(detunings/1e9/(2*np.pi),Cdet)
            # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
            # plt.show()
            if np.size(FWHM) < 1:
                print('FWHM fitting didnt work')
            # elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.01:
            #     bw = bw*2
            #     continue
            elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/6/1e9/(2*np.pi):
                nu = FWHM[0]*1e6
                C = np.max(Cdet)
                break
            elif FWHM >= bw/3/1e9/2/np.pi:
                bw = bw*2
                continue
            else:
                bw = bw/1.6
                continue

            # delpop[k] = np.sum(popOn*kappaOn - popOff*kappaOff)
            # mOn[k] = np.sum(popOn*kappaOn)
            # mOff[k] = np.sum(popOff*kappaOff)
    # C = np.max(delpop)
    maxSignal, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), mOn)
    # nu = FWHM[0]
    # mOn = maxSignal
    # mOff = np.max(mOff)

    print(C/nu)
    sens = calcSensitivityCW(nu, C, I_s, lamb)

    return sens

# def objective(x):
#     # I_p = 10**optuna.trial.Trial.suggest_float("logIp", 5, 10)
#     # I_s = 10**optuna.trial.suggest_float("logIs", 6, 11)
#     # OmegaR = 10**optuna.trial.suggest_float("logOmegaR", 4, 7)

#     I_p = 10**x[0]
#     I_s = 10**x[1]
#     OmegaR = 10**x[2]

#     return solveSensitivitySingle(
#                                 OmegaR=OmegaR,
#                                 I_p=I_p,
#                                 I_s=I_s,
#                                 bw=0.1e9*2*np.pi,
#                                 Npoints=401,
#                                 lamb=1042e-9
#                             )
# def objective_wrapped(x):
#     with counter.get_lock():
#         counter.value += 1
#         n = counter.value

#     val = objective(x)

#     if n % 10 == 0:
#         print(f"Eval {n}: sens = {val:.3e}")

#     return val

def objective(trial: optuna.Trial):
    """Optuna objective: sensitivity (via solveSensitivitySingle) to minimize over I_p/I_s/OmegaR."""

    # --- Log-parameterized search space ---
    log_Ip = trial.suggest_float("log10_Ip", 8, 8.6)
    log_Is = trial.suggest_float("log10_Is", 9.9, 10.3)
    log_OmegaR = trial.suggest_float("log10_OmegaR", 5.8, 6.3)

    I_p = 10**log_Ip
    I_s = 10**log_Is
    OmegaR = 10**log_OmegaR

    # --- Optional physics constraints ---
    # Example: avoid extreme imbalance
    if I_s / I_p > 1e7:
        return np.inf

    # --- Evaluate sensitivity ---
    sens = solveSensitivitySingle(
                                    OmegaR=OmegaR,
                                    I_p=I_p,
                                    
                                    I_s=I_s,
                                    bw=0.01e9*2*np.pi,
                                    Npoints=101,
                                    lamb=1042e-9
                                    )

    # --- Report intermediate value (for pruning) ---
    trial.report(sens, step=0)

    # Optional pruning if clearly bad
    if trial.should_prune():
        raise optuna.TrialPruned()

    return sens

def objectiveVis(trial: optuna.Trial):
    """Optuna objective: sensitivity (via solveSensitivityVisSingle) to minimize over I_p/OmegaR."""

    # --- Log-parameterized search space ---
    log_Ip = trial.suggest_float("log10_Ip", 7.5, 8.5)
    log_OmegaR = trial.suggest_float("log10_OmegaR", 5, 7)

    I_p = 10**log_Ip
    OmegaR = 10**log_OmegaR


    # --- Evaluate sensitivity ---
    sens = solveSensitivityVisSingle(
                                    OmegaR=OmegaR,
                                    I_p=I_p,
                                    bw=0.01e9*2*np.pi,
                                    Npoints=101,
                                    lamb=532e-9
                                    )

    # --- Report intermediate value (for pruning) ---
    trial.report(sens, step=0)

    # Optional pruning if clearly bad
    if trial.should_prune():
        raise optuna.TrialPruned()

    return sens

def performVisOptimization():
    """Run an Optuna study minimizing objectiveVis and print the best I_p/OmegaR found."""
    # --- Create study ---
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

    # --- Run optimization ---
    study.optimize(
        objectiveVis,
        n_trials=100,              # increase if slow landscape
        show_progress_bar=True,
        n_jobs=1                   # >1 only if solver is thread-safe
    )

    elapsed = time.time() - start_time

    # -------------------------------
    # Results
    # -------------------------------
    print("\nOptimization finished")
    print(f"Elapsed time: {elapsed/60:.1f} min")
    print("Best sensitivity:", study.best_value)

    best = study.best_params
    print(
        "Best parameters:\n"
        f"  I_p     = {10**best['log10_Ip']:.3e}\n"
        f"  OmegaR  = {10**best['log10_OmegaR']:.3e}"
    )

# def optimizeSystem():

#     bounds = [
#         (5, 10),     # log10(I_p)
#         (6, 11),    # log10(I_s)
#         (4, 7),     # log10(OmegaR)
#     ]
#     result = differential_evolution(
#                                         objective_wrapped,
#                                         bounds,
#                                         # strategy='best1bin',
#                                         # popsize=15,
#                                         # tol=1e-4,
#                                         # polish=False,  # IMPORTANT: polish only if smooth
#                                         updating='deferred',
#                                         workers=-1     # parallel
#                                     )
    
#     print(result.x, result.fun)
    
#     # study = optuna.create_study(direction="minimize")
#     # study.optimize(objective, n_trials=300)
    
#     # print(study.best_params, study.best_value)
    
    

    
def solvePopulationDetuningLineplot(saveBool = False, 
                                    OmegaR = OmegaROptRadps,          
                                    I_p = I_pOpt,
                                    I_s = I_sOpt,
                                    bw = 0.01e9*2*np.pi,
                                    Npoints = 101,
                                    fastBool = False,
                                    bool1042 = False,
                                    plotBool = True,
                                    returnBool = False,
                                    returnPops = False):
    """Sweep microwave detuning at fixed drive powers and plot/return the resulting m_7 (IR-probed) line shape."""
    # detuning
    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
    kappaTotal = np.zeros(Npoints)
    p7 = np.zeros(Npoints)
    pops = np.zeros([11, Npoints])
    for i, det in enumerate(detunings):
        pop = nv_steady_state_default(
            Omega = OmegaR,
            I_s = I_s,
            I_p = I_p,
            detuning = (det),
            zeeman = zeeman,
            steadystatebool=True,
            bool1042 = bool1042,
        )
        p7[i] = (pop[7])
        pops[:, i] = pop
    
    peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), p7)
    
    delta = 2*np.pi*2.16e6  # example shift
    dx = detunings[1] - detunings[0]
    shift_pts = int(delta / dx)
    
    x = detunings
    y = p7/p7.min()-1
    
    y_shift_plus  = np.roll(y, -shift_pts)
    y_shift_minus = np.roll(y,  shift_pts)
    y_shift_pplus  = np.roll(y, -2*shift_pts)
    y_shift_mminus = np.roll(y,  2*shift_pts)

    y_total = (3*y)/3
    
    if plotBool:
        plt.figure(figsize=(4,4))
        plt.plot(detunings/1e9/(2*np.pi), y_total)
        # plt.plot(detunings/1e9/(2*np.pi), kappaTotal/np.min(kappaTotal))
        print('Max p7: ', np.max(p7), '   ', np.max(p7)/np.min(p7))
        print('Min p7: ', np.min(p7))
        print('FWHM: ', FWHM)
        plt.ylabel(r'change in population (au)')
        plt.xlabel(r'microwave detuning (GHz)')
        plt.show()
        
    if returnBool:
        return pops, detunings - zeeman
    if returnPops:
        return np.min(p7), np.max(p7)
    
def solvePopulationDetuningLineplotPL(saveBool = False, 
                                    OmegaR = gamma_phi,          
                                    I_p = 1e7,
                                    I_s = 1e12,
                                    bw = 0.01e9*2*np.pi,
                                    Npoints = 101,
                                    fastBool = False,
                                    bool1042 = False,
                                    plotBool = True,
                                    returnBool = False):
    """Sweep microwave detuning at fixed drive powers and plot the resulting NV- PL rate line shape."""
    # detuning
    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
    kappaTotal = np.zeros(Npoints)
    PLRate = np.zeros(Npoints)
    pops = np.zeros([11, Npoints])
    for i, det in enumerate(detunings):
        pop = nv_steady_state_default(
            Omega = OmegaR,
            I_s = I_s,
            I_p = I_p,
            detuning = (det),
            zeeman = zeeman,
            steadystatebool=True,
            bool1042 = bool1042,
        )
        PLRate[i] = (pop[3] + pop[4]+pop[5])*gamma_eg
        pops[:, i] = pop
    
    # peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), PLRate)
        
    if plotBool:
        plt.figure(figsize=(4,4))
        plt.plot(detunings/1e9/(2*np.pi), PLRate/np.min(PLRate))
        plt.plot(detunings/1e9/(2*np.pi), kappaTotal/np.min(kappaTotal))
        print('Max Rate: ', np.max(PLRate), '   ', np.max(PLRate)/np.min(PLRate))
        print('Min Rate: ', np.min(PLRate))
        # print('FWHM: ', FWHM)
        plt.ylabel(r'change in population (au)')
        plt.xlabel(r'microwave detuning (GHz)')
        plt.show()
        
    if returnBool:
        return pops

# def solvePopulationDetuningLineplot(saveBool = False, 
#                                     OmegaR = gamma_phi,          
#                                     I_p = 1e7,
#                                     I_s = 1e12,
#                                     bw = 0.01e9*2*np.pi,
#                                     Npoints = 101):
#     # detuning
#     detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
#     p7 = np.zeros(Npoints)
#     for i, det in enumerate(detunings):
       
#         pop = nv_steady_state(
#                                         Omega = OmegaR, #rad/s
#                                         I_s = I_s,   #rad/s
#                                         I_p = I_p, #W/m^2
#                                         gamma_phi = gamma_phi,   #Hz
#                                         detuning = (det), #rad/s
#                                         zeeman   = zeeman, #rad/s
#                                         steadystatebool=True, 
#                                         tEnd = 2e-6,
#                                         gamma_eg=gamma_eg,
#                                         gamma_isc_0=gamma_isc_0,
#                                         gamma_isc_1=gamma_isc_1,
#                                         gamma_isc_g0=gamma_isc_g0,
#                                         gamma_isc_g1=gamma_isc_g1,
#                                         gamma_s=gamma_s,
#                                         gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
#                                         gamma_NV0_em = gamma_NV0_em, #k_75
#                                         gamma_NV0_mg = gamma_NV0_mg,   #k_56
#                                         sigma_p = sigma_p, # m^%-2
#                                         sigma_s = sigma_s, # m^%-2
#                                         lamb_s = lamb_s,   # IR probe
#                                         lamb_p = lamb_p,    # green pump
#                                         c = c_const, #m/s
#                                         h = h      # Planck constant [J·s]
#                                         )
#         p7[i] = (pop[7])
    
#     peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), p7)
        
#     plt.figure(figsize=(4,4))
#     plt.plot(detunings/1e9/(2*np.pi), p7/np.min(p7))
#     plt.ylabel(r'change in population (au)')
#     plt.xlabel(r'microwave detuning (GHz)')
#     plt.show()


def solvePopulationDetuningHyperfineLineplot(saveBool = False):
    """Sweep detuning across the hyperfine triplet (+/- hyperfine split lines) and plot the summed m_7 signal."""
    # detuning
    OmegaR = gamma_phi
    I_p = 1e7
    I_s = 1e12
    detunings = np.linspace(0.095e9, 0.105e9, 200)
    p7 = np.array([])
    for det in detunings:
        pophf1 = nv_steady_state(
                                        Omega = OmegaR, #rad/s
                                        I_s = I_s,   #rad/s
                                        I_p = I_p, #W/m^2
                                        gamma_phi = gamma_phi,   #Hz
                                        detuning = (det-hyperfine)*2*np.pi, #rad/s
                                        zeeman   = zeeman, #rad/s
                                        steadystatebool=True, 
                                        tEnd = 2e-6,
                                        gamma_eg=gamma_eg,
                                        gamma_isc_0=gamma_isc_0,
                                        gamma_isc_1=gamma_isc_1,
                                        gamma_isc_g0=gamma_isc_g0,
                                        gamma_isc_g1=gamma_isc_g1,
                                        gamma_s=gamma_s,
                                        gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
                                        gamma_NV0_em = gamma_NV0_em, #k_75
                                        gamma_NV0_mg = gamma_NV0_mg,   #k_56
                                        sigma_p = sigma_p, # m^%-2
                                        sigma_s = sigma_s, # m^%-2
                                        lamb_s = lamb_s,   # IR probe
                                        lamb_p = lamb_p,    # green pump
                                        c = c_const, #m/s
                                        h = h,      # Planck constant [J·s]
                                        paramsRates = paramsRatesOnes,
                                        paramsRatesReal = paramsRatesReal
                                        )
        pophf2 = nv_steady_state(
                                        Omega = OmegaR, #rad/s
                                        I_s = I_s,   #rad/s
                                        I_p = I_p, #W/m^2
                                        gamma_phi = gamma_phi,   #Hz
                                        detuning = (det)*2*np.pi, #rad/s
                                        zeeman   = zeeman, #rad/s
                                        steadystatebool=True, 
                                        tEnd = 2e-6,
                                        gamma_eg=gamma_eg,
                                        gamma_isc_0=gamma_isc_0,
                                        gamma_isc_1=gamma_isc_1,
                                        gamma_isc_g0=gamma_isc_g0,
                                        gamma_isc_g1=gamma_isc_g1,
                                        gamma_s=gamma_s,
                                        gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
                                        gamma_NV0_em = gamma_NV0_em, #k_75
                                        gamma_NV0_mg = gamma_NV0_mg,   #k_56
                                        sigma_p = sigma_p, # m^%-2
                                        sigma_s = sigma_s, # m^%-2
                                        lamb_s = lamb_s,   # IR probe
                                        lamb_p = lamb_p,    # green pump
                                        c = c_const, #m/s
                                        h = h,      # Planck constant [J·s]
                                        paramsRates = paramsRatesOnes,
                                        paramsRatesReal = paramsRatesReal
                                        )
        pophf3 = nv_steady_state(
                                        Omega = OmegaR, #rad/s
                                        I_s = I_s,   #rad/s
                                        I_p = I_p, #W/m^2
                                        gamma_phi = gamma_phi,   #Hz
                                        detuning = (det+hyperfine)*2*np.pi, #rad/s
                                        zeeman   = zeeman, #rad/s
                                        steadystatebool=True, 
                                        tEnd = 2e-6,
                                        gamma_eg=gamma_eg,
                                        gamma_isc_0=gamma_isc_0,
                                        gamma_isc_1=gamma_isc_1,
                                        gamma_isc_g0=gamma_isc_g0,
                                        gamma_isc_g1=gamma_isc_g1,
                                        gamma_s=gamma_s,
                                        gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
                                        gamma_NV0_em = gamma_NV0_em, #k_75
                                        gamma_NV0_mg = gamma_NV0_mg,   #k_56
                                        sigma_p = sigma_p, # m^%-2
                                        sigma_s = sigma_s, # m^%-2
                                        lamb_s = lamb_s,   # IR probe
                                        lamb_p = lamb_p,    # green pump
                                        c = c_const, #m/s
                                        h = h,      # Planck constant [J·s]
                                        paramsRates = paramsRatesOnes,
                                        paramsRatesReal = paramsRatesReal
                                        )
        p7 = np.append(p7,pophf1[7] + pophf2[7] + pophf3[7])
    
    plt.figure(figsize=(6,6))
    plt.plot(detunings/1e9, p7/np.min(p7))
    plt.show()
    
    # #%% detuning and power Sweep
    
    # N_IR = 5
    # N_green = 5
    # N_MW = 5
    # Ndets = 100;
    # I_IR = np.logspace(6, 16, N_IR)
    # I_green = np.logspace(6, 12, N_green)
    # Omega = np.logspace(2, 7, N_MW)
    # omega = 0.5e6
    # detunings = np.linspace(-0.2e9, 0.2e9, Ndets)
    
    # p7Green = np.zeros([N_green, Ndets])
    # p7IR = np.zeros([N_green, Ndets])
    # p7MW = np.zeros([N_green, Ndets])
    
    # for i, I_p in enumerate(I_green):
    #     for j, det in enumerate(detunings):
    #         populations =         nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = omega*2*np.pi, I_p = I_p)
    #         populationsDetuning = nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = 0, I_p = I_p)
    #         p7Green[i,j] = (populations[7] - populationsDetuning[7])
    #     p7Green[i,:] /= np.max(p7Green[i,:])
    
    
    # for i, I_s in enumerate(I_IR):
    #     for j, det in enumerate(detunings):
    #         populations =         nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = I_s, Omega = omega*2*np.pi, I_p = 1e8)
    #         populationsDetuning = nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = I_s, Omega = 0, I_p = 1e8)
    #         p7IR[i,j] = (populations[7] - populationsDetuning[7])
    #     p7IR[i,:] /= np.max(p7IR[i,:])
    
    
    # for i, Om in enumerate(Omega):
    #     for j, det in enumerate(detunings):
    #         populations =         nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = Om*2*np.pi, I_p = 1e8)
    #         populationsDetuning = nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = 0, I_p = 1e8)
    #         p7MW[i,j] = (populations[7] - populationsDetuning[7])
    #     p7MW[i,:] /= np.max(p7MW[i,:])
    
    # #%%
    # fig, ax = plot_custom(
    #     PRQuantumWidthFull*0.6, 0.35*PRQuantumWidthFull, 
    #     [r"$I_{\text{P}}$ (W/m$^2$)", r"$I_{\text{P}}$ (W/m$^2$)", r"$I_{\text{P}}$ (W/m$^2$)"], 
    #     r'microwave detuning (MHz)',
    #     labelPadX = 3, labelPadY = 8, commonY = True, commonX = False, wSpaceSetting=0,
    #     spineColor = 'black', tickColor = 'black', textColor = 'black',
    #     nrow = 1, ncol = 3, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
    #     widthBool =False
    # )
    
    # X, Y = np.meshgrid(np.log10(I_green), detunings/1e6)
    # im = ax[0].pcolormesh(X, Y, np.transpose(np.log10(p7Green)), shading = 'auto')
    # X, Y = np.meshgrid(np.log10(I_IR), detunings/1e6)
    # im = ax[1].pcolormesh(X, Y, np.transpose(np.log10(p7IR)), shading = 'auto')
    # X, Y = np.meshgrid(np.log10(Omega), detunings/1e6)
    # im = ax[2].pcolormesh(X, Y, np.transpose(np.log10(p7MW)), shading = 'auto')
    # # im = ax.contourf(X, Y, p7, 100)
    # plt.tight_layout()
    
    # fig.subplots_adjust(wspace=0.1)
    
    # # cbar = plt.colorbar(im, orientation = 'horizontal', location = 'top')
    # # cbar.set_label(r'$m_7^{\text{(on)}} - m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)
    
    # plt.show()

def _solvePopulationIR_worker(I_s, OmegaR, I_p, bw, Ndets):
    """Compute (p7On, p7Off, nu, C) for one I_s point of solvePopulationIR.

    Module-level (not nested) so it can be pickled and dispatched to a
    ProcessPoolExecutor worker process -- see the parallelized loop in
    solvePopulationIR. Each call starts from the function's own `bw` rather
    than inheriting a previously-converged bw from a prior point (as the
    original serial loop did), since points now run independently/out of
    order; this can mean a few more bisection attempts per point but does
    not change the converged nu/C values.
    """
    popOn = nv_steady_state_default(
        Omega=OmegaR,
        I_s=I_s,
        I_p=I_p,
        detuning=detuning,
        zeeman=zeeman,
        steadystatebool=True,
    )
    popOff = nv_steady_state_default(
        Omega=OmegaR*0,
        I_s=I_s,
        I_p=I_p,
        detuning=detuning,
        zeeman=zeeman,
        steadystatebool=True,
    )
    p7On_i = popOn[7]
    p7Off_i = popOff[7]
    nu_i = 0.0
    C_i = 0.0

    for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
        detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
        Cdet = np.zeros(Ndets)
        for k, det in enumerate(detunings):
            popDet = nv_steady_state_default(
                Omega=OmegaR,
                I_s=I_s,
                I_p=I_p,
                detuning=det,
                zeeman=zeeman,
            )
            Cdet[k] = (popDet[7]) - (p7Off_i)

        peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
        if np.size(FWHM) < 1:
            print('FWHM fitting didnt work')
        elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
            nu_i = FWHM[0]*1e6
            C_i = np.max(Cdet)
            break
        elif FWHM >= bw/2e9/2/np.pi:
            bw = bw*2
            continue
        else:
            bw = bw/1.6
            continue

    return p7On_i, p7Off_i, nu_i, C_i


def solvePopulationIR(saveBool = False,
                            N_IR = 10,
                            N_OmegaR = 120,
                            OmegaR = 1e6,
                            I_smin = 7,
                            I_smax = 11,
                            I_p = 1e8,
                            bw = 0.005e9*2*np.pi,
                            Ndets = 51,
                            plotBool = True,
                            logBoolI = False,
                            parallelBool = True,
                            maxWorkers = None):
    """Sweep IR probe intensity I_s; fit contrast/linewidth vs power via an adaptive detuning scan at each point.

    Each I_s point is independent (a fresh nv_steady_state solve + bisection
    search), so when parallelBool=True the points are dispatched across
    processes via ProcessPoolExecutor instead of run serially. Set
    parallelBool=False to use the original serial loop (e.g. for comparison
    or debugging).
    """


    if logBoolI:
        I_IR = np.logspace(I_smin, I_smax, N_IR)
    else:
        I_IR = np.linspace(10**I_smin, 10**I_smax, N_IR)
    p7 = np.zeros(N_IR)
    p7On = np.zeros(N_IR)
    p7Off = np.zeros(N_IR)
    nu = np.zeros(N_IR)
    C = np.zeros(N_IR)

    if parallelBool:
        with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            futures = [
                executor.submit(_solvePopulationIR_worker, I_s, OmegaR, I_p, bw, Ndets)
                for I_s in I_IR
            ]
            for i, future in enumerate(tqdm(futures, desc="IR intensity (parallel)")):
                p7On[i], p7Off[i], nu[i], C[i] = future.result()
                p7[i] = p7On[i]/p7Off[i]
    else:
        for i, I_s in enumerate(tqdm(I_IR, desc="IR intensity")):
            p7On[i], p7Off[i], nu[i], C[i] = _solvePopulationIR_worker(I_s, OmegaR, I_p, bw, Ndets)
            p7[i] = p7On[i]/p7Off[i]

    # C = C/np.max(C)#
    power = I_IR/np.max(I_IR)*31.7050  # 31.7050 calibration factor
    if plotBool:
        fig, ax = plot_custom(
           PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull,  
            r"$I_{\text{S}}$ (kW/cm$^2$)", 
            r'$\Omega_{\text{R}}/2\pi$ (MHz)',
            labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
            spineColor = 'black', tickColor = 'black', textColor = 'black',
            nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
            widthBool =False
        )
        
        # fig = plt.figure(figsize=(PRQuantumWidthFull*0.3, 0.25*PRQuantumWidthFull))
        # ax = plt.gca()
        ax.set_xlabel(r"$I_{\text{S}}$ (kW/cm$^2$)", fontsize = labelfontsize)
        ax.set_ylabel(r'signal', fontsize = labelfontsize)
            
        ax.plot(I_IR, C*np.sqrt(power)/nu)
        plt.tight_layout()
        
        plt.show()
    else:
        return C, power, nu, I_IR, p7On, p7Off
    
    
def solvePopulationGreen(saveBool = False,
                            N_Green = 11,
                            OmegaR = 1e6,
                            I_pmin = 7,
                            I_pmax = 11,
                            I_s = 1e10,
                            OmegaROpt = 0,
                            I_sOpt = 0,
                            bw = 0.005e9*2*np.pi,
                            Ndets = 51,
                            plotBool = True):
    """Sweep green pump intensity I_p; fit contrast/linewidth vs power via an adaptive detuning scan at each point."""
    
    
    
    # I_g = np.logspace(I_pmin, I_pmax, N_Green)
    I_g = np.linspace(10**I_pmin, 10**I_pmax, N_Green)
    p7 = np.zeros(N_Green)
    p7On = np.zeros(N_Green)
    p7Off = np.zeros(N_Green)
    nu = np.zeros(N_Green)
    C = np.zeros(N_Green)
    for i, I_p in enumerate(tqdm(I_g, desc="Green intensity")):
        populations =         nv_steady_state_default(
            Omega = OmegaR,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
            tEnd = 2e-6,
        )
        populationsDetuning = nv_steady_state_default(
            Omega = OmegaR*0,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
            tEnd = 2e-6,
        )
        p7On[i] = (populations[7])
        p7Off[i] = (populationsDetuning[7])
        p7[i] = p7On[i]-p7Off[i]
    
        for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
            detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
            Cdet = np.zeros(Ndets)
            for k, det in enumerate(detunings):
                popDet =    nv_steady_state_default(
                    Omega = OmegaR,
                    I_s = I_s,
                    I_p = I_p,
                    detuning = det,
                    zeeman = zeeman,
                )
                Cdet[k] = (popDet[7])/(p7Off[i]+ p7On[i])*2
                
            
            peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
            # plt.plot(detunings/1e9/(2*np.pi),Cdet)
            # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
            # plt.show()
            if np.size(FWHM) < 1:
                print('FWHM fitting didnt work')
            # elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.01:
            #     bw = bw*2
            #     continue
            elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
                nu[i] = FWHM[0]*1e6
                C[i] = p7[i]
                break
            elif FWHM >= bw/2e9/2/np.pi:
                bw = bw*2
                continue
            else:
                bw = bw/3
                continue
    
    C = C/np.max(C)#*6.7e-4  
    if plotBool:
        fig, ax = plot_custom(
           PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull,  
            r"$I_{\text{P}}$ (W/m$^2$)", 
            r'$\Omega_{\text{R}}/2\pi$ (MHz)',
            labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
            spineColor = 'black', tickColor = 'black', textColor = 'black',
            nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
            widthBool =False
        )
        
        # fig = plt.figure(figsize=(PRQuantumWidthFull*0.3, 0.25*PRQuantumWidthFull))
        # ax = plt.gca()
        # ax.set_xlabel(r"$I_{\text{S}}$ (kW/cm$^2$)", fontsize = labelfontsize)
        ax.set_ylabel(r'signal', fontsize = labelfontsize)
            
        # power = I_IR/np.max(I_IR)*31.7050
        # ax.plot(I_g, C/nu)
        ax.semilogx(I_g, C/nu)
        plt.tight_layout()
        
      
        plt.show()
    else:
        return I_g, C, nu
    
def solvePopulationOmegaR(saveBool = False,
                            N_OmegaR = 10,
                            OmegaRmin = 7,
                            OmegaRmax = 11,
                            I_p = 1e8,
                            I_s = 1e10,
                            bw = 0.02e9*2*np.pi,
                            Ndets = 201,
                            plotBool = True):
    """Sweep microwave Rabi frequency OmegaR; fit contrast/linewidth via an adaptive detuning scan at each point."""
    
    
    
    OmegaRArr = np.linspace(10**OmegaRmin, 10**OmegaRmax, N_OmegaR)
    p7 = np.zeros(N_OmegaR)
    p7On = np.zeros(N_OmegaR)
    p7Off = np.zeros(N_OmegaR)
    nu = np.zeros(N_OmegaR)
    C = np.zeros(N_OmegaR)
    for i, OmegaR in enumerate(tqdm(OmegaRArr, desc="Rabi freq")):
        populations =         nv_steady_state_default(
            Omega = OmegaR,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
            tEnd = 2e-6,
        )
        populationsDetuning = nv_steady_state_default(
            Omega = OmegaR*0,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
            tEnd = 2e-6,
        )
        p7On[i] = (populations[7])
        p7Off[i] = (populationsDetuning[7])
        p7[i] = p7On[i]-p7Off[i]
    
        for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
            detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
            Cdet = np.zeros(Ndets)
            for k, det in enumerate(detunings):
                popDet =    nv_steady_state_default(
                    Omega = OmegaR,
                    I_s = I_s,
                    I_p = I_p,
                    detuning = det,
                    zeeman = zeeman,
                )
                # Cdet[k] = (popDet[7])/(p7Off[i]+ p7On[i])*2
                Cdet[k] = popDet[7]-p7Off[i]
                
            
            peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
            # plt.plot(detunings/1e9/(2*np.pi),Cdet)
            # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
            # plt.show()
            if np.size(FWHM) < 1:
                print('FWHM fitting didnt work')
            # elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.01:
            #     bw = bw*2
            #     continue
            elif FWHM < bw/3e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
                nu[i] = FWHM[0]*1e6
                C[i] = np.max(Cdet)
                break
            elif FWHM >= bw/4e9/2/np.pi:
                bw = bw*2
                continue
            else:
                bw = bw/1.3
                continue
    
    C = C
    if plotBool:
        fig, ax = plot_custom(
           PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull,  
            r'$\Omega_{\text{R}}/2\pi$ (MHz)',
             r"$C/\nu$ (MHz$^{-1}$)", 
            labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
            spineColor = 'black', tickColor = 'black', textColor = 'black',
            nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
            widthBool =False
        )
        
        # fig = plt.figure(figsize=(PRQuantumWidthFull*0.3, 0.25*PRQuantumWidthFull))
        # ax = plt.gca()
            
        # ax.plot(OmegaRArr, C/nu*1e6)
        ax.semilogx(OmegaRArr/2/np.pi, C/nu*1e6)
        plt.tight_layout()
        
      
        plt.show()
    else:
        return OmegaRArr, C, nu

    # if saveBool:
    #     fig.savefig(
    #         (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRIR.png", dpi=300, bbox_inches="tight",transparent=True
    #         )

def solvePopulationIROmegaR(saveBool = False,
                            N_IR = 10,
                            N_OmegaR = 120,
                            OmegaRmin = 4,
                            OmegaRmax = 7,
                            I_smin = 7,
                            I_smax = 11,
                            I_p = 1e8,
                            OmegaROpt = 0,
                            I_sOpt = 0,
                            bool1042 = False):
    """2D sweep of IR intensity vs Rabi frequency; plot the m_7 on/off population contrast as a contour map."""
    
    I_IR = np.logspace(I_smin, I_smax, N_IR)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    p7 = np.zeros([N_IR, N_OmegaR])
    p7On = np.zeros([N_IR, N_OmegaR])
    p7Off = np.zeros([N_IR, N_OmegaR])
    for i, I_s in enumerate(tqdm(I_IR, desc="IRintensity")):
        for j, OmegaR in enumerate(OmegaRArr):
            populations =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            populationsDetuning = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            # p7[i,j] = (populations[7] - populationsDetuning[7])
            p7On[i,j] = (populations[7])
            p7Off[i,j] = (populationsDetuning[7])
            # p7[i,j] = (p7On[i,j])/p7Off[i,j]
            p7[i] = p7On[i]-p7Off[i]
            # p7[i,j] = (p7On[i,j] - p7Off[i,j])
            
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull,  
        r"$I_{\text{S}}$ (kW/cm$^2$)", 
        r'$\Omega_{\text{R}}/2\pi$ (MHz)',
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    # fig = plt.figure(figsize=(PRQuantumWidthFull*0.3, 0.25*PRQuantumWidthFull))
    # ax = plt.gca()
    # ax.set_xlabel(r"$I_{\text{S}}$ (kW/cm$^2$)", fontsize = labelfontsize)
    # ax.set_ylabel(r'$\Omega_{\text{R}}/2\pi$ (MHz)', fontsize = labelfontsize)
        
    X, Y = np.meshgrid(np.log10(I_IR), np.log10(OmegaRArr))
    # im = ax.pcolormesh(X-7, Y-7, np.transpose(p7), cmap = 'hot')
    im = plt.contourf(X-7, Y-6, np.transpose(p7)*100, 21, cmap = 'Greys_r')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$p_7^{\text{(on)}}-p_7^{\text{(off)}}$ ($\%$)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    ticks = [0, np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    ax.scatter(np.log10(I_sOpt)-7, np.log10(OmegaROpt)-6, marker = '*', color = 'red')
    plt.show()

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRIR.png", dpi=300, bbox_inches="tight",transparent=True
            )
        
            
# def solvePopulationGreenOmegaR(saveBool = False,
#                             N_green = 10,
#                             N_OmegaR = 10,
#                             OmegaRmin = 4,
#                             OmegaRmax = 7,
#                             I_pmin = 5,
#                             I_pmax = 8,
#                             I_s = 1e9,
#                             fastBool = False,
#                             OmegaROpt = 0,
#                             I_pOpt = 0,
#                             bool1042 = False):
    
#     I_green = np.logspace(I_pmin, I_pmax, N_green)
#     OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
#     p7 = np.zeros([N_green, N_OmegaR])
#     p7On = np.zeros([N_green, N_OmegaR])
#     p7Off = np.zeros([N_green, N_OmegaR])
#     for i, I_p in enumerate(tqdm(I_green, desc="green intensity")):
#         for j, OmegaR in enumerate(OmegaRArr):
#             if fastBool:
#                 populations =         nv_steady_state_fast(
#                                                 Omega = OmegaR, #rad/s
#                                                 I_s = I_s,   #rad/s
#                                                 I_p = I_p, #W/m^2
#                                                 gamma_phi = gamma_phi,   #Hz
#                                                 detuning = detuning, #rad/s
#                                                 zeeman   = zeeman, #rad/s
#                                                 gamma_eg=gamma_eg,
#                                                 gamma_isc_0=gamma_isc_0,
#                                                 gamma_isc_1=gamma_isc_1,
#                                                 gamma_isc_g0=gamma_isc_g0,
#                                                 gamma_isc_g1=gamma_isc_g1,
#                                                 gamma_s=gamma_s,
#                                                 gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
#                                                 gamma_NV0_em = gamma_NV0_em, #k_75
#                                                 gamma_NV0_mg = gamma_NV0_mg,   #k_56
#                                                 sigma_p = sigma_p, # m^%-2
#                                                 sigma_s = sigma_s, # m^%-2
#                                                 lamb_s = lamb_s,   # IR probe
#                                                 lamb_p = lamb_p,    # green pump
#                                                 c = c_const, #m/s
#                                                 h = h,      # Planck constant [J·s]
#                                                 paramsRates = paramsRatesOnes,
#                                                 paramsRatesReal = paramsRatesReal
#                                                 )


#                 populationsDetuning = nv_steady_state_fast(
#                                                 Omega = OmegaR*0, #rad/s
#                                                 I_s = I_s,   #rad/s
#                                                 I_p = I_p, #W/m^2
#                                                 gamma_phi = gamma_phi,   #Hz
#                                                 detuning = detuning, #rad/s
#                                                 zeeman   = zeeman, #rad/s
#                                                 gamma_eg=gamma_eg,
#                                                 gamma_isc_0=gamma_isc_0,
#                                                 gamma_isc_1=gamma_isc_1,
#                                                 gamma_isc_g0=gamma_isc_g0,
#                                                 gamma_isc_g1=gamma_isc_g1,
#                                                 gamma_s=gamma_s,
#                                                 gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
#                                                 gamma_NV0_em = gamma_NV0_em, #k_75
#                                                 gamma_NV0_mg = gamma_NV0_mg,   #k_56
#                                                 sigma_p = sigma_p, # m^%-2
#                                                 sigma_s = sigma_s, # m^%-2
#                                                 lamb_s = lamb_s,   # IR probe
#                                                 lamb_p = lamb_p,    # green pump
#                                                 c = c_const, #m/s
#                                                 h = h,      # Planck constant [J·s]
#                                                 paramsRates = paramsRatesOnes,
#                                                 paramsRatesReal = paramsRatesReal
#                                                 )
#             else:
#                 populations =         nv_steady_state(
#                                                 Omega = OmegaR, #rad/s
#                                                 I_s = I_s,   #rad/s
#                                                 I_p = I_p, #W/m^2
#                                                 gamma_phi = gamma_phi,   #Hz
#                                                 detuning = detuning, #rad/s
#                                                 zeeman   = zeeman, #rad/s
#                                                 steadystatebool=True, 
#                                                 tEnd = 2e-6,
#                                                 gamma_eg=gamma_eg,
#                                                 gamma_isc_0=gamma_isc_0,
#                                                 gamma_isc_1=gamma_isc_1,
#                                                 gamma_isc_g0=gamma_isc_g0,
#                                                 gamma_isc_g1=gamma_isc_g1,
#                                                 gamma_s=gamma_s,
#                                                 gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
#                                                 gamma_NV0_em = gamma_NV0_em, #k_75
#                                                 gamma_NV0_mg = gamma_NV0_mg,   #k_56
#                                                 sigma_p = sigma_p, # m^%-2
#                                                 sigma_s = sigma_s, # m^%-2
#                                                 lamb_s = lamb_s,   # IR probe
#                                                 lamb_p = lamb_p,    # green pump
#                                                 c = c_const, #m/s
#                                                 h = h,      # Planck constant [J·s]
#                                                 paramsRates = paramsRatesOnes,
#                                                 paramsRatesReal = paramsRatesReal,
#                                                 bool1042 = bool1042
#                                                 )
#                 populationsDetuning = nv_steady_state(
#                                                 Omega = OmegaR*0, #rad/s
#                                                 I_s = I_s,   #rad/s
#                                                 I_p = I_p, #W/m^2
#                                                 gamma_phi = gamma_phi,   #Hz
#                                                 detuning = detuning, #rad/s
#                                                 zeeman   = zeeman, #rad/s
#                                                 steadystatebool=True, 
#                                                 tEnd = 2e-6,
#                                                 gamma_eg=gamma_eg,
#                                                 gamma_isc_0=gamma_isc_0,
#                                                 gamma_isc_1=gamma_isc_1,
#                                                 gamma_isc_g0=gamma_isc_g0,
#                                                 gamma_isc_g1=gamma_isc_g1,
#                                                 gamma_s=gamma_s,
#                                                 gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
#                                                 gamma_NV0_em = gamma_NV0_em, #k_75
#                                                 gamma_NV0_mg = gamma_NV0_mg,   #k_56
#                                                 sigma_p = sigma_p, # m^%-2
#                                                 sigma_s = sigma_s, # m^%-2
#                                                 lamb_s = lamb_s,   # IR probe
#                                                 lamb_p = lamb_p,    # green pump
#                                                 c = c_const, #m/s
#                                                 h = h,      # Planck constant [J·s]
#                                                 paramsRates = paramsRatesOnes,
#                                                 paramsRatesReal = paramsRatesReal,
#                                                 bool1042 = bool1042
#                                                 )
#             # p7[i,j] = (populations[7] - populationsDetuning[7])
#             p7On[i,j] = (populations[7])
#             p7Off[i,j] = (populationsDetuning[7])
#             # p7[i,j] = (p7On[i,j])/p7Off[i,j]
#             p7[i,j] = (p7On[i,j] - p7Off[i,j])
    
#     fig, ax = plot_custom(
#        PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
#        r"$I_{\text{p}}$ (kW/cm$^2$)", 
#        r'$\Omega_{\text{R}}/2\pi$ (MHz)',
#        labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
#        spineColor = 'black', tickColor = 'black', textColor = 'black',
#        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
#        widthBool =False
#     )
    
#     X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
#     im = ax.contourf(X-7, Y-6, np.transpose(p7)*100, 21, cmap = 'Greys_r')
#     plt.tight_layout()
    
#     cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
#     cbar.set_label(r'$p_7^{\text{(on)}}-p_7^{\text{(off)}}$ ($\%$)', labelpad=2, fontsize = labelfontsize)
#     yticks = ax.get_yticks()
#     # yticks = [-2, -1, 0, 1]
#     ax.set_xticks([-1, 1, 3])
#     xticks = ax.get_xticks()
#     ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
#     ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
#     # ticks = [0, np.max(p7)]
#     # cbar.set_ticks(ticks)
#     # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
#     ax.scatter(np.log10(I_pOpt)-7, np.log10(OmegaROpt)-6, marker = '*', color = 'red')
#     plt.show()

#     if saveBool:
#         fig.savefig(
#             (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRGreen.png", dpi=300, bbox_inches="tight",transparent=True
#             )
        
def solveNVIRAbs_GreenOmegaR(saveBool = False,
                            N_green = 10,
                            N_OmegaR = 10,
                            OmegaRmin = 4,
                            OmegaRmax = 7,
                            I_pmin = 5,
                            I_pmax = 8,
                            fastBool = False,
                            OmegaROpt = 0,
                            I_pOpt = 0,
                            I_sOpt = 0,
                            bw = 0.01e9*2*np.pi,
                            Ndets = 51):
    """2D sweep of green power vs Rabi frequency; compute IR-absorption sensitivity via an adaptive detuning fit and save to IRAbsfits.npz."""
    
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    C = np.zeros([N_green, N_OmegaR])
    nu = np.zeros([N_green, N_OmegaR])
    R_on = np.zeros([N_green, N_OmegaR])
    R_off = np.zeros([N_green, N_OmegaR])
    for i, I_p in enumerate(tqdm(I_green, desc="Green intensity")):
        for j, OmegaR in enumerate(tqdm(OmegaRArr, desc="OmegaR", leave=False)):
            popOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_sOpt,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            popOff = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = I_sOpt,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
                
            # The absorption rate is compared to the input photon rate of a focused beam
            R_photon_gaussian = I_sOpt*np.pi*lamb_s**3/(2*h*c_const)
            
            R_on[i,j] = (popOn[7])*I_sOpt*sigma_s/(h*c_const/lamb_s)
            R_off[i,j] = (popOff[7])*I_sOpt*sigma_s/(h*c_const/lamb_s)
            C[i,j] = (R_on[i,j]-R_off[i,j])/(R_photon_gaussian-R_off[i,j])

            for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
                detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
                Cdet = np.zeros(Ndets)
                for k, det in enumerate(detunings):
                    popDet =    nv_steady_state_default(
                        Omega = OmegaR,
                        I_s = I_sOpt,
                        I_p = I_p,
                        detuning = det,
                        zeeman = zeeman,
                    )
                    Cdet[k] = -(R_off[i,j] - ((popDet[7])*I_sOpt*sigma_s/(h*c_const/lamb_s)))/R_off[i,j]
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
                plt.show()
                if np.size(FWHM) < 1:
                    print('FWHM fitting didnt work')
                    break
                elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.9:
                    bw = bw*2
                    continue
                elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
                    nu[i,j] = FWHM[0]*1e9
                    break
                elif FWHM >= bw/2:
                    bw = bw*2
                    continue
                else:
                    bw = bw/3
                    continue
            
            R_on[i,j] = R_photon_gaussian-R_on[i,j]
            R_off[i,j] = R_photon_gaussian-R_off[i,j]
                    
            
            
            
    eta = calcSensitivityCW_countrate(nu, C, R_off )
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
       r"$I_{\text{p}}$ (kW/cm$^2$)", 
       r'$\Omega_{\text{R}}/2\pi$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    min_index = np.argmin(eta)
    row, col = np.unravel_index(min_index, eta.shape)
    
    X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
    im = ax.contourf(X-7, Y-6, np.transpose(np.log10(eta)), 21, cmap = 'Greys_r')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$C$ (au)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    # yticks = [-2, -1, 0, 1]
    ax.set_xticks([-2,-1,0,1])
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    # ticks = [0, np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    ax.scatter(np.log10(I_green[col])-7, np.log10(OmegaRArr[row])-6, marker = '*', color = 'red')
    plt.show()
    
    np.savez(os.path.join(_REFFILES_DIR, "IRAbsfits.npz"),
         nu=nu,
         C=C,
         I_green=I_green,
         OmegaRArr=OmegaRArr,
         eta = eta,
         R_off = R_off,
         R_on = R_on)
    
    print('eta ', np.min(eta))

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRGreen.png", dpi=300, bbox_inches="tight",transparent=True
            )
        

               
def solveNVPL_GreenOmegaR(saveBool = False,
                            N_green = 10,
                            N_OmegaR = 10,
                            OmegaRmin = 6,
                            OmegaRmax = 8,
                            I_pmin = 8,
                            I_pmax = 10,
                            fastBool = False,
                            OmegaROpt = 0,
                            I_pOpt = 0,
                            bw = 0.01e9*2*np.pi,
                            Ndets = 51):
    """2D sweep of green power vs Rabi frequency; compute NV- PL (states 3-5) sensitivity via an adaptive detuning fit and save to IRPLfits.npz."""
    
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    C = np.zeros([N_green, N_OmegaR])
    nu = np.zeros([N_green, N_OmegaR])
    R_on = np.zeros([N_green, N_OmegaR])
    R_off = np.zeros([N_green, N_OmegaR])
    for i, I_p in enumerate(tqdm(I_green, desc="Green intensity")):
        for j, OmegaR in enumerate(tqdm(OmegaRArr, desc="OmegaR", leave=False)):
            popOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = 0,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            popOff = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = 0,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
                
            R_on[i,j] = (popOn[3] + popOn[4] + popOn[5])#*gamma_eg
            R_off[i,j] = (popOff[3] + popOff[4] + popOff[5])#*gamma_eg
            C[i,j] = (R_off[i,j]-R_on[i,j])/R_off[i,j] 

            for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
                detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
                Cdet = np.zeros(Ndets)
                for k, det in enumerate(detunings):
                    popDet =    nv_steady_state_default(
                        Omega = OmegaR,
                        I_s = 0,
                        I_p = I_p,
                        detuning = det,
                        zeeman = zeeman,
                    )
                    Cdet[k] = (R_off[i,j] - (popDet[3] + popDet[4] + popDet[5]))/R_off[i,j]
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                # plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
                # plt.show()
                # if np.size(FWHM) < 1:
                #     print('FWHM fitting didnt work')
                # # elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.01:
                # #     bw = bw*2
                # #     continue
                # elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
                nu[i] = FWHM[0]*1e6
                C[i,j] = np.max(Cdet)
                break
                #     break
                # elif FWHM >= bw/2e9/2/np.pi:
                #     bw = bw*2
                #     continue
                # else:
                #     bw = bw/1.6
                #     continue
                    
                    
            
            
            
    eta = calcSensitivityCW_countrate(nu, C, R_off)
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
       r"$I_{\text{p}}$ (kW/cm$^2$)", 
       r'$\Omega_{\text{R}}/2\pi$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    min_index = np.argmin(eta)
    row, col = np.unravel_index(min_index, eta.shape)
    
    X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
    im = ax.contourf(X-7, Y-6, np.transpose(C), 20, cmap = 'viridis')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$C$ (au)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    # yticks = [-2, -1, 0, 1]
    ax.set_xticks([-2,-1,0,1])
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    # ticks = [0, np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    ax.scatter(np.log10(I_green[col])-7, np.log10(OmegaRArr[row])-6, marker = '*', color = 'red')
    plt.show()
    
    np.savez(os.path.join(_REFFILES_DIR, "IRPLfits.npz"),
         nu=nu,
         C=C,
         I_green=I_green,
         OmegaRArr=OmegaRArr,
         eta = eta,
         R_off = R_off,
         R_on = R_on)

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRGreen.png", dpi=300, bbox_inches="tight",transparent=True
            )
        
def solveNV0PL_GreenOmegaR(saveBool = False,
                            N_green = 10,
                            N_OmegaR = 10,
                            OmegaRmin = 4,
                            OmegaRmax = 7,
                            I_pmin = 5,
                            I_pmax = 8,
                            fastBool = False,
                            OmegaROpt = 0,
                            I_pOpt = 0,
                            bw = 0.01e9*2*np.pi,
                            Ndets = 51):
    """2D sweep of green power vs Rabi frequency; compute NV0 PL (state 10) sensitivity via an adaptive detuning fit and save to NV0PLfits.npz."""
    
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    C = np.zeros([N_green, N_OmegaR])
    nu = np.zeros([N_green, N_OmegaR])
    R_on = np.zeros([N_green, N_OmegaR])
    R_off = np.zeros([N_green, N_OmegaR])
    for i, I_p in enumerate(tqdm(I_green, desc="Green intensity")):
        for j, OmegaR in enumerate(tqdm(OmegaRArr, desc="OmegaR", leave=False)):
            popOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = 0,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            popOff = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = 0,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
                
            R_on[i,j] = (popOn[10])*gamma_NV0_eg
            R_off[i,j] = (popOff[10])*gamma_NV0_eg
            C[i,j] = (R_off[i,j]-R_on[i,j])/R_off[i,j]

            for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
                detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
                Cdet = np.zeros(Ndets)
                for k, det in enumerate(detunings):
                    popDet =    nv_steady_state_default(
                        Omega = OmegaR,
                        I_s = 0,
                        I_p = I_p,
                        detuning = det,
                        zeeman = zeeman,
                    )
                    Cdet[k] = (R_off[i,j] - (popDet[10])*gamma_NV0_eg)/R_off[i,j]
                    
                
                if Cdet[int(np.ceil(Ndets/2))] < Cdet[0]:
                    Cdet = -Cdet
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
                plt.show()
                if np.size(FWHM) < 1:
                    print('FWHM fitting didnt work')
                elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.8:
                    bw = bw*2
                    continue
                elif FWHM < bw/1e9/(2*np.pi)/2 and FWHM > bw/7/1e9/(2*np.pi):
                    nu[i,j] = FWHM[0]*1e9
                    break
                elif FWHM >= bw/2:
                    bw = bw*2
                    continue
                else:
                    bw = bw/3
                    continue
                    
                    
            
            
            
    eta = calcSensitivityCW_countrate(nu, C, R_off)
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
       r"$I_{\text{p}}$ (kW/cm$^2$)", 
       r'$\Omega_{\text{R}}/2\pi$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    min_index = np.argmin(eta)
    row, col = np.unravel_index(min_index, eta.shape)
    
    X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
    im = ax.contourf(X-7, Y-6, np.transpose(np.log10(-eta)), 20, cmap = 'viridis')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$C$ (au)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    # yticks = [-2, -1, 0, 1]
    ax.set_xticks([-2,-1,0,1])
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    # ticks = [0, np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    # ax.scatter(np.log10(I_green[col])-7, np.log10(OmegaRArr[row])-6, marker = '*', color = 'red')
    plt.show()
    
    np.savez(os.path.join(_REFFILES_DIR, "NV0PLfits.npz"),
         nu=nu,
         C=C,
         I_green=I_green,
         OmegaRArr=OmegaRArr,
         eta = eta,
         R_off = R_off,
         R_on = R_on)

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRGreen.png", dpi=300, bbox_inches="tight",transparent=True
            )
        
        
def solveNVIRPL_GreenOmegaR(saveBool = False,
                            N_green = 10,
                            N_OmegaR = 10,
                            OmegaRmin = 4,
                            OmegaRmax = 7,
                            I_pmin = 5,
                            I_pmax = 8,
                            fastBool = False,
                            OmegaROpt = 0,
                            I_pOpt = 0,
                            bw = 0.01e9*2*np.pi,
                            Ndets = 51):
    """2D sweep of green power vs Rabi frequency; compute singlet IR emission (state 6) sensitivity via an adaptive detuning fit and save to IRPLfits.npz."""
    
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    C = np.zeros([N_green, N_OmegaR])
    nu = np.zeros([N_green, N_OmegaR])
    R_on = np.zeros([N_green, N_OmegaR])
    R_off = np.zeros([N_green, N_OmegaR])
    for i, I_p in enumerate(tqdm(I_green, desc="Green intensity")):
        for j, OmegaR in enumerate(tqdm(OmegaRArr, desc="OmegaR", leave=False)):
            popOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = 0,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            popOff = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = 0,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
                
            R_on[i,j] = (popOn[6])*gamma_s
            R_off[i,j] = (popOff[6])*gamma_s
            C[i,j] = -(R_off[i,j]-R_on[i,j])/R_off[i,j]

            for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
                detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
                Cdet = np.zeros(Ndets)
                for k, det in enumerate(detunings):
                    popDet =    nv_steady_state_default(
                        Omega = OmegaR,
                        I_s = 0,
                        I_p = I_p,
                        detuning = det,
                        zeeman = zeeman,
                    )
                    Cdet[k] = -(R_off[i,j] - (popDet[6])*gamma_s)/R_off[i,j]
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
                plt.show()
                if np.size(FWHM) < 1:
                    print('FWHM fitting didnt work')
                elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.8:
                    bw = bw*2
                    continue
                elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
                    nu[i,j] = FWHM[0]*1e9
                    break
                elif FWHM >= bw/2:
                    bw = bw*2
                    continue
                else:
                    bw = bw/3
                    continue
                    
                    
            
            
            
    eta = calcSensitivityCW_countrate(nu, C, R_off)
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
       r"$I_{\text{p}}$ (kW/cm$^2$)", 
       r'$\Omega_{\text{R}}/2\pi$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    min_index = np.argmin(eta)
    row, col = np.unravel_index(min_index, eta.shape)
    
    X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
    im = ax.contourf(X-7, Y-6, np.transpose(np.log10(eta)), 20, cmap = 'viridis')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$C$ (au)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    # yticks = [-2, -1, 0, 1]
    ax.set_xticks([-2,-1,0,1])
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    # ticks = [0, np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    ax.scatter(np.log10(I_green[col])-7, np.log10(OmegaRArr[row])-6, marker = '*', color = 'red')
    plt.show()
    
    np.savez(os.path.join(_REFFILES_DIR, "IRPLfits.npz"),
         nu=nu,
         C=C,
         I_green=I_green,
         OmegaRArr=OmegaRArr,
         eta = eta,
         R_off = R_off,
         R_on = R_on)

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastOmegaRGreen.png", dpi=300, bbox_inches="tight",transparent=True
            )
        
        

def solvePopulationIRGreen(saveBool = False,
                            N_green = 10,
                            N_IR = 10,
                            I_smin = 7,
                            I_smax = 11,
                            I_pmin = 5,
                            I_pmax = 8,
                            OmegaR = gamma_phi,
                            I_sOpt = 0,
                            I_pOpt = 0,
                            bool1042 = False):
    """2D sweep of IR intensity vs green intensity; plot the m_7 on/off population contrast as a contour map."""
    
    I_IR = np.logspace(I_smin, I_smax, N_IR)
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    p7 = np.zeros([N_IR, N_green])
    p7On = np.zeros([N_IR, N_green])
    p7Off = np.zeros([N_IR, N_green])
    for i, I_s in enumerate(tqdm(I_IR, desc="IR intensity")):
        for j, I_p in enumerate(I_green):
            populations =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            populationsDetuning = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            # p7[i,j] = (populations[7] - populationsDetuning[7])
            p7On[i,j] = (populations[7])
            p7Off[i,j] = (populationsDetuning[7])
            # p7[i,j] = (p7On[i,j])/p7Off[i,j]
            p7[i,j] = (p7On[i,j] - p7Off[i,j])
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
       r"$I_{\text{S}}$ (kW/cm$^2$)", 
       r'$I_{\text{P}}$ (kW/cm$^2$)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    X, Y = np.meshgrid(np.log10(I_IR), np.log10(I_green))
    # im = ax.pcolormesh(X-7, Y-7, np.transpose(p7), cmap = 'hot')
    im = ax.contourf(X-7, Y-7, np.transpose(p7)*100, 21, cmap = 'Greys_r')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$p_7^{\text{(on)}}-p_7^{\text{(off)}}$ ($\%$)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    ticks = [np.min(p7), np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    
    ax.scatter(np.log10(I_sOpt)-7, np.log10(I_pOpt)-7, marker = '*', color = 'red')
    
    plt.show()

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastGreenIR.png", dpi=300, bbox_inches="tight",transparent=True
            )
        
def solvePopulationIRGreenOmegaR(saveBool = False,
                            N = 10,
                            I_smin = 7,
                            I_smax = 11,
                            I_pmin = 5,
                            I_pmax = 8,
                            OmegaRmin = 5,
                            OmegaRmax = 7,
                            I_sOpt = 0,
                            I_pOpt = 0,
                            OmegaROpt = 6,
                            bool1042 = False):
    """Three pairwise 2D sweeps (IR/green, IR/Rabi, green/Rabi) of the m_7 contrast, plotted as a shared-colorbar contour panel."""
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.32, 0.53*PRQuantumWidthFull, 
       [r"$I_{\text{s}}$ (MW/cm$^2$)", r"$I_{\text{s}}$ (MW/cm$^2$)", r"$I_{\text{p}}$ (kW/cm$^2$)"], 
       [r'$I_{\text{p}}$ (kW/cm$^2$)', r'$\Omega_{\text{R}}/2\pi$ (MHz)', r'$\Omega_{\text{R}}/2\pi$ (MHz)'],
       labelPadX =-5, labelPadY = 5, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 3, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    I_IR = np.logspace(I_smin, I_smax, N)
    I_green = np.logspace(I_pmin, I_pmax, N)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N)
    p7IRGreen = np.zeros([N, N])
    p7IRRabi = np.zeros([N, N])
    p7GreenRabi = np.zeros([N, N])
    p7On = np.zeros([N, N])
    p7Off = np.zeros([N, N])
    for i, I_s in enumerate(tqdm(I_IR, desc="IR/Green")):
        for j, I_p in enumerate(I_green):
            pOn =         nv_steady_state_default(
                Omega = OmegaROpt*2*np.pi,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            pOff = nv_steady_state_default(
                Omega = 0,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            p7On[i,j] = (pOn[7])
            p7Off[i,j] = (pOff[7])
            p7IRGreen[i,j] = (p7On[i,j] - p7Off[i,j])
            
    for i, I_s in enumerate(tqdm(I_IR, desc="IR/Rabi")):
        for j, OmegaR in enumerate(OmegaRArr):
            pOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_pOpt,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            pOff = nv_steady_state_default(
                Omega = 0,
                I_s = I_s,
                I_p = I_pOpt,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            p7On[i,j] = (pOn[7])
            p7Off[i,j] = (pOff[7])
            p7IRRabi[i,j] = (p7On[i,j] - p7Off[i,j])
            
    for i, I_p in enumerate(tqdm(I_green, desc="Green/Rabi")):
        for j, OmegaR in enumerate(OmegaRArr):
            pOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_sOpt,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            pOff = nv_steady_state_default(
                Omega = 0,
                I_s = I_sOpt,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                steadystatebool=True,
                tEnd = 2e-6,
                bool1042 = bool1042,
            )
            p7On[i,j] = (pOn[7])
            p7Off[i,j] = (pOff[7])
            p7GreenRabi[i,j] = (p7On[i,j] - p7Off[i,j])
    
    vmin = np.min([
        p7IRGreen.min(),
        p7IRRabi.min(),
        p7GreenRabi.min()
    ]) * 100
    
    vmax = np.max([
        p7IRGreen.max(),
        p7IRRabi.max(),
        p7GreenRabi.max()
    ]) * 100
    
    levels = np.linspace(vmin, vmax, 256)  # smoother + hits endpoints better
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    X, Y = np.meshgrid(np.log10(I_IR), np.log10(I_green))
    im = ax[0].pcolormesh(X-10, Y-7, np.transpose(p7IRGreen)*100, cmap='Greys_r', norm = norm,rasterized=True, shading='gouraud',  )
    X, Y = np.meshgrid(np.log10(I_IR), np.log10(OmegaRArr))
    im = ax[1].pcolormesh(X-10, Y-6, np.transpose(p7IRRabi)*100, cmap='Greys_r', norm = norm,rasterized=True , shading='gouraud'  )
    X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
    im = ax[2].pcolormesh(X-7, Y-6, np.transpose(p7GreenRabi)*100, cmap='Greys_r', norm = norm,rasterized=True  , shading='gouraud' )
    plt.tight_layout()
    
    
    for i, axx in enumerate(ax):
        yticks = axx.get_yticks()
        xticks = axx.get_xticks()
        yticks = yticks[::len(yticks)-1]
        xticks = xticks[::len(xticks)-1]
        axx.set_yticks(yticks)
        axx.set_xticks(xticks)
        axx.set_xticklabels([rf"${{{val:.1f}}}$" for val in 10**xticks])
        if i > 0:
            axx.set_yticklabels([rf"${{{val:.2f}}}$" for val in 10**yticks])
        else:
            axx.set_yticklabels([rf"${{{val:.0f}}}$" for val in 10**yticks])
            
        # ticks = [np.min(p7), np.max(p7)]
        # cbar.set_ticks(ticks)
        # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    
    ax[0].scatter(np.log10(I_sOpt)-10, np.log10(I_pOpt)-7, marker = '*', color = 'red')
    ax[1].scatter(np.log10(I_sOpt)-10, np.log10(OmegaROpt)-6, marker = '*', color = 'red')
    ax[2].scatter(np.log10(I_pOpt)-7, np.log10(OmegaROpt)-6, marker = '*', color = 'red')
    
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', location='right', aspect=30, pad = 0.15)
    cbar.set_label(r'$p_7^{\text{(on)}}-p_7^{\text{(off)}}$ ($\%$)',
               labelpad=-15, fontsize=labelfontsize)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([rf"${{{vmin:.2f}}}$", rf"${{{vmax:.2f}}}$"], fontsize = numfontsize)
    plt.show()

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastAll.svg", dpi=300, bbox_inches="tight",transparent=True
            )
        
def solvePopulationIRGreenOmegaRDetuning(
                            I_IR,
                            I_green,
                            OmegaRArr,
                            saveBool = False,
                            N = 3,
                            I_sOpt = 0,
                            I_pOpt = 0,
                            OmegaROpt = 6,
                            Ndets = 101,
                            bw = 0.01e9*np.pi*2):
    """Detuning line shapes at a few IR/green/Rabi power levels, offset-summed and scaled for display in one 3-panel figure."""
    
    colorsGreen = [colorDarkGreen, colorGreen, colorLightGreen]
    colorsRabi = [colorDarkBlue, colorBlue, colorLightBlue]
    colorsIR = [colorDarkRed, colorRed, colorLightRed]
    scaleIR = [1, 1, 1]
    scaleGreen = [1, 1, 1]
    scaleRabi = [1, 1, 1]
    
    maxGreen= np.zeros(4)
    maxRabi= np.zeros(4)
    maxIR= np.zeros(4)
    
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.28, 0.53*PRQuantumWidthFull, 
       r"$\Delta_{\text{MW}}/2\pi$ (MHz)", 
       [r'$\Delta p_7$ ($\%$)', r'$\Delta p_7$ ($\%$)', r'$\Delta p_7$ ($\%$)'],
       labelPadX =5, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 3, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    for j, I_s in enumerate(tqdm(I_IR, desc="IR")):
        # detuning
        detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
        p7 = np.zeros(Ndets)
        
        popOff = nv_steady_state_default(
            Omega = OmegaROpt*0,
            I_s = I_s,
            I_p = I_pOpt,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
        )
        
        for i, det in enumerate(detunings):
            pop = nv_steady_state_default(
                Omega = OmegaROpt,
                I_s = I_s,
                I_p = I_pOpt,
                detuning = (det),
                zeeman = zeeman,
                steadystatebool=True,
            )
            p7[i] = (pop[7])
        
        peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), p7)
        # print(FWHM)
        
        delta = 2*np.pi*2.16e6  # example shift
        dx = detunings[1] - detunings[0]
        shift_pts = int(delta / dx)
        
        x = detunings
        y = p7 - popOff[7]
        
        y_shift_plus  = np.roll(y, -shift_pts)
        y_shift_minus = np.roll(y,  shift_pts)

        y_total = (y + y_shift_plus + y_shift_minus)*100*scaleIR[j]
        # y_total = p7*100
        maxIR[1+j] = y_total.max()
        
        
        ax[2].plot((detunings-zeeman)/1e6/(2*np.pi), y_total, color = colorsIR[j], lw = lw) #*I_IR[j]
            
    for j, OmegaR in enumerate(tqdm(OmegaRArr, desc="Rabi")):# detuning
        detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
        p7 = np.zeros(Ndets)
            
        popOff = nv_steady_state_default(
            Omega = OmegaR*0,
            I_s = I_sOpt,
            I_p = I_pOpt,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
        )
        
        for i, det in enumerate(detunings):
            pop = nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_sOpt,
                I_p = I_pOpt,
                detuning = (det),
                zeeman = zeeman,
                steadystatebool=True,
            )
            p7[i] = (pop[7])
        
        peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), p7)
        
        
        delta = 2*np.pi*2.16e6  # example shift
        dx = detunings[1] - detunings[0]
        shift_pts = int(delta / dx)
        
        x = detunings
        y = p7 - popOff[7]
        
        y_shift_plus  = np.roll(y, -shift_pts)
        y_shift_minus = np.roll(y,  shift_pts)
        y_shift_pplus  = np.roll(y, -2*shift_pts)
        y_shift_mminus = np.roll(y,  2*shift_pts)
    
        y_total = (y + y_shift_plus + y_shift_minus)*100*scaleRabi[j]
        if j == 1:
            print('p7 on:', p7.max())
            print('p7 off:', popOff[7])
            print('Delta p7 :', p7.max()- popOff[7])
            print('FWHM: ', FWHM)
        # y_total /= y_total.max()
        maxRabi[1+j] = y_total.max()
        
        
        ax[1].plot((detunings-zeeman)/1e6/(2*np.pi), y_total, color = colorsRabi[j], lw = lw) #*I_IR[j]
            
    for j, I_p in enumerate(tqdm(I_green, desc="Green")):
        detunings = np.linspace(detuning - bw, detuning + bw, Ndets)
        p7 = np.zeros(Ndets)
            
        popOff = nv_steady_state_default(
            Omega = OmegaROpt*0,
            I_s = I_sOpt,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
            steadystatebool=True,
        )
        for i, det in enumerate(detunings):
            pop = nv_steady_state_default(
                Omega = OmegaROpt,
                I_s = I_sOpt,
                I_p = I_p,
                detuning = (det),
                zeeman = zeeman,
                steadystatebool=True,
            )
            p7[i] = (pop[7])
        
        peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), p7)
        
        delta = 2*np.pi*2.16e6  # example shift
        dx = detunings[1] - detunings[0]
        shift_pts = int(delta / dx)
        
        x = detunings
        y = p7 - popOff[7]
        
        y_shift_plus  = np.roll(y, -shift_pts)
        y_shift_minus = np.roll(y,  shift_pts)
        y_shift_pplus  = np.roll(y, -2*shift_pts)
        y_shift_mminus = np.roll(y,  2*shift_pts)
    
        y_total = (y + y_shift_plus + y_shift_minus)*100*scaleGreen[j]
        
        maxGreen[1+j] = y_total.max()
        
        
        ax[0].plot((detunings-zeeman)/1e6/(2*np.pi), y_total, color = colorsGreen[j], lw = lw) #*I_IR[j]

    ax[0].set_yticks(  maxGreen)
    ax[0].set_yticklabels([rf"${{{val:.1f}}}$" for val in ( maxGreen)], fontsize = numfontsize)
    ax[1].set_yticks(  maxRabi)
    ax[1].set_yticklabels([rf"${{{val:.1f}}}$" for val in ( maxRabi)], fontsize = numfontsize)
    ax[2].set_yticks( [0] + maxIR)
    ax[2].set_yticklabels([rf"${{{val:.1f}}}$" for val in ( maxIR)], fontsize = numfontsize)
    
    
    
    for i, axx in enumerate(ax):
        axx.set_xlim([-11,11])
        axx.spines['bottom'].set_bounds(-10, 10)
        yticks = axx.get_yticks()
        axx.set_ylim([-yticks.max()/14, yticks.max()])
        axx.spines['left'].set_bounds(0, yticks.max())
        axx.spines["right"].set_visible(False)
        axx.spines["top"].set_visible(False)
        axx.spines["bottom"].set_visible(False)
        axx.yaxis.set_ticks_position('left')
        axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        axx.tick_params(axis='x', which='both', top=False, bottom=False)
        
        # axx.xaxis.set_label_position("bottom")
        # axx.xaxis.set_ticks_position('bottom')
        # axx.set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal
        # axx.xaxis.set_label_coords(0.55,-0.27)
    
    
    ax[2].spines["bottom"].set_visible(True)
    ax[2].tick_params(axis='x', which='both', top=False, bottom=True)
    
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    
    # cbar = fig.colorbar(im, ax=ax, orientation='vertical', location='right', aspect=30, pad = 0.15)
    # cbar.set_label(r'$p_7^{\text{(on)}}-p_7^{\text{(off)}}$ ($\%$)',
    #            labelpad=-15, fontsize=labelfontsize)
    # cbar.set_ticks([vmin, vmax])
    # cbar.set_ticklabels([rf"${{{vmin:.2f}}}$", rf"${{{vmax:.2f}}}$"], fontsize = numfontsize)
    plt.show()

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastAllDetuning.svg", dpi=300, bbox_inches="tight",transparent=True
            )
        


def solvePopulationLineplots(saveBool = False):
    """Plot measured + fitted NV0/NV- PL vs IR power alongside a global least-squares fit of the shared rate parameters."""
    # IR Power Sweep dark state paper all populations
    
    fig, ax = plot_custom(
        PRQuantumWidthFull*0.4, 0.45*PRQuantumWidthFull, 
        r"$I_s$ (W/m$^2$)", 
        [r'NV$^{0}$ PL, $m_{10}$ (au)',r'NV$^{-}$ PL, $m_{3}$ (au)'],
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    ms = 1  # local override of nvConstants.ms/lw for this figure's marker/line size
    lw = 1
    cmap = plt.cm.Greens
    colors = cmap(np.linspace(0.3, 1, 3))   # avoid very light end
    
    dataNV01 = loadmat(('C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/' +
                    'dataBarclay/measurementsNLO/OG__Sigurd_DarkState/x0_final.mat'), simplify_cells=True)
    dataNV02 = loadmat(('C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/' +
                    'dataBarclay/measurementsNLO/OG__Sigurd_DarkState/p0_final.mat'), simplify_cells=True)
    
    NV0counts1 = dataNV02['p0_norm_inp_final'][:,0,0]/1.1
    NV0counts2 = dataNV02['p0_norm_inp_final'][:,1,0]/1.05
    NV0counts3 = dataNV02['p0_norm_inp_final'][:,2,0]/1.02
    NV0power1 = dataNV01['x0_inp'][:,0,0]*5e2 * 1e4
    # filt = NV0power1 < 1e9
    # NV0power1 = NV0power1[0:600]
    # NV0counts1 = NV0counts1[0:600]
    NV0power2 = dataNV01['x0_inp'][:,1,0]*5e2 * 1e4
    # NV0power2 = NV0power2[0:600]
    # NV0counts2 = NV0counts2[0:600]
    NV0power3 = dataNV01['x0_inp'][:,2,0]*5e2 * 1e4
    # NV0power3 = NV0power3[0:600]
    # NV0counts3 = NV0counts3[0:600]
    
    ax[0].scatter(NV0power1, NV0counts1, s = ms, marker = '.', color = colors[0], edgecolors='none')
    ax[0].scatter(NV0power2, NV0counts2, s = ms, marker = '.', color = colors[1], edgecolors='none')
    ax[0].scatter(NV0power3, NV0counts3, s = ms, marker = '.', color = colors[2], edgecolors='none')
    
    ax[0].set_ylim([0,1.2e0])
    
    ax[0].set_xscale('log');
    
    dataNVm1 = loadmat(('C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/' +
                    'dataBarclay/measurementsNLO/OG__Sigurd_DarkState/xm_final.mat'), simplify_cells=True)
    dataNVm2 = loadmat(('C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/' +
                    'dataBarclay/measurementsNLO/OG__Sigurd_DarkState/pm_final.mat'), simplify_cells=True)
    
    NVmcounts1 = dataNVm2['pm_norm_inp_final'][:,0,0]*1.02
    NVmcounts2 = dataNVm2['pm_norm_inp_final'][:,1,0]*1.02
    NVmcounts3 = dataNVm2['pm_norm_inp_final'][:,2,0]*1.02
    NVmpower1 = dataNVm1['xm_inp'][:,0,0]*5e2 * 1e4
    # 0:600 = NVmpower1 < 1e9
    # NVmpower1 = NVmpower1[0:600]
    # NVmcounts1 = NVmcounts1[0:600]
    NVmpower2 = dataNVm1['xm_inp'][:,1,0]*5e2 * 1e4
    # NVmpower2 = NVmpower2[0:600]
    # NVmcounts2 = NVmcounts2[0:600]
    NVmpower3 = dataNVm1['xm_inp'][:,2,0]*5e2 * 1e4
    # NVmpower3 = NVmpower3[0:600]
    # NVmcounts3 = NVmcounts3[0:600]
    
    ax[1].scatter(NVmpower1, NVmcounts1, s = ms, marker = '.', color = colors[0], edgecolors='none')
    ax[1].scatter(NVmpower2, NVmcounts2,s = ms, marker = '.', color = colors[1], edgecolors='none')
    ax[1].scatter(NVmpower3, NVmcounts3, s = ms, marker = '.', color = colors[2], edgecolors='none')
    
    ax[1].set_ylim([0,1.2e0])
    
    ax[1].set_xscale('log');
    
    # Pack experimental datasets
    exp_data = [
        dict(I_IR = NV0power1, NV0 = NV0counts1, NVm = NVmcounts1),
        dict(I_IR = NV0power2, NV0 = NV0counts2, NVm = NVmcounts2),
        dict(I_IR = NV0power3, NV0 = NV0counts3, NVm = NVmcounts3),
    ]
    
    # Model IR grid (fixed)
    I_green = 1e-3/(0.3e-6**2*np.pi)  # 1 mW over a beam of radius 0.3 um -> W/m^2
    I_pump  = I_green*np.array([0.4, 1.3, 4.6])
    N_IR = 100 
    I_IR_model = np.logspace(6,14, N_IR) #W/cm^2 * cm^2/m^2
    
    # I_green = 1e-3/(0.3e-6**2*np.pi)
    # I_pump = I_green*np.array([0.4, 1.3, 4.6])
    OmegaR = 0
    
    def simulate_curves(paramsRates, I_p):
        p = np.zeros((len(I_IR_model), 11))
    
        for i, I_s in enumerate(I_IR_model):
            result = nv_steady_state_fast(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                gamma_phi = gamma_phi,
                detuning = detuning,
                zeeman   = zeeman,
                gamma_eg=gamma_eg,
                gamma_isc_0=gamma_isc_0,
                gamma_isc_1=gamma_isc_1,
                gamma_isc_g0=gamma_isc_g0,
                gamma_isc_g1=gamma_isc_g1,
                gamma_s=gamma_s,
                gamma_NV0_eg = gamma_NV0_eg,
                gamma_NV0_em = gamma_NV0_em,
                gamma_NV0_mg = gamma_NV0_mg,
                sigma_p = sigma_p,
                sigma_s = sigma_s,
                lamb_s = lamb_s,
                lamb_p = lamb_p,
                c = c_const,
                h = h,
                paramsRates = paramsRates,
                paramsRatesReal = paramsRatesReal
            )
            p[i,:] = result
    
        NV0_model = p[:,10] / np.max(p[0,10])
        NVm_model = (p[:,3] + p[:,4] + p[:,5]) \
                    / np.max(p[0,3] + p[0,4] + p[0,5])

        return NV0_model, NVm_model
    
    def interp_to_exp(I_model, y_model, I_exp):
        f = interp1d(I_model, y_model, kind='linear', fill_value="extrapolate")
        return f(I_exp)

    def residuals(paramsRates):
    
        res = []
    
        for k, d in enumerate(exp_data):
            I_p = I_pump[k]
    
            NV0_model, NVm_model = simulate_curves(paramsRates, I_p)
    
            NV0_fit = interp_to_exp(I_IR_model, NV0_model, d["I_IR"])
            NVm_fit = interp_to_exp(I_IR_model, NVm_model, d["I_IR"])
    
            # residuals
            res.append(NV0_fit - d["NV0"])
            res.append(NVm_fit - d["NVm"])
    
        return np.concatenate(res)
    
    p = np.zeros([N_IR, 11])
    if True:
        OmegaR = 0 
        for j, I_p in enumerate(I_pump):
            for i, I_s in enumerate(I_IR_model):
                result =  nv_steady_state_default(
                    Omega = OmegaR,
                    I_s = I_s,
                    I_p = I_p,
                    detuning = detuning,
                    zeeman = zeeman,
                    verboseResults = True,
                )
                p[i,:] = result
            # print(paramsRatesReal)
            ax[1].plot(I_IR_model, (p[:,3] + p[:,4] + p[:,5])/np.max(p[0,3] + p[0,4] + p[0,5]), color = 'k', lw = lw/2)
            ax[0].plot(I_IR_model, p[:,10]/np.max(p[0,10]), color = 'k', lw = lw/2)
        
    else:
        x0 = np.ones(len(paramsRatesOnes))   # your initial guess
        
        lower = 0.01 * x0
        lower[0:2] = x0[0:2]*0.999
        upper = 10.0 * x0
        upper[0:2] = x0[0:2]*1.001
        
        sol = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            verbose=2,
            ftol=1e-6,
            xtol=1e-6
        )
        
        params_opt = sol.x
        print(params_opt*paramsRatesReal)
        # print(params_opt)
        
        for j, I_p in enumerate(I_pump):
            for i, I_s in enumerate(I_IR_model):
                result =  nv_steady_state_fast(
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = detuning, #rad/s
                                            zeeman   = zeeman, #rad/s
                                            gamma_eg=gamma_eg,
                                            gamma_isc_0=gamma_isc_0,
                                            gamma_isc_1=gamma_isc_1,
                                            gamma_isc_g0=gamma_isc_g0,
                                            gamma_isc_g1=gamma_isc_g1,
                                            gamma_s=gamma_s,
                                            gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
                                            gamma_NV0_em = gamma_NV0_em, #k_75
                                            gamma_NV0_mg = gamma_NV0_mg,   #k_56
                                            sigma_p = sigma_p, # m^%-2
                                            sigma_s = sigma_s, # m^%-2
                                            lamb_s = lamb_s,   # IR probe
                                            lamb_p = lamb_p,    # green pump
                                            c = c_const, #m/s
                                            h = h,      # Planck constant [J·s],
                                            paramsRates = params_opt,
                                            paramsRatesReal = paramsRatesReal
                                            )
                p[i,:] = result
            ax[1].plot(I_IR_model, (p[:,3] + p[:,4] + p[:,5])/np.max(p[0,3] + p[0,4] + p[0,5]),color = colors[j])
            ax[0].plot(I_IR_model, p[:,10]/np.max(p[0,10]),color = colors[j])
    
    plt.tight_layout()
    plt.xscale('log');
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "DarkStateFigure.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    plt.show()
    
def solvePopulationSteadyStateContrast(saveBool = False, bool1042 = False):
    """Plot steady-state on/off contrast vs IR and green intensity (paper figure)."""
    # IR Power Sweep dark state paper all populations
    lw = 1  # local override of nvConstants.lw for this figure's line width
    fig, ax = plot_custom(
        PRQuantumWidthHalf, PRQuantumWidthHalf, 
        [r"532 nm intensity, $I_p$ (W/m$^2$)", r"1042 nm intensity, $I_s$ (W/m$^2$)"],
        [r'contrast, $C$',r'contrast, $C$'],
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = False, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    for axx in ax:
        axx.set_xscale('log');
        axx.set_yscale('log');
    N_IR = 100
    NV0power1 = np.logspace(7, 12, N_IR)
    I_pp = np.array([I_pOpt])
    I_IR = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR) #W/cm^2 * cm^2/m^2
    OmegaR = OmegaROptRadps
    p = np.zeros([N_IR, 11])
    pOff = np.zeros([N_IR, 11])
    # detuning = 0
    for I_p in I_pp:
        for i, I_s in enumerate(I_IR):
            result =  nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            resultOff =  nv_steady_state_fast(
                    Omega = OmegaR*0,
                    I_s = I_s,
                    I_p = I_p,
                    gamma_phi = gamma_phi,
                    detuning = detuning,
                    zeeman   = zeeman,
                    gamma_eg=gamma_eg,
                    gamma_isc_0=gamma_isc_0,
                    gamma_isc_1=gamma_isc_1,
                    gamma_isc_g0=gamma_isc_g0,
                    gamma_isc_g1=gamma_isc_g1,
                    gamma_s=gamma_s,
                    gamma_NV0_eg = gamma_NV0_eg,
                    gamma_NV0_em = gamma_NV0_em,
                    gamma_NV0_mg = gamma_NV0_mg,
                    sigma_p = sigma_p,
                    sigma_s = sigma_s,
                    lamb_s = lamb_s,
                    lamb_p = lamb_p,
                    c = c_const,
                    h = h,
                    paramsRates = paramsRatesOnes,
                    paramsRatesReal = paramsRatesReal
                )
            p[i,:] = result
            pOff[i,:] = resultOff
        labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
                  r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
        colors = ['#3dfe00ff','#35db00ff','#218800ff',
                  '#fe7900ff','#fe0000ff','#fe0076ff',
                  '#190cffff','#800000ff','#800cffff', '#ff0cf4ff', '#feb400ff'];
        for i in range(11):
            if i in [7, 10]:
                ax[1].plot(NV0power1, np.abs(p[:, i]-pOff[:, i])/np.abs(p[:, i]+pOff[:, i])*2, color = colors[i], lw =lw, label = labels[i])
            if i in [3]:
                ax[1].plot(NV0power1, np.abs(p[:, i]+p[:, i+1]+p[:, i+2] - (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))/np.abs(p[:, i]+p[:, i+1]+p[:, i+2] + (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))*2, color = colors[i+1], lw =lw, label = labels[i])
            if i in [0]:
                ax[1].plot(NV0power1, np.abs(p[:, i]+p[:, i+1]+p[:, i+2] - (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))/np.abs(p[:, i]+p[:, i+1]+p[:, i+2] + (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))*2, color = colors[i+1], lw =lw, label = labels[i])
             
    # ax.legend()
    # ax[1].set_ylim([1e-3, 1])
    ax[1].set_xscale('log');
    
    NV0power1 = np.logspace(7, 10, N_IR)
    I_s = I_sOpt
    I_g = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR) #W/cm^2 * cm^2/m^2

    p = np.zeros([N_IR, 11])
    # detuning = 0
    
    for i, I_p in enumerate(I_g):
        result =  nv_steady_state_default(
            Omega = OmegaR,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
        )            
        resultOff =  nv_steady_state_default(
            Omega = OmegaR*0,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
        )
        p[i,:] = result
        pOff[i,:] = resultOff
    C = np.abs(p-pOff)
    for i in range(11):
        if i in [7]:
            ax[0].plot(NV0power1, C[:, i], color = colors[i], lw =lw, label = labels[i], linestyle = '-')
        if i in [10]:
            ax[0].plot(NV0power1, C[:, i], color = colors[i], lw =lw, label = labels[i], linestyle = ':')
        if i in [3]:
            ax[0].plot(NV0power1, np.abs(p[:, i]+p[:, i+1]+p[:, i+2] - (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))/np.abs(p[:, i]+p[:, i+1]+p[:, i+2] + (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))*2, color = colors[i+1], lw =lw, label = labels[i])
            Cmax = np.max(np.abs(p[0, i]+p[0, i+1]+p[0, i+2] - (pOff[0, i]+pOff[0, i+1]+pOff[0, i+2]))/np.abs(p[:, i]+p[:, i+1]+p[:, i+2] + (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))*2)
        if i in [0]:
            ax[0].plot(NV0power1, np.abs(p[:, i]+p[:, i+1]+p[:, i+2] - (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))/np.abs(p[:, i]+p[:, i+1]+p[:, i+2] + (pOff[:, i]+pOff[:, i+1]+pOff[:, i+2]))*2, color = colors[i+1], lw =lw, label = labels[i])
               
    # ax.legend()
    # ax[0].set_ylim([1e-5, 1])
    plt.tight_layout()
    ax[0].set_xscale('log');
    print(Cmax)
    
    
    ax[0].set_ylim([0.4e-3, 1])
    ax[1].set_ylim([0.2e-6, 1])
    ax[0].set_yticks([1e0, 1e-1, 1e-2, 1e-3],minor = False)
    ax[0].spines['left'].set_bounds(1e-3, 1)
    ax[1].set_yticks([1e0, 1e-2, 1e-4, 1e-6],minor = False)
    ax[1].spines['left'].set_bounds(1e-6, 1)
    ax[1].spines['bottom'].set_bounds(1e7, 1e12)
    ax[0].spines['bottom'].set_bounds(1e7, 1e10)
    for axx in ax:
        # axx.plot(NV0power1, Cmax + NV0power1*0, 'k--', lw =lw, zorder = -1)
        
        # axx.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12],minor = False)
        axx.set_xticks([],minor = True)
        axx.set_yticks([],minor = True)
        axx.spines["right"].set_visible(False)
        axx.spines["top"].set_visible(False)
        axx.yaxis.set_ticks_position('left')
        axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        axx.tick_params(axis='x', which='both', top=False, bottom=True)
        axx.spines["bottom"].set_visible(True)
    
    plt.tight_layout()
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NVPopulationContrastPaper.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    
    plt.show()
    
    
    
def solvePopulationSteadyStateOmegaR(saveBool = False, bool1042 = False):
    """Plot steady-state on/off contrast vs Rabi frequency OmegaR."""
    # IR Power Sweep dark state paper all populations
    lw = 1  # local override of nvConstants.lw for this figure's line width
    fig, ax = plot_custom(
        PRQuantumWidthHalf, PRQuantumWidthHalf, 
        r"$\Omega_R$ ",
        r'population',
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = False, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    ax.set_xscale('log');
    N_IR = 100
    NV0power1 = np.logspace(1, 6, N_IR)
    I_pp = np.array([1e9])
    I_IR = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR) #W/cm^2 * cm^2/m^2
    I_s = 1e-10
    OmegaRm = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR)*2*np.pi
    p = np.zeros([N_IR, 11])
    pOff = np.zeros([N_IR, 11])
    # detuning = 0
    for I_p in I_pp:
        for i, OmegaR in enumerate(OmegaRm):
            result =  nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            resultOff =  nv_steady_state_fast(
                    Omega = 1e-10,
                    I_s = I_s,
                    I_p = I_p,
                    gamma_phi = gamma_phi,
                    detuning = detuning,
                    zeeman   = zeeman,
                    gamma_eg=gamma_eg,
                    gamma_isc_0=gamma_isc_0,
                    gamma_isc_1=gamma_isc_1,
                    gamma_isc_g0=gamma_isc_g0,
                    gamma_isc_g1=gamma_isc_g1,
                    gamma_s=gamma_s,
                    gamma_NV0_eg = gamma_NV0_eg,
                    gamma_NV0_em = gamma_NV0_em,
                    gamma_NV0_mg = gamma_NV0_mg,
                    sigma_p = sigma_p,
                    sigma_s = sigma_s,
                    lamb_s = lamb_s,
                    lamb_p = lamb_p,
                    c = c_const,
                    h = h,
                    paramsRates = paramsRatesOnes,
                    paramsRatesReal = paramsRatesReal
                )
            p[i,:] = result
            pOff[i,:] = resultOff
        labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
                  r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
        colors = ['#3dfe00ff','#35db00ff','#218800ff',
                  '#fe7900ff','#fe0000ff','#fe0076ff',
                  '#190cffff','#800000ff','#800cffff', '#ff0cf4ff', '#feb400ff'];
        for i in range(11):
            if i in [7]:
                ax.plot(NV0power1, (p[:, i]-pOff[:, i])/(p[:, i]+pOff[:, i])*2, color = colors[i], lw =lw, label = labels[i])
                # ax.plot(NV0power1, (pOff[:, i]), color = colors[i], lw =lw, label = labels[i])
            if i in [3]:
                ax.plot(NV0power1, np.abs(p[:, i]+p[:, i+1]+p[:, i+2]-pOff[:, i]-pOff[:, i+1]-pOff[:, i+2])/np.abs(p[:, i]+p[:, i+1]+p[:, i+2]+pOff[:, i]+pOff[:, i+1]+pOff[:, i+2])*2, color = colors[i+1], lw =lw, label = labels[i])
            
    # ax.legend()
    # ax[1].set_ylim([1e-3, 1])
    ax.set_xscale('log');
    ax.set_yscale('log');
    
    
    
    # ax[0].set_ylim([0.4e-3, 1])
    # ax[1].set_ylim([0.2e-6, 1])
    # ax[0].set_yticks([1e0, 1e-1, 1e-2, 1e-3],minor = False)
    # ax[0].spines['left'].set_bounds(1e-3, 1)
    # ax[1].set_yticks([1e0, 1e-2, 1e-4, 1e-6],minor = False)
    # ax[1].spines['left'].set_bounds(1e-6, 1)
    # ax[1].spines['bottom'].set_bounds(1e7, 1e12)
    # ax[0].spines['bottom'].set_bounds(1e7, 1e10)
    # for axx in ax:
    #     # axx.plot(NV0power1, Cmax + NV0power1*0, 'k--', lw =lw, zorder = -1)
        
    #     # axx.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12],minor = False)
    #     axx.set_xticks([],minor = True)
    #     axx.set_yticks([],minor = True)
    #     axx.spines["right"].set_visible(False)
    #     axx.spines["top"].set_visible(False)
    #     axx.yaxis.set_ticks_position('left')
    #     axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    #     axx.tick_params(axis='x', which='both', top=False, bottom=True)
    #     axx.spines["bottom"].set_visible(True)
    
    plt.tight_layout()
    
    # if saveBool:
    #     fig.savefig(
    #         (_FIGURES_DIR + "/") + "NVPopulationContrastPaper.svg", dpi=300, bbox_inches="tight",transparent=True
    #         )
    
    plt.show()
    
def solveSensitivityVaryIRPower(saveBool = False,
                            N_IR = 50,
                            OmegaR = 1e6*2*np.pi,
                            P_smin = -6,
                            P_smax = 2,
                            I_p = 2.101e8,
                            bw = 0.01e9*2*np.pi,
                            Npoints = 100,
                            A = 10e-6**2,
                            Larr = [1e-5,1e-3, 125e-3]):
    """Sweep IR probe power (for several absorption path lengths L) and plot the resulting shot-noise-limited sensitivity."""
    
    
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.5, PRQuantumWidthHalf*0.6,  
        r"ss", 
        r'',
        labelPadX = 11.5, labelPadY = 0, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False, spineWidth = 0.15
    )
    
    colors = [colorLightBlue, colorBlue,colorDarkBlue]
    for j, L in enumerate(Larr):
        Parr = np.logspace(P_smin, P_smax, N_IR)
        I_IR = Parr/A
        p7 = np.zeros([N_IR])
        R_on = np.zeros([N_IR])
        R_off = np.zeros([N_IR])
        C = np.zeros([N_IR])
        M = rho_NV
        alpha0 = M*sigma_s
        print(alpha0)
        nu = np.zeros([N_IR])
        detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
        for i, I_s in enumerate(I_IR):
            popOn =         nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
            popOff = nv_steady_state_default(
                Omega = OmegaR*0,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
            )
                
            R_on[i] = (popOn[7])
            R_off[i] = (popOff[7])
            C[i] = -(R_off[i]-R_on[i])/(R_off[i]+R_on[i])*2
    
            for _bisection_attempt in range(MAX_BISECTION_ITERS):  # bounded bisection search (was 'while True:')
                detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
                Cdet = np.zeros(Npoints)
                for k, det in enumerate(detunings):
                    popDet =    nv_steady_state_default(
                        Omega = OmegaR,
                        I_s = I_s,
                        I_p = I_p,
                        detuning = det,
                        zeeman = zeeman,
                    )
                    Cdet[k] = -(R_off[i] - (popDet[7]))/R_off[i]
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                # plt.plot(detunings/1e9/(2*np.pi),Cdet)
                print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
                # plt.show()
                if np.size(FWHM) < 1:
                    print('FWHM fitting didnt work')
                elif (np.max(Cdet) - np.min(Cdet))/np.max(Cdet) < 0.8:
                    bw = bw*2
                    continue
                elif FWHM < bw/1e9/(2*np.pi) and FWHM > bw/5/1e9/(2*np.pi):
                    nu[i] = FWHM[0]*1e9
                    break
                elif FWHM >= bw/2:
                    bw = bw*2
                    continue
                else:
                    bw = bw/3
                    continue
                    
                    
    
        TOff = np.exp(-alpha0*R_off*L)
        TOn = np.exp(-alpha0*R_on*L) 
        C = np.abs(TOff-TOn)/(TOff + TOn)*2
        print(C)
        
        lamb = 1042e-9
        eta = calcSensitivityCW(nu, C, Parr*(TOff + TOn)/2, lamb)
        eta0 = calcSensitivityCW(nu[0], C[0], Parr*(TOff[0] + TOn[0])/2, lamb)
        print(eta)
        
        ax.loglog(I_IR*1e-4, eta, lw = 0.2/0.358, color = colors[j] )
        ax.loglog(I_IR*1e-4, eta0, 'k--', lw = 0.2/0.358)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    min_x_value=1e0
    max_x_value = 1e8
    min_y_value=1e-14
    max_y_value = 1e-6
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([min_y_value/4, max_y_value*4])
    ax.set_xlim([min_x_value/4, max_x_value*4])

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')



    # ax.spines['left'].set_bounds(min_y_value,max_y_value)
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NaiveSensitivity.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    plt.show()

    
    
def solvePopulationSteadyState(saveBool = False, bool1042 = False):
    """Plot steady-state level populations vs IR intensity and vs green intensity."""
    # IR Power Sweep dark state paper all populations
    
    fig, ax = plot_custom(
        PRQuantumWidthHalf, PRQuantumWidthHalf, 
        [r"$I_p$ (W/m$^2$)", r"$I_s$ (W/m$^2$)"],
        [r'',r''],
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = False, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    for axx in ax:
        axx.set_xscale('log');
        # axx.set_yscale('log');
    N_IR = 100
    NV0power1 = np.logspace(6, 12, N_IR)
    I_pp = [2.101e8]
    I_IR = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR) #W/cm^2 * cm^2/m^2
    OmegaR = 1e0*2*np.pi
    p = np.zeros([N_IR, 11])
    # detuning = 0
    for I_p in I_pp:
        for i, I_s in enumerate(I_IR):
            result =  nv_steady_state_default(
                Omega = OmegaR,
                I_s = I_s,
                I_p = I_p,
                detuning = detuning,
                zeeman = zeeman,
                bool1042 = bool1042,
            )
            p[i,:] = result
        labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
                  r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
        colors = ['#3dfe00ff','#35db00ff','#218800ff',
                  '#fe7900ff','#fe0000ff','#fe0076ff',
                  '#190cffff','#800000ff','#800cffff', '#ff0cf4ff', '#feb400ff'];
        print(np.max(p[:,7]))
        for i in range(11):
            if i in [7, 10]:
                ax[1].plot(NV0power1, p[:, i], color = colors[i], lw = 0.5, label = labels[i])
            if i in [3]:
                ax[1].plot(NV0power1, p[:, i]+p[:, i+1]+p[:, i+2], color = colors[i], lw = 0.5, label = labels[i])
            
    # ax.legend()
    ax[1].set_ylim([1e-3, 1])
    ax[1].set_xscale('log');
    
    NV0power1 = np.logspace(6, 12, N_IR)
    I_s = 1e-10
    I_g = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR) #W/cm^2 * cm^2/m^2

    p = np.zeros([N_IR, 11])
    # detuning = 0
    
    for i, I_p in enumerate(I_g):
        result =  nv_steady_state_default(
            Omega = OmegaR,
            I_s = I_s,
            I_p = I_p,
            detuning = detuning,
            zeeman = zeeman,
        )
        p[i,:] = result
    labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
              r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
    colors = ['#3dfe00ff','#35db00ff','#218800ff',
              '#fe7900ff','#fe0000ff','#fe0076ff',
              '#190cffff','#800000ff','#800cffff', '#ff0cf4ff', '#feb400ff'];
    print(np.max(p[:,7]))
    for i in range(11):
        if i in [6, 7, 10]:
            ax[0].plot(NV0power1, p[:, i], color = colors[i], lw = 0.5, label = labels[i])
        if i in [3]:
            ax[0].plot(NV0power1, p[:, i]+p[:, i+1]+p[:, i+2], color = colors[i], lw = 0.5, label = labels[i])
            
    # ax.legend()
    ax[0].set_ylim([1e-5, 1])
    plt.tight_layout()
    ax[0].set_xscale('log');
    ax[0].set_yscale('log');
    ax[1].set_yscale('log');
    
    
    plt.tight_layout()
    plt.xscale('log');
    plt.yscale('log');
    
    plt.show()
    
def performOptimization():
    """Run an Optuna study minimizing objective (IR-absorption sensitivity) and print the best I_p/I_s/OmegaR found."""
    # --- Create study ---
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

    # --- Run optimization ---
    study.optimize(
        objective,
        n_trials=50,              # increase if slow landscape
        show_progress_bar=True,
        n_jobs=1                   # >1 only if solver is thread-safe
    )

    elapsed = time.time() - start_time

    # -------------------------------
    # Results
    # -------------------------------
    print("\nOptimization finished")
    print(f"Elapsed time: {elapsed/60:.1f} min")
    print("Best sensitivity:", study.best_value)

    best = study.best_params
    print(
        "Best parameters:\n"
        f"  I_p     = {10**best['log10_Ip']:.3e}\n"
        f"  I_s     = {10**best['log10_Is']:.3e}\n"
        f"  OmegaR  = {10**best['log10_OmegaR']:.3e}"
    )

def performNVPLVisOptimization():
    """Run an Optuna study minimizing objectiveVis (green/NV0-PL sensitivity) and print the best I_p/OmegaR found."""
    # --- Create study ---
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

    # --- Run optimization ---
    study.optimize(
        objectiveVis,
        n_trials=100,              # increase if slow landscape
        show_progress_bar=True,
        n_jobs=1                   # >1 only if solver is thread-safe
    )

    elapsed = time.time() - start_time

    # -------------------------------
    # Results
    # -------------------------------
    print("\nOptimization finished")
    print(f"Elapsed time: {elapsed/60:.1f} min")
    print("Best sensitivity:", study.best_value)

    best = study.best_params
    print(
        "Best parameters:\n"
        f"  I_p     = {10**best['log10_Ip']:.3e}\n"
        f"  OmegaR  = {10**best['log10_OmegaR']:.3e}"
    )



if __name__ == "__main__":
    saveBool = False
    
    N = 31
    OmegaRmin = 4
    OmegaRmax = 7
    I_pmin = 7
    I_pmax = 8
    I_smin = 9
    I_smax = 11
    
    # solvePopulationIRGreenOmegaRDetuning(
    #                             I_IR = np.array([0.1, 1, 5])*I_sOpt,
    #                             I_green = np.array([0.2, 1, 5])*I_pOpt,
    #                             OmegaRArr = np.array([0.5, 1, 2])*OmegaROpt,
    #                             saveBool = saveBool,
    #                             N = 3,
    #                             I_sOpt = I_sOpt,
    #                             I_pOpt = I_pOpt,
    #                             OmegaROpt = OmegaROpt)
    
    
    # solvePopulationLineplots()
    # solveTimeDependentPopulation(OmegaR =  2*np.pi*5e6, I_s = I_sOpt, I_p = I_pOpt, detuning = detuning, saveBool = saveBool)
    # solvePopulationSteadyState(saveBool, bool1042 = True)
    # solvePopulationSteadyStateOmegaR(saveBool, bool1042 = False)
    # solvePopulationSteadyStateContrast(saveBool = saveBool, bool1042 = False)
    # solvePopulationDetuningHyperfineLineplot(saveBool)
    # solveNVPL_GreenOmegaR()
    
    # solvePopulationDetuningLineplot(saveBool = False, 
    #                                 OmegaR = OmegaROptRadps,          
    #                                 I_p = I_pOpt,
    #                                 I_s = I_sOpt,
    #                                 bw = 0.01e9*2*np.pi,
    #                                 Npoints = 201,
    #                                 fastBool = False)
    # solveTimeDependentPopulation(OmegaR =  OmegaROptRadps, 
    #                              I_s = I_sOpt*5, I_p = I_pOpt, 
    #                              detuning = detuning, saveBool = saveBool)
    # solvePopulationDetuningLineplotPL(saveBool = False, 
    #                                 OmegaR = OmegaROptRadps_vis,          
    #                                 I_p = I_pOpt_vis,
    #                                 I_s = 0,
    #                                 bw = 0.01e9*2*np.pi,
    #                                 Npoints = 201,
    #                                 fastBool = False)
    
    
    
    # solveSensitivityVaryIRPower(saveBool = saveBool)
    # performOptimization()
    # performNVPLVisOptimization()
    

    
