# -*- coding: utf-8 -*-
"""
NV center 11-level model in a cavity (microdisk)
@author: nicho (refactor)
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
from numba import njit
from linbladMESolverFunctionChargeConversionLiouvillian import nv_steady_state, nv_steady_state_fast, nv_steady_state_IRRate
from commonlabcode.plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible
from scipy.io import loadmat
from tqdm import tqdm
from scipy.optimize import differential_evolution
import optuna
import time
import multiprocessing as mp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

counter = mp.Value('i', 0)

labelfontsize = 9
numfontsize = 9
textwidth = 174/25.4#8.5 - 2 # in inches
PRQuantumWidthFull = textwidth

# ------------------------------
# Physical / experimental inputs
# ------------------------------

# intrinsic transition rates (power dependent rates are magic numbers)
gamma_eg=66e6
gamma_isc_0=7.9e6
gamma_isc_1=53e6
gamma_isc_g0=1e6
gamma_isc_g1=0.7e6
gamma_s=1e9
gamma_NV0_eg = 53e6  #k^0_f
gamma_NV0_em = 22540.55 #k_75
gamma_NV0_mg = 357687.5   #k_56

# intrinsic absorption cross sections
sigma_p = 3e-21 # m^%-2
sigma_s = 2e-22 # m^%-2

# physical constants
c = 299792458.0 #m/s
h = 6.62607015e-34      # Planck constant [J·s]

# diamond/NV material constants
gamma = 1.761e11/2/np.pi # Hz/T
carbon_density = 1.76e29  # atoms/m³
rho_NV = carbon_density*4.5e-6 # atoms/m^3
T2s = 500e-9
gamma_phi = 1/np.pi/T2s
nd = 2.4
hyperfine = 2.16e6

# experimental constants
eta_coll = 1
eta_loss_post_taper = 1
respDet = 4e4 # V/W detector responsivity
lamb_s = 1042e-9   # IR probe
lamb_p = 532e-9    # green pump
w_s = c/lamb_s   # IR probe
w_p = c/lamb_p    # green pump
detuning = 0.1e9*2*np.pi
zeeman = 0.1e9*2*np.pi

paramsRatesReal = np.array([
    1e7, #kem W
    1.8e7, # Ke0
    66150.4, #K25 g [2]
    7.11e5, #K51 g [4]
    25690.73, #K71 [15]
    25474.34, #K73 [17]
    1012.032, #K74 [6]
    4853.41, #K46 [8]
    0.371923, # k25 double [3]
    439, # k25 single [1]
    0.006162, # k51 [16]
    0.1,        # k73 [18]
    1.71e3,   # k74 [5]
    0.1,        # k46 [9]
    208.5184, # k71 [14]
    43546.84, # k36 [10]
    20000.02, # k36 [11]
    943, # k43 [7]
    22540.55, #k_75
    357687.5   #k_56
    ])

paramsRatesReal = np.array([2.11092428e+07, 8.42670441e+06, 9.87669580e-01, 1.84244697e+07,
 1.63887472e+04, 7.21569264e+01, 8.47372932e+03 ,7.56363296e-02,
 4.30538553e-01, 3.96020576e+02, 5.77818321e-01 ,1.00050274e-01,
 5.11859226e+03, 9.67228125e-06, 6.77459364e+01 ,3.76675450e+06,
 5.30853269e+02, 2.05445831e+04, 1.85925176e+05 ,1.59937326e+07])

# paramsRatesReal = np.array([2.16835246e+07, 3.55505362e+08, 1.43381055e+05, 2.05729005e+08,
#  5.36217299e+03 ,1.65764710e+02, 5.63278596e+01, 1.15519940e+03,
#  9.08293994e-01, 6.40706791e+02, 8.74943457e-02, 1.73536611e-02,
#  4.98479124e+02, 4.94189201e+01, 7.58960481e-01, 8.99368481e+01,
#  4.74493324e+03, 7.66486695e+01, 4.73949446e+03, 3.93566781e+07,])


# paramsRatesReal = np.array([2.85478211e+00, 8.01730333e-02, 1.08992131e-03, 1.27300994e-03,
#   3.12687288e+00, 1.34819446e+00, 1.62573825e+00, 1.35763412e-01,
#   8.69454551e-03, 8.17862301e-02, 5.61416230e-03, 8.78249635e-01,
#   1.56044197e-03, 1.00000000e+00, 1.32760924e+01,]) 

# # paramsRatesReal *= paramsScale

# paramsRatesReal = np.array([4.21077617e+07, 2.04066394e+06, 4.64367028e+02, 1.03720868e+03,
#   8.18945308e+01, 1.12421365e+05, 2.58572302e+03, 5.65061214e+02,
#   3.13494789e-03, 4.01280257e+01, 3.27503613e-05, 0.00000000e+00,
#   1.11155680e+03, 1.00008162e-03, 2.88145543e+03, 43546.84, 20000.02])

paramsRatesOnes = np.ones(20)

def solveTimeDependentPopulation(saveBool = False,
                                    OmegaR = 2*np.pi*1e6,
                                    I_p = 1e8,
                                    I_s = 1e8, bool1042 = False):

    result, tlist = nv_steady_state(
                                    Omega = OmegaR, #rad/s
                                    I_s = I_s,   #rad/s
                                    I_p = I_p, #W/m^2
                                    gamma_phi = gamma_phi,   #Hz
                                    detuning = detuning, #rad/s
                                    zeeman   = zeeman, #rad/s
                                    steadystatebool=False, 
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
                                    c = c, #m/s
                                    h = h,      # Planck constant [J·s]
                                    paramsRates = paramsRatesOnes,
                                    paramsRatesReal= paramsRatesReal,
                                    bool1042 = bool1042
                                    )
    
    # ---------------------------
    # Plot populations
    # ---------------------------
    fig, ax = plot_custom(
        PRQuantumWidthFull*0.4, 0.58*PRQuantumWidthFull, 
        r"$\lambda$ (nm)", 
        r'$T$ (au)',
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
              r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
    colors = ['#b3b3b3ff','#666666ff','#000000ff','#ff5e09ff','#fe9800ff','#ffbd09ff','#800000ff','#190cffff',
              '#800cffff', '#fe0000ff', '#ff0cf1ff'];
    for i in range(11):
        ax.plot(tlist*1e6, result.expect[i], label=labels[i], color = colors[i], lw = 0.5)
    
    ax.set_yscale('log'); 
    # plt.yscale('log');
    ax.set_xlabel(r"time ($\upmu$s)")
    ax.set_ylabel(r"population, $m_i$ (au)")
    # plt.legend(fontsize = labelfontsize, loc = 'lower right')
    # plt.tight_layout()
    
    tmin = 0
    tmax = 2
    ax.set_ylim([8e-5, 1.1e0])
    ax.spines['bottom'].set_bounds(tmin, tmax)
    xtick = ax.get_xticks()
    ax.set_xticklabels([rf"${val:.1f}$" for val in xtick])
    
    ax.spines['left'].set_bounds(1e-4,1)
    ytick = ax.get_yticks(minor = True)
    filt = np.logical_and(ytick > 1e-4, ytick < 1) 
    ytick = ytick[filt]
    ax.set_yticks(ytick,minor = True)
    ax.spines['left'].set_bounds(1e-4,1)
    
    ax.set_xticklabels([rf"${val:.1f}$" for val in xtick])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
    
    if saveBool:
        fig.savefig(
            "./figures/" + "NVTimeSolution.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    
    plt.show()

def solvePeakWidthAndHeight(x, y):
    peaks, _ = find_peaks(y)
    FWHM, _, _, _  = peak_widths(y, peaks, rel_height=0.5)
    try:
        maxSignal = y[peaks[0]]
    except:
        maxSignal= np.max(y)
    FWHM = FWHM*(x[1] - x[0])
    return maxSignal, FWHM
    
def calcSensitivityCW(nu, C, P, lamb):
    return nu/C 

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
                    c = c,
                    bool1042 = True):
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
                    
                        popOn = nv_steady_state(
                                                        Omega = OmegaR, #rad/s
                                                        I_s = I_s,   #rad/s
                                                        I_p = I_p, #W/m^2
                                                        gamma_phi = gamma_phi,   #Hz
                                                        detuning = (det), #rad/s
                                                        zeeman   = zeeman, #rad/s
                                                        steadystatebool=True, 
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
                                                        c = c, #m/s
                                                        h = h,      # Planck constant [J·s]
                                                        paramsRates = paramsRatesOnes,
                                                        paramsRatesReal = paramsRatesReal,
                                                        bool1042 = bool1042
                                                        )
                        popOff = nv_steady_state(
                                                        Omega = OmegaR*0, #rad/s
                                                        I_s = I_s,   #rad/s
                                                        I_p = I_p, #W/m^2
                                                        gamma_phi = gamma_phi,   #Hz
                                                        detuning = (det), #rad/s
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
                                                        c = c, #m/s
                                                        h = h,      # Planck constant [J·s]
                                                        paramsRates = paramsRatesOnes,
                                                        paramsRatesReal = paramsRatesReal,
                                                        bool1042 = bool1042
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
    
    
def solveSensitivitySingle(OmegaR = 0,          
                            I_p = 1e7,
                            I_s = 1e12,
                            bw = 0.1e9*2*np.pi,
                            Npoints = 401,
                            lamb = 1042e-9,
                            bool1042 = True):
    
    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
    
    delpop = np.zeros(Npoints)
    mOn = np.zeros(Npoints)
    mOff = np.zeros(Npoints)
    
    for k, det in enumerate(detunings):
        # kappaOn = nv_steady_state_IRRate(
        #                                 Omega = OmegaR, #rad/s
        #                                 I_s = I_s,   #W/m^2
        #                                 I_p = I_p, #W/m^2
        #                                 gamma_phi = gamma_phi,   #Hz
        #                                 detuning = (det), #rad/s
        #                                 zeeman   = zeeman, #rad/s
        #                                 steadystatebool=True, 
        #                                 gamma_eg=gamma_eg,
        #                                 gamma_isc_0=gamma_isc_0,
        #                                 gamma_isc_1=gamma_isc_1,
        #                                 gamma_isc_g0=gamma_isc_g0,
        #                                 gamma_isc_g1=gamma_isc_g1,
        #                                 gamma_s=gamma_s,
        #                                 gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
        #                                 gamma_NV0_em = gamma_NV0_em, #k_75
        #                                 gamma_NV0_mg = gamma_NV0_mg,   #k_56
        #                                 sigma_p = sigma_p, # m^%-2
        #                                 sigma_s = sigma_s, # m^%-2
        #                                 lamb_s = lamb_s,   # IR probe
        #                                 lamb_p = lamb_p,    # green pump
        #                                 c = c, #m/s
        #                                 h = h,      # Planck constant [J·s]
        #                                 paramsRates = paramsRatesOnes,
        #                                 paramsRatesReal = paramsRatesReal,
        #                                 bool1042 = bool1042
        #                                 )
        popOn = nv_steady_state(
                                        Omega = OmegaR, #rad/s
                                        I_s = I_s,   #W/m^2
                                        I_p = I_p, #W/m^2
                                        gamma_phi = gamma_phi,   #Hz
                                        detuning = (det), #rad/s
                                        zeeman   = zeeman, #rad/s
                                        steadystatebool=True, 
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
                                        c = c, #m/s
                                        h = h,      # Planck constant [J·s]
                                        paramsRates = paramsRatesOnes,
                                        paramsRatesReal = paramsRatesReal,
                                        bool1042 = bool1042
                                        )
        popOff = nv_steady_state(
                                        Omega = OmegaR*0, #rad/s
                                        I_s = I_s,   #W/m^2
                                        I_p = I_p, #W/m^2
                                        gamma_phi = gamma_phi,   #Hz
                                        detuning = (det), #rad/s
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
                                        c = c, #m/s
                                        h = h,      # Planck constant [J·s]
                                        paramsRates = paramsRatesOnes,
                                        paramsRatesReal = paramsRatesReal,
                                        bool1042 = bool1042
                                        )
        # kappaOff = nv_steady_state_IRRate(
        #                                 Omega = OmegaR*0, #rad/s
        #                                 I_s = I_s,   #W/m^2
        #                                 I_p = I_p, #W/m^2
        #                                 gamma_phi = gamma_phi,   #Hz
        #                                 detuning = (det), #rad/s
        #                                 zeeman   = zeeman, #rad/s
        #                                 steadystatebool=True, 
        #                                 tEnd = 2e-6,
        #                                 gamma_eg=gamma_eg,
        #                                 gamma_isc_0=gamma_isc_0,
        #                                 gamma_isc_1=gamma_isc_1,
        #                                 gamma_isc_g0=gamma_isc_g0,
        #                                 gamma_isc_g1=gamma_isc_g1,
        #                                 gamma_s=gamma_s,
        #                                 gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
        #                                 gamma_NV0_em = gamma_NV0_em, #k_75
        #                                 gamma_NV0_mg = gamma_NV0_mg,   #k_56
        #                                 sigma_p = sigma_p, # m^%-2
        #                                 sigma_s = sigma_s, # m^%-2
        #                                 lamb_s = lamb_s,   # IR probe
        #                                 lamb_p = lamb_p,    # green pump
        #                                 c = c, #m/s
        #                                 h = h,      # Planck constant [J·s]
        #                                 paramsRates = paramsRatesOnes,
        #                                 paramsRatesReal = paramsRatesReal,
        #                                 bool1042 = bool1042
        #                                 )

        delpop[k] = (popOn[7] - popOff[7])
        mOn[k] = (popOn[7])
        mOff[k] = (popOff[7])
        
        # delpop[k] = np.sum(popOn*kappaOn - popOff*kappaOff)
        # mOn[k] = np.sum(popOn*kappaOn)
        # mOff[k] = np.sum(popOff*kappaOff)
    C = np.max(delpop)
    maxSignal, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), mOn)
    nu = FWHM[0]
    mOn = maxSignal
    mOff = np.max(mOff)
    
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

    # --- Log-parameterized search space ---
    log_Ip = trial.suggest_float("log10_Ip", 7, 10)
    log_Is = trial.suggest_float("log10_Is", 8, 11)
    log_OmegaR = trial.suggest_float("log10_OmegaR", 5, 7)

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
                                    bw=0.1e9*2*np.pi,
                                    Npoints=401,
                                    lamb=1042e-9
                                    )

    # --- Report intermediate value (for pruning) ---
    trial.report(sens, step=0)

    # Optional pruning if clearly bad
    if trial.should_prune():
        raise optuna.TrialPruned()

    return sens

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
                                    OmegaR = gamma_phi,          
                                    I_p = 1e7,
                                    I_s = 1e12,
                                    bw = 0.01e9*2*np.pi,
                                    Npoints = 101,
                                    fastBool = False,
                                    bool1042 = False):
    # detuning
    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
    kappaTotal = np.zeros(Npoints)
    p7 = np.zeros(Npoints)
    for i, det in enumerate(detunings):
        if fastBool:
            pop = nv_steady_state_fast(
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = (det), #rad/s
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
            
        else:
            GammaIRTot = nv_steady_state_IRRate(
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = (det), #rad/s
                                            zeeman   = zeeman, #rad/s
                                            steadystatebool=True, 
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
            pop = nv_steady_state(
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = (det), #rad/s
                                            zeeman   = zeeman, #rad/s
                                            steadystatebool=True, 
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
        defectIRLoss = pop*GammaIRTot
        p7[i] = (pop[7])
        kappaTotal[i] = np.sum(defectIRLoss)
    
    peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), p7)
        
    plt.figure(figsize=(4,4))
    plt.plot(detunings/1e9/(2*np.pi), p7/np.min(p7))
    plt.plot(detunings/1e9/(2*np.pi), kappaTotal/np.min(kappaTotal))
    print('Max p7: ', np.max(p7), '   ', np.max(p7)/np.min(p7))
    print('Min p7: ', np.min(p7))
    print('FWHM: ', FWHM)
    plt.ylabel(r'change in population (au)')
    plt.xlabel(r'microwave detuning (GHz)')
    plt.show()
    
def solvePopulationDetuningLineplotPL(saveBool = False, 
                                    OmegaR = gamma_phi,          
                                    I_p = 1e7,
                                    I_s = 1e12,
                                    bw = 0.01e9*2*np.pi,
                                    Npoints = 101,
                                    fastBool = False,
                                    bool1042 = False):
    # detuning
    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
    kappaTotal = np.zeros(Npoints)
    PLRate = np.zeros(Npoints)
    for i, det in enumerate(detunings):
        pop = nv_steady_state(
                                        Omega = OmegaR, #rad/s
                                        I_s = I_s,   #rad/s
                                        I_p = I_p, #W/m^2
                                        gamma_phi = gamma_phi,   #Hz
                                        detuning = (det), #rad/s
                                        zeeman   = zeeman, #rad/s
                                        steadystatebool=True, 
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
                                        c = c, #m/s
                                        h = h,      # Planck constant [J·s]
                                        paramsRates = paramsRatesOnes,
                                        paramsRatesReal = paramsRatesReal,
                                        bool1042 = bool1042
                                        )
        PLRate[i] = (pop[3] + pop[4]+pop[5])*gamma_eg
    
    # peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), PLRate)
        
    plt.figure(figsize=(4,4))
    plt.plot(detunings/1e9/(2*np.pi), PLRate/np.min(PLRate))
    plt.plot(detunings/1e9/(2*np.pi), kappaTotal/np.min(kappaTotal))
    print('Max Rate: ', np.max(PLRate), '   ', np.max(PLRate)/np.min(PLRate))
    print('Min Rate: ', np.min(PLRate))
    # print('FWHM: ', FWHM)
    plt.ylabel(r'change in population (au)')
    plt.xlabel(r'microwave detuning (GHz)')
    plt.show()

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
#                                         c = c, #m/s
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
                                        c = c, #m/s
                                        h = h      # Planck constant [J·s]
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
                                        c = c, #m/s
                                        h = h      # Planck constant [J·s]
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
                                        c = c, #m/s
                                        h = h      # Planck constant [J·s]
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
    
    I_IR = np.logspace(I_smin, I_smax, N_IR)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    p7 = np.zeros([N_IR, N_OmegaR])
    p7On = np.zeros([N_IR, N_OmegaR])
    p7Off = np.zeros([N_IR, N_OmegaR])
    for i, I_s in enumerate(I_IR):
        for j, OmegaR in enumerate(OmegaRArr):
            populations =         nv_steady_state(
                                            Omega = OmegaR*2*np.pi, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = detuning, #rad/s
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
            populationsDetuning = nv_steady_state(
                                            Omega = OmegaR*0, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = detuning, #rad/s
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
            # p7[i,j] = (populations[7] - populationsDetuning[7])
            p7On[i,j] = (populations[7])
            p7Off[i,j] = (populationsDetuning[7])
            p7[i,j] = (p7On[i,j])/p7Off[i,j]
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
    ax.set_xlabel(r"$I_{\text{S}}$ (kW/cm$^2$)", fontsize = labelfontsize)
    ax.set_ylabel(r'$\Omega_{\text{R}}/2\pi$ (MHz)', fontsize = labelfontsize)
        
    X, Y = np.meshgrid(np.log10(I_IR), np.log10(OmegaRArr))
    # im = ax.pcolormesh(X-7, Y-7, np.transpose(p7), cmap = 'hot')
    im = plt.contourf(X-7, Y-6, np.transpose(p7), 10, cmap = 'viridis')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$m_7^{\text{(on)}}/m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)
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
            "./figures/" + "NVPopulationContrastOmegaRIR.png", dpi=300, bbox_inches="tight",transparent=True
            )
        
            
def solvePopulationGreenOmegaR(saveBool = False,
                            N_green = 10,
                            N_OmegaR = 10,
                            OmegaRmin = 4,
                            OmegaRmax = 7,
                            I_pmin = 5,
                            I_pmax = 8,
                            I_s = 1e9,
                            fastBool = False,
                            OmegaROpt = 0,
                            I_pOpt = 0,
                            bool1042 = False):
    
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    OmegaRArr = np.logspace(OmegaRmin, OmegaRmax, N_OmegaR)
    p7 = np.zeros([N_green, N_OmegaR])
    p7On = np.zeros([N_green, N_OmegaR])
    p7Off = np.zeros([N_green, N_OmegaR])
    for i, I_p in enumerate(I_green):
        for j, OmegaR in enumerate(OmegaRArr):
            if fastBool:
                populations =         nv_steady_state_fast(
                                                Omega = OmegaR*2*np.pi, #rad/s
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
                                                c = c, #m/s
                                                h = h,      # Planck constant [J·s]
                                                paramsRates = paramsRatesOnes,
                                                paramsRatesReal = paramsRatesReal
                                                )


                populationsDetuning = nv_steady_state_fast(
                                                Omega = OmegaR*0, #rad/s
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
                                                c = c, #m/s
                                                h = h,      # Planck constant [J·s]
                                                paramsRates = paramsRatesOnes,
                                                paramsRatesReal = paramsRatesReal
                                                )
            else:
                populations =         nv_steady_state(
                                                Omega = OmegaR*2*np.pi, #rad/s
                                                I_s = I_s,   #rad/s
                                                I_p = I_p, #W/m^2
                                                gamma_phi = gamma_phi,   #Hz
                                                detuning = detuning, #rad/s
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
                                                c = c, #m/s
                                                h = h,      # Planck constant [J·s]
                                                paramsRates = paramsRatesOnes,
                                                paramsRatesReal = paramsRatesReal,
                                                bool1042 = bool1042
                                                )
                populationsDetuning = nv_steady_state(
                                                Omega = OmegaR*0, #rad/s
                                                I_s = I_s,   #rad/s
                                                I_p = I_p, #W/m^2
                                                gamma_phi = gamma_phi,   #Hz
                                                detuning = detuning, #rad/s
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
                                                c = c, #m/s
                                                h = h,      # Planck constant [J·s]
                                                paramsRates = paramsRatesOnes,
                                                paramsRatesReal = paramsRatesReal,
                                                bool1042 = bool1042
                                                )
            # p7[i,j] = (populations[7] - populationsDetuning[7])
            p7On[i,j] = (populations[7])
            p7Off[i,j] = (populationsDetuning[7])
            p7[i,j] = (p7On[i,j])/p7Off[i,j]
            # p7[i,j] = (p7On[i,j] - p7Off[i,j])
    
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.33, 0.28*PRQuantumWidthFull, 
       r"$I_{\text{p}}$ (kW/cm$^2$)", 
       r'$\Omega_{\text{R}}/2\pi$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    X, Y = np.meshgrid(np.log10(I_green), np.log10(OmegaRArr))
    im = ax.contourf(X-7, Y-6, np.transpose(p7), 10, cmap = 'viridis')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$m_7^{\text{(on)}}/m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)
    yticks = ax.get_yticks()
    # yticks = [-2, -1, 0, 1]
    ax.set_xticks([-1, 1, 3])
    xticks = ax.get_xticks()
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
    # ticks = [0, np.max(p7)]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in ticks], fontsize = numfontsize)
    ax.scatter(np.log10(I_pOpt)-7, np.log10(OmegaROpt)-6, marker = '*', color = 'red')
    plt.show()

    if saveBool:
        fig.savefig(
            "./figures/" + "NVPopulationContrastOmegaRGreen.png", dpi=300, bbox_inches="tight",transparent=True
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
    
    I_IR = np.logspace(I_smin, I_smax, N_IR)
    I_green = np.logspace(I_pmin, I_pmax, N_green)
    p7 = np.zeros([N_IR, N_green])
    p7On = np.zeros([N_IR, N_green])
    p7Off = np.zeros([N_IR, N_green])
    for i, I_s in enumerate(I_IR):
        for j, I_p in enumerate(I_green):
            populations =         nv_steady_state(
                                            Omega = OmegaR*2*np.pi, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = detuning, #rad/s
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
            populationsDetuning = nv_steady_state(
                                            Omega = OmegaR*0, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            gamma_phi = gamma_phi,   #Hz
                                            detuning = detuning, #rad/s
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s]
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            bool1042 = bool1042
                                            )
            # p7[i,j] = (populations[7] - populationsDetuning[7])
            p7On[i,j] = (populations[7])
            p7Off[i,j] = (populationsDetuning[7])
            p7[i,j] = (p7On[i,j])/p7Off[i,j]
            # p7[i,j] = (p7On[i,j] - p7Off[i,j])
    
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
    im = ax.contourf(X-7, Y-7, np.transpose(p7), 10, cmap = 'viridis')
    plt.tight_layout()
    
    cbar = plt.colorbar(im, orientation = 'vertical', location = 'right')
    cbar.set_label(r'$m_7^{\text{(on)}}/m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)
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
            "./figures/" + "NVPopulationContrastGreenIR.png", dpi=300, bbox_inches="tight",transparent=True
            )
        



def solvePopulationLineplots(saveBool = False):
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
    
    ms = 1
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
    
    NVmcounts1 = dataNVm2['pm_norm_inp_final'][:,0,0]/0.95
    NVmcounts2 = dataNVm2['pm_norm_inp_final'][:,1,0]/0.95
    NVmcounts3 = dataNVm2['pm_norm_inp_final'][:,2,0]/0.95
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
    I_green = 1e-3/(0.3e-6**2*np.pi)
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
                c = c,
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
                result =  nv_steady_state(
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s],
                                            paramsRates = paramsRatesOnes,
                                            paramsRatesReal = paramsRatesReal,
                                            verboseResults = True
                                            )
                p[i,:] = result
            ax[1].plot(I_IR_model, (p[:,3] + p[:,4] + p[:,5])/np.max(p[0,3] + p[0,4] + p[0,5]), color = colors[j], lw = lw)
            ax[0].plot(I_IR_model, p[:,10]/np.max(p[0,10]), color = colors[j], lw = lw)
        
    else:
        x0 = np.ones(len(paramsRatesOnes))   # your initial guess
        
        lower = 0.00001 * x0
        upper = 10000.0 * x0
        
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
        print(params_opt)
        
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
                                            c = c, #m/s
                                            h = h,      # Planck constant [J·s],
                                            paramsRates = params_opt,
                                            paramsRatesReal = paramsRatesReal
                                            )
                p[i,:] = result
            ax[1].plot(I_IR_model, (p[:,3] + p[:,4] + p[:,5])/np.max(p[0,3] + p[0,4] + p[0,5]))
            ax[0].plot(I_IR_model, p[:,10]/np.max(p[0,10]))
    
    plt.tight_layout()
    plt.xscale('log');
    if saveBool:
        fig.savefig(
            "./figures/" + "DarkStateFigure.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    plt.show()
    
def solvePopulationSteadyState(saveBool = False, bool1042 = False):
    # IR Power Sweep dark state paper all populations
    
    fig, ax = plot_custom(
        PRQuantumWidthFull*0.4, 0.45*PRQuantumWidthFull, 
        r"$I_s$ (W/m$^2$)", 
        r'',
        labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    
    
    ax.set_xscale('log');
    ax.set_yscale('log');
    N_IR = 100
    NV0power1 = np.logspace(4, 12, N_IR)
    I_p = 1e-3/(0.3e-6**2*np.pi)*1e-1
    I_IR = np.logspace(np.log10(np.min(NV0power1)), np.log10(np.max(NV0power1)), N_IR) #W/cm^2 * cm^2/m^2
    OmegaR = 1e0
    p = np.zeros([N_IR, 11])
    # detuning = 0
    
    for i, I_s in enumerate(I_IR):
        result =  nv_steady_state_fast(
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
                c = c,
                h = h,
                paramsRates = paramsRatesOnes,
                paramsRatesReal = paramsRatesReal,
                bool1042 = bool1042
            )
        p[i,:] = result
    labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',
              r'$|6\rangle$',r'$|7\rangle$',r'$|8\rangle$',r'$|9\rangle$',r'$|10\rangle$']
    colors = ['#b3b3b3ff','#666666ff','#000000ff','#ff5e09ff','#fe9800ff','#ffbd09ff','#800000ff','#190cffff',
              '#800cffff', '#ff0cf1ff', '#fe0000ff'];
    for i in range(11):
        ax.plot(NV0power1, p[:, i], color = colors[i], lw = 0.5, label = labels[i])
        
    # ax.legend()
    ax.set_ylim([1e-6, 1])
    plt.tight_layout()
    plt.xscale('log');
    
    plt.show()
    
def performOptimization():
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
        n_trials=300,              # increase if slow landscape
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


if __name__ == "__main__":
    saveBool = True
    
    # solveTimeDependentPopulation(saveBool, bool1042 = False)
    # solvePopulationSteadyState(saveBool, bool1042 = False)
    # solvePopulationDetuningLineplot(saveBool = False, 
    #                                 OmegaR = 2*np.pi*0.16e6,          
    #                                 I_p = 1e8,
    #                                 I_s = 1e9,
    #                                 bw = 0.01e9*2*np.pi,
    #                                 Npoints = 201,
    #                                 fastBool = False)
    # solvePopulationDetuningHyperfineLineplot(saveBool)
    
    N = 40
    NOmega = 12
    OmegaRmin = 4
    OmegaRmax = 7
    I_pmin = 5
    I_pmax = 10
    I_smin = 6
    I_smax = 11
    I_sOpt = 1.3097e10
    # I_sOpt = 1e5
    I_pOpt = 2.101e8
    # I_pOpt = 1e9
    OmegaROpt = 1.138e6/2/np.pi
    # solveTimeDependentPopulation(saveBool, bool1042 = True,
    #                               I_p=I_pOpt, I_s=I_sOpt, OmegaR = OmegaROpt*20*np.pi)
    # solvePopulationGreenOmegaR(saveBool = saveBool,
    #                             N_green = N,
    #                             N_OmegaR = N,
    #                             OmegaRmin = OmegaRmin,
    #                             OmegaRmax = OmegaRmax,
    #                             I_pmin = I_pmin,
    #                             I_pmax = I_pmax,
    #                             I_s = I_sOpt,
    #                             OmegaROpt = OmegaROpt,
    #                             I_pOpt = I_pOpt,
    #                             bool1042 = True)
    # solvePopulationIROmegaR(saveBool = saveBool,
    #                             N_IR = N,
    #                             N_OmegaR = N,
    #                             OmegaRmin = OmegaRmin,
    #                             OmegaRmax = OmegaRmax,
    #                             I_smin = I_smin,
    #                             I_smax = I_smax,
    #                             I_p =I_pOpt,
    #                             OmegaROpt = OmegaROpt,
    #                             I_sOpt = I_sOpt,
    #                             bool1042 = True)
    # solvePopulationIRGreen(saveBool = saveBool,
    #                             N_green = N,
    #                             N_IR = N,
    #                             I_smin = I_smin,
    #                             I_smax = I_smax,
    #                             I_pmin = I_pmin,
    #                             I_pmax = I_pmax,
    #                             OmegaR = OmegaROpt,
    #                             I_sOpt = I_sOpt,
    #                             I_pOpt = I_pOpt,
    #                             bool1042 = True)
    
    
    # solvePopulationDetuningLineplot(saveBool = False, 
    #                                 OmegaR = 2*np.pi*OmegaROpt,          
    #                                 I_p = I_pOpt,
    #                                 I_s = I_sOpt,
    #                                 bw = 0.01e9*2*np.pi,
    #                                 Npoints = 201,
    #                                 fastBool = False,
    #                                 bool1042 = True)
    
    # solvePopulationDetuningLineplotPL(saveBool = False, 
    #                                 OmegaR = 2*np.pi*OmegaROpt,          
    #                                 I_p = I_pOpt*10,
    #                                 I_s = I_sOpt*0,
    #                                 bw = 0.01e9*2*np.pi,
    #                                 Npoints = 201,
    #                                 fastBool = False,
    #                                 bool1042 = True)
    
    # C, nu, mOnArr, mOffArr, I_sArr, I_pArr, OmegaRArr = solveSensitivity(saveBool = saveBool,
    #                             N_green = N,
    #                             N_IR = N,
    #                             N_OmegaR = NOmega,
    #                             I_smin = I_smin,
    #                             I_smax = I_smax,
    #                             I_pmin = I_pmin,
    #                             I_pmax = I_pmax,
    #                             OmegaRmin = OmegaRmin,
    #                             OmegaRmax = OmegaRmax,
    #                             Npoints=301, 
    #                             bw=0.1e9*2*np.pi,
    #                             c = c,
    #                             h = h)
    
    solvePopulationLineplots(saveBool)
    # 
    # plt.figure(figsize=(4,4))
    # X, Y = np.meshgrid(np.log10(I_sArr), np.log10(I_pArr))
    # im = plt.pcolormesh(X, Y, (C), cmap = 'hot')
    # # im = ax.contourf(X, Y, p7, 100)
    # plt.tight_layout()
    # plt.xlabel(r'$I_s$ (W/m$^2$)')
    # plt.ylabel(r'$I_p$ (W/m$^2$)')
    
    # cbar = plt.colorbar(im, orientation = 'horizontal', location = 'top')
    # cbar.set_label(r'population contrast, $m_7^{\text{(on)}} - m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)
    
    # plt.show()
    
    # plt.figure(figsize=(4,4))
    # X, Y = np.meshgrid(np.log10(I_sArr), np.log10(I_pArr))
    # im = plt.pcolormesh(X, Y, (nu)/1e6, cmap = 'hot')
    # # im = ax.contourf(X, Y, p7, 100)
    # plt.tight_layout()
    # plt.xlabel(r'$I_s$ (W/m$^2$)')
    # plt.ylabel(r'$I_p$ (W/m$^2$)')
    
    # cbar = plt.colorbar(im, orientation = 'horizontal', location = 'top')
    # cbar.set_label(r'electronic spin linewidth, $\nu$ (MHz)', labelpad=2, fontsize = labelfontsize)
    
    # plt.show()
    
    performOptimization()


    
