# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:51:41 2026

@author: nicho
"""
import sys
sys.path.append('C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/customPyDrivers')

import matplotlib.colors as pltcolors
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# import seaborn as sns
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
from scipy.signal import butter, filtfilt
import re
import pandas as pd
import os
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from scipy.signal import welch

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import solve_ivp
from scipy.signal import square


from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from linbladMESolverFunctionChargeConversionLiouvillian import nv_steady_state#, nv_steady_state_fast, nv_steady_state_IRRate
from commonlabcode.plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible
from scipy.io import loadmat
from tqdm import tqdm
import optuna
import time
import multiprocessing as mp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
from NVpopulationDynamics import *
from allantools import oadev 

import glob

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIGURES_DIR = os.path.join(_THIS_DIR, "..", "__figures")
_REFFILES_DIR = os.path.join(_THIS_DIR, "..", "__refFiles")

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")

labelfontsize = 8
numfontsize = 8
textwidth = 174/25.4#8.5 - 2 # in inches
PRQuantumWidthFull = textwidth
PRQuantumWidthHalf = 85/25.4

# === Constants ===
h = 6.62607015e-34      # Planck constant [J·s]
hbar = h/(2*np.pi)
c_const = 299792458.0 #m/s

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

ms = 2
msa = 3
lw = 0.2/0.3528
nd = 2.3904
GHzprad = 1e-9/2/np.pi

lamb_s = 1042e-9   # IR probe
lamb_p = 532e-9    # green pump
nus = c_const/lamb_s
omegas = nus*2*np.pi
detuning = 0.1e9*2*np.pi
zeeman = 0.1e9*2*np.pi
etaFibreIR = 0.25
etaPostFibre = 0.3
DeltaTCorrection = 1

# diamond/NV material constants
gamma = 1.761e11/2/np.pi # Hz/T
T2s = 0.43e-6
gamma_phi = 1/T2s
nd = 2.4
hyperfine = 2.16e6

carbon_density = 1.76e29  # atoms/m³
rho_NV = carbon_density*4.5e-6 # atoms/m^3


# I_pOpt     = 1.785e+08
# I_sOpt     = 1.005e+10
# OmegaROptRadps  = 9.973e+05
# I_pOpt     = 3.135e+08
# I_sOpt     = 1.162e+10
# OmegaROptRadps  = 3.824e+06

I_pOpt     = 3.043e+08
I_sOpt     = 1.014e+10
OmegaROptRadps  = 1.984e+06


p7Off, p7On = solvePopulationDetuningLineplot(saveBool = False, 
                                    OmegaR = OmegaROptRadps,          
                                    I_p = I_pOpt,
                                    I_s = I_sOpt,
                                    bw = 0.02e9*2*np.pi,
                                    Npoints = 3,
                                    fastBool = False,
                                    plotBool = False,
                                    returnPops = True)
    
    
def p72AlphaIdeal(p7):
    alpha = rho_NV*p7*sigma_s
    return alpha

alphaBulk0 = p72AlphaIdeal(p7Off)
alphaBulk1 = p72AlphaIdeal(p7On)
alphaBulk = alphaBulk0/2 + alphaBulk1/2

alpha0 = alphaBulk0/3 #(per m)
alpha1 = alpha0 + (alphaBulk1 - alphaBulk0)/12
        
alpha = alpha0/2 + alpha1/2

# alpha0 = 4.599 #(per m)
# alpha1 = alpha0 + 0.359
        
# alpha = alpha0/2 + alpha1/2

OmegaROpt_vis  =  OmegaROptRadps# 3.018e+06
    
I_sat = hbar*omegas/sigma_s*gamma_s*2

nuOpt = 1840020
A = 0.1e-3**2

paramsRatesReal = np.array([2.97702015e+07, 1.80180000e+07, 5.65801443e+04 ,3.79558334e+06,
 3.69347363e+03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 3.68409690e-01, 1.90125394e+02, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 3.02741247e+02, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 2.12417732e+03, 1.35680211e+05,
 6.85335419e+02 ,3.96197208e+01])

paramsRatesSimple = np.array([2.97702015e+07, 1.80180000e+07, 5.65801443e+04 ,3.79558334e+06,
 3.69347363e+03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0, 0, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 2.12417732e+03, 1.35680211e+05,
 6.85335419e+02 ,0])

paramsRatesOnes = np.ones(22)

colorBlue = '#190cffff'
colorDarkBlue = '#0900a2ff'
colorLightBlue = '#3e9effff'

colorRed = '#fe0000ff'
colorDarkRed = '#800000ff'
colorLightRed = '#ff8484ff'

colorGreen = '#07b700ff'
colorDarkGreen = '#058000ff'


default_params = dict(
    gamma_phi=gamma_phi,
    detuning=detuning,
    zeeman=zeeman,
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
    paramsRates=paramsRatesOnes,
    paramsRatesReal=paramsRatesReal
)

simple_params = dict(
    gamma_phi=gamma_phi,
    detuning=detuning,
    zeeman=zeeman,
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
    paramsRates=paramsRatesOnes,
    paramsRatesReal=paramsRatesSimple
)

#%% data files
filepathTransmission = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/'
filenameTransmission4p2um = '4.2um_0/20260202/1040_1042nm_1nmps_averaged_norm.mat'

filepathODMR =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/'
filenameODMR4p2umBiasLI1 =  '20260128/ODMR_20260128_CW__16_14dBAtt_4.2um_0_1_2.7765GHz_10MHz_003.mat'
filenameODMR4p2umBiasLI2 =  '20260201/ODMR_20260201_CW__16_12dBAtt_4.2um_0_1__2.7767GHz_16MHz_001.mat'
filenameODMR4p4umBiasLI1 =  '20260205/ODMR_20260205_CW__16_12dBAtt_4.4um_0_1__2.7767GHz_10MHz_006.mat'
filenameODMR4p4umBiasLI2 =  '20260205/ODMR_20260205_CW__16_12dBAtt_4.4um_0_1__2.7767GHz_10MHz_008.mat'

#%% conversion functions
def wavelength_to_freq(lambda_nm):
    """Convert wavelength (nm) to frequency (THz)."""
    lambda_m = lambda_nm * 1e-9
    return c_const / lambda_m / 1e12

def freq_to_wavelength(freq_THz):
    """Convert frequency (THz) to wavelength (nm)."""
    return (c_const / (freq_THz * 1e12)) * 1e9

def P2ResNcav(P, etaFibre, omega, kex, kc):
    return P*np.sqrt(etaFibre)/(hbar*omega)*4*kex/(kex + kc)**2

def Ncav2I(Ncav, omega, ng, Veff):
    return c_const*Ncav*hbar*omega/(2*ng*Veff)

def NtoI1042(N, lambdacav = 1042e-9, ng = 2.41, V0 = 18.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2

def DeltaT2Deltak(DeltaT, tau, kex, kmean):
    return DeltaT/(4*kex/kmean**2 * (tau - 2*kex/kmean))

#%% fitting functions
def lorentzian(x, x0=0, gamma=1, A=1):
    """
    Lorentzian function
    """
    return A * (1/2*gamma)**2 / ((x - x0)**2 + (1/2*gamma)**2)

def derivlorentzian(x, x0=0, gamma=1, A=1):
    """
    First derivative of the Lorentzian function
    """
    return A * (1/2*gamma)**2 * 2 * (x - x0) / ((x - x0)**2 + (1/2*gamma)**2)**2


def dderivlorentzian(x, x0=0, gamma=1, A=1):
    """
    Second derivative of the Lorentzian function
    """
    num1 = 8 * A * (1/2*gamma)**2 * (x - x0)**2
    num2 = 2 * A * (1/2*gamma)**2
    denom = ((x - x0)**2 + (1/2*gamma)**2)**3
    return num1 / denom - num2 / ((x - x0)**2 + (1/2*gamma)**2)**2

def microwaveDetuning3(f, gamma, f0, amp, offset):
    return 1/3*amp * (
        3*lorentzian(f, f0, gamma) +
        2*lorentzian(f, f0 -  hyperfine, gamma) +
        2*lorentzian(f, f0 +  hyperfine, gamma) +
        lorentzian(f, f0 -  2*hyperfine, gamma) +
        lorentzian(f, f0 +  2*hyperfine, gamma) + offset
    )
def microwaveDetuning1(f, gamma, f0, amp, offset):
    return amp * (
        lorentzian(f, f0, gamma) +
        lorentzian(f, f0 -  hyperfine, gamma) +
        lorentzian(f, f0 +  hyperfine, gamma) + offset
    )
def microwaveDetuningDeriv3(f, gamma, f0, amp, offset):
    return 1/3*amp * (
        3*derivlorentzian(f, f0, gamma) +
        2*derivlorentzian(f, f0 -  hyperfine, gamma) +
        2*derivlorentzian(f, f0 +  hyperfine, gamma) +
        derivlorentzian(f, f0 -  2*hyperfine, gamma) +
        derivlorentzian(f, f0 +  2*hyperfine, gamma) + offset
    )
def microwaveDetuningDeriv1(f, gamma, f0, amp, offset):
    return amp * (
        derivlorentzian(f, f0, gamma) +
        derivlorentzian(f, f0 -  hyperfine, gamma) +
        derivlorentzian(f, f0 +  hyperfine, gamma)
    )

def microwaveDetuningDDeriv3(f, gamma, f0, amp, offset):
    return 1/3*amp * (
        3*dderivlorentzian(f, f0, gamma) +
        2*dderivlorentzian(f, f0 -  hyperfine, gamma) +
        2*dderivlorentzian(f, f0 +  hyperfine, gamma) +
        dderivlorentzian(f, f0 -  2*hyperfine, gamma) +
        dderivlorentzian(f, f0 +  2*hyperfine, gamma)
    )
def microwaveDetuningDDeriv1(f, gamma, f0, amp, offset):
    return amp * (
        dderivlorentzian(f, f0, gamma) +
        dderivlorentzian(f, f0 -  hyperfine, gamma) +
        dderivlorentzian(f, f0 +  hyperfine, gamma)
    )

def lorentzianSingleWithAmplitude(f, gamma, f0, amp):
    return amp*lorentzian(f, f0, gamma)

def calcIntracavityField(kex1, k1, kex2, k2, Delta, g):
    
    kex1 = np.complex128(kex1)
    k1   = np.complex128(k1)
    kex2 = np.complex128(kex2)
    k2   = np.complex128(k2)
    Delta = np.complex128(Delta)
    g     = np.complex128(g)
    
    return  np.sqrt(kex1/2)/(-1j*(Delta+g/2) + k1/2) + np.sqrt(kex2/2)/(-1j*(Delta-g/2) + k2/2)

# def calcTransmission(tau, kex, k, Delta):
#     return np.abs(tau - 1/2*(kex1)/(-1j*(Delta+g/2) + k1/2) - 1
#     return tau**2 + (kex**2 - tau*kex*k)/(Delta**2 + k**2/4)
# def calcTransmission(tau, kex, k, Delta):
#     return np.abs(tau - (kex)/(-1j*(Delta+g/2) + k/2))**2
def etalon(contrast, freq, delta, offset):
    return (1 + contrast*np.cos(delta/freq + offset))

def calcTransmission(tau, kex1, k1, kex2, k2, Delta, g):
    return np.abs(tau - 1/2*(kex1)/(-1j*(Delta+g/2) + k1/2) - 1/2*(kex2)/(-1j*(Delta-g/2) + k2/2))**2 #* etalon(0.15, 15e9, Delta, -1e9) * etalon(0.4, 40e9, Delta, 0e9)

def calcPhotonsInCavity(a, Pin):
    return np.abs(a)**2*Pin/(h*c_const/lamb_s)

def calcVeff():
    return 19*(lamb_s/nd)**3

def calcVeffHighQ():
    return 19*(lamb_s/nd)**3
    
def calcIntracavityIntensity(ncav, ng, Veff):
    return 1/2 * ncav * c_const * h * c_const/lamb_s /ng / Veff


def calcEnergyLossRate(ng, alpha):
    return c_const/ng*alpha


def solve_self_consistent(Delta, P_IR_in, tau, ki1, kex1, ki2, kex2, omegas, g = 0, defectDetuning = 0,
                          Veff=None, ng=2.41, max_iter=80, tol=1e-6,damping = 0.8, alphascale = 1,
                          verbose=False, angleNV = 0 ):
    """
    Solve self-consistently for each Ip (rows) and Delta (cols):
      - compute cavity linewidth k = kex + kip + kNV
      - compute intracavity field 'a', transmission T, intracavity photon number & intensity I_IR
      - compute NV populations from I_p, I_s, OmegaR
      - compute alpha_NV and kNV_new = c/ng * alpha_NV
    Iterate until kNV converges (relative change < tol) or max_iter reached.

    Returns dict with:
      'T'    : shape (Ni, Nd) transmission
      'I_IR' : shape (Ni, Nd) intracavity intensity (W/m^2)
      'kNV'  : shape (Ni, Nd) NV-induced energy loss rate (1/s)
      'ZFull': shape (Ni, Nd, 6) populations (converged)
    """
    Nd = np.size(Delta)
    if Veff is None or Veff == 0:
        Veff = calcVeff()

    # derived rates
    omegas_local = omegas

    # output arrays
    T_mat = np.zeros((Nd), dtype=float)
    a_mat = np.zeros((Nd), dtype=float)
    a_imag_mat = np.zeros((Nd), dtype=float)
    a_real_mat = np.zeros(( Nd), dtype=float)
    ncav_mat = np.zeros((Nd), dtype=float)
    IIR_mat = np.zeros((Nd), dtype=float)
    kabs_mat = np.zeros((Nd), dtype=float)
    alpha_mat = np.zeros(( Nd), dtype=float)
    alphaOn_mat = np.zeros(( Nd), dtype=float)
    alphaOff_mat = np.zeros(( Nd), dtype=float)
    # n_mat = np.full((Ni, Nd, 6), np.nan, dtype=float)

    def lorentzian(x, gamma):
        return (gamma**2 / ((x)**2 + gamma**2))

    # Loop over pump intensities and detunings
    
    for itDisp in range(1):
        
        for j, d in enumerate(Delta):
            # initial guess
            kabs = 0.0
            k1 = kex1 + ki1 + kabs
            k2 = kex2 + ki2 + kabs

            # small safety to avoid division by zero
            converged = False
            prev_kNV = kabs

            for it in range(max_iter):
                # cavity quantities for current k
                # calcIntracavityField returns complex field amplitude a (uses sqrt(kex)/(k/2 - i*Delta))
                a = calcIntracavityField(kex1, k1, kex2, k2, d, g= g)
                T = calcTransmission(kex1, k1, kex2, k2, d, g = g, tau = tau)
                ncav = calcPhotonsInCavity(a, P_IR_in)
                # ncav = np.abs(a)**2
                I_IR_loc = calcIntracavityIntensity(ncav, ng, Veff)  # W/m^2


                # populations for current intensities
                # pops = solvePopulation_fast(Ip, I_IR_loc, OmegaR)
                p7Off = ItoPOff(I_IR_loc)
                p7On = ItoPOn(I_IR_loc)

                if np.any(np.isnan(p7On)) or np.any(np.isnan(p7Off)):
                    # failed to solve -> mark and break
                    kabs = np.nan
                    break

                # compute NV absorption coefficient (per meter) 
                alphaOff = p72AlphaIdeal(p7Off)/3#* np.cos(np.pi/180 * angleNV)**2
                alphaOn = p72AlphaIdeal(p7Off/3 + (p7On-p7Off)/12)#* np.cos(np.pi/180 * angleNV)**2
                alpha_NV = (alphaOff + alphaOn)/2

                kNV_new = calcEnergyLossRate(ng, alpha_NV)

                # damping for stability
                kabs = (1 - damping) * kabs + damping * kNV_new

                # update cavity k
                k1 = kex1 + ki1 + kabs
                k2 = kex2 + ki2 + kabs

                # check convergence of kNV (relative)
                if (np.isfinite(prev_kNV) and prev_kNV > 0):
                    rel_change = np.abs(kabs - prev_kNV) / prev_kNV
                else:
                    rel_change = np.abs(kabs - prev_kNV)

                prev_kNV = kabs

                if rel_change < tol:
                    converged = True
                    break

            # store converged values (or latest if not converged)
            T_mat[j] = T
            a_real_mat[j] = np.real(a)
            a_mat[j] = np.abs(a)
            a_imag_mat[j] = np.imag(a)
            ncav_mat[j] = ncav
            IIR_mat[j] = I_IR_loc
            kabs_mat[j] = kabs
            alpha_mat[j] = alpha_NV
            alphaOff_mat[j] = alphaOff
            alphaOn_mat[j] = alphaOn

            if verbose and (j % max(1, Nd//10) == 0):
                print(f"  Δ index {j+1}/{Nd}: kNV={kabs:.3e}, I_IR={I_IR_loc:.3e}, conv={converged}, it={it+1}")


    return dict(T=T_mat, I_IR=IIR_mat, kNV=kabs_mat, alpha_NV = alpha_mat, alphaOn = alphaOn_mat, alphaOff = alphaOff_mat)


def calcDeltaMicrodisk(Delta, kex1, kex2, ki1, ki2, g, tau, phi, P_in, scale):
    
    #convert delta to alpha
    solution = solve_self_consistent(Delta, P_in, tau, ki1, kex1, ki2, kex2, omegas, g)
    
    alphaOn = solution['alphaOn']
    alphaOff = solution['alphaOff']
    alphaMean = solution['alpha_NV']
    
    kabs0 = calcEnergyLossRate(nd, alphaOff)
    kabs1 = calcEnergyLossRate(nd, alphaOn)
    
    # If you still want ki defined relative to the *mean absorption*:
    kabs_mean = 0.5 * (kabs0 + kabs1)
    ki = ki1 - kabs_mean
    
    # total loss rates
    k0 = ki + kabs0
    k1 = ki + kabs1
    
    T0 = calcTransmission(
        kex1, kex2, k0, k0, Delta,
        g=g,
        tau=tau * np.exp(1j * phi)
    )
    T1 = calcTransmission(
        kex1, kex2, k1, k1, Delta,
        g=g,
        tau=tau * np.exp(1j * phi)
    )
    
    DeltaT = T1 - T0
    return np.real(DeltaT) * scale

def calcDeltaMicrodiskReturnT(Delta, kex1, kex2, ki1, ki2, g, tau, phi, P_in, scale, printBool = False, Veff = 0, doubletBool = False):

    
    #convert delta to alpha
    solution = solve_self_consistent(Delta, P_in, tau, ki1, kex1, ki2, kex2, omegas, g, Veff = Veff)
    
    if doubletBool:
        alphaOn = solution['alphaOn']
        alphaOff = solution['alphaOff']
        alphaMean = solution['alpha_NV']
        
    else:
        alphaOn = solution['alphaOn']
        alphaOff = solution['alphaOff']
        alphaMean = solution['alpha_NV']

    
    
    kabs0 = calcEnergyLossRate(nd, alphaOff)
    kabs1 = calcEnergyLossRate(nd, alphaOn)
    DeltaK = np.max(kabs1-kabs0)
    
    # If you still want ki defined relative to the *mean absorption*:
    kabs_mean = 0.5 * (kabs0 + kabs1)
    ki1 = ki1 - kabs_mean
    ki2 = ki2 - kabs_mean
    
    # total loss rates
    k01 = ki1 + kabs0
    k11 = ki1 + kabs1
    k02 = ki2 + kabs0
    k12 = ki2 + kabs1
    
    T0 = calcTransmission(
        kex1, kex2, k01, k02, Delta,
        g=g,
        tau=tau * np.exp(1j * phi)
    )
    T1 = calcTransmission(
        kex1, kex2, k11, k12, Delta,
        g=g,
        tau=tau * np.exp(1j * phi)
    )
    
    DeltaT = T1 - T0
    
    if printBool:
        return np.real(DeltaT) * scale, (T1 + T0)/2, kabs0 ,kabs1, alphaOff, alphaOn
    else:
        return np.real(DeltaT) * scale, (T1 + T0)/2

def cleanUpAverageData(filePath, filenames, bgOffset):
    W_list = []
    T_list = []
    
    for filename in filenames:
        data = loadmat(filePath + filename, squeeze_me=True)
        data = data['scanData']
        W = data['W'].item()
        TOut = data['T'].item()
        # TOut /= TOut.max()
        W_list.append(W[:].T)
        T_list.append(TOut[:].T - bgOffset)
        plt.plot(W, TOut)
    
    # 1. Define common x grid (overlapping region is safest)
    W_min = max([np.min(x) for x in W_list])
    W_max = min([np.max(x) for x in W_list])
    
    # Choose resolution (you can tune this)
    fs = 1000
    W_common = np.linspace(W_min, W_max, fs)
    
    # 2. Interpolate each dataset
    T_interp_list = []
    
    for x, y in zip(W_list, T_list):
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
        T_interp = f(W_common)
        T_interp_list.append(T_interp)
    
    # Convert to array
    T_interp_stack = np.array(T_interp_list)
    
    # 3. Average (ignore NaNs if any)
    T_mean = np.nanmean(T_interp_stack, axis=0)
        
    T_mean /=T_mean.max()
    
    def lowpass_fft(signal, fs, f_cut):
        """
        signal : 1D array
        fs     : sampling frequency (Hz)
        f_cut  : cutoff frequency (Hz)
        """
        N = len(signal)
        
        # FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, d=1/fs)
        
        # Create mask
        mask = np.abs(freqs) <= f_cut
        
        
        # Apply filter
        fft_vals_filtered = fft_vals * mask
        
        # Inverse FFT
        filtered_signal = np.fft.ifft(fft_vals_filtered)
        
        return np.real(filtered_signal)
    
    from scipy.signal import iirnotch, filtfilt

    def notch_filter(signal, fs, f0, Q=30):
        """
        f0 : center frequency (Hz)
        Q  : quality factor (higher = narrower notch)
        """
        b, a = iirnotch(f0, Q, fs)
        return filtfilt(b, a, signal)
    
    T_mean /= T_mean.max()
    
    DeltanuGHz = c_const/W_common
    
    return W_common,DeltanuGHz,  T_mean

def loadAndFilterTransmissionData(filePath ):
    
    data = loadmat(filePath, squeeze_me=True)
    # dataParams = loadmat(filePathParams, squeeze_me=True)
    
    try:
        W = data['W']
        TOut = data['TOut']
    except:
        W = data['W'].item()
        TOut = data['TOut'].item()
        
        
    # params = dataParams['params']
    lambmin = 1040.2
    lambmax = 1041.4
    filt = np.logical_and(W< lambmax, W>lambmin)
    W= W[filt]
    nuArr = c_const/W/1e-9
    nu0 = c_const/1040.77e-9
    DeltanuGHz = (nuArr - nu0)*1e-9
    TOut= TOut[filt]
    return W, DeltanuGHz, TOut

def loadODMRData(filePath, lamb0nm = 1042, lambbwnm = 2, f0GHz = 2870, 
                 nuOptbwGHz = 200, fbwGHz = 10, powerScaling = 99/9, nubwGHz = 200, 
                 boolFreqFiltering = True, idx = 2, bgOffset = 0):
    # load data
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    
    if boolFreqFiltering:
        
        nu0GHz = c_const/lamb0nm
        numinGHz = nu0GHz - nubwGHz
        numaxGHz = nu0GHz + nubwGHz
        nuGHz = c_const/W_nm
        filt = np.logical_and(nuGHz< numaxGHz, nuGHz>numinGHz)
        
    else:
        lambminnm = lamb0nm - lambbwnm
        lambmaxnm = lamb0nm + lambbwnm
        nu0GHz = c_const/lamb0nm
        numinGHz = c_const/lambmaxnm
        numaxGHz = c_const/lambminnm
        nuGHz = c_const/W_nm
        filt = np.logical_and(W_nm< lambmaxnm, W_nm>lambminnm)
    

    W_nm= W_nm[filt]
    nuGHz = nuGHz[filt]
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        
        if boolFreqFiltering:
            
            nu0GHz = c_const/lamb0nm
            numinGHz = nu0GHz - nubwGHz
            numaxGHz = nu0GHz + nubwGHz
            nuGHz = c_const/W_nm
            filt = np.logical_and(nuGHz< numaxGHz, nuGHz>numinGHz)
            
        else:
            lambminnm = lamb0nm - lambbwnm
            lambmaxnm = lamb0nm + lambbwnm
            nu0GHz = c_const/lamb0nm
            numinGHz = c_const/lambmaxnm
            numaxGHz = c_const/lambminnm
            nuGHz = c_const/W_nm
            filt = np.logical_and(W_nm< lambmaxnm, W_nm>lambminnm)
            
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    P_in = data['powerIn'].item()*powerScaling
    # print('Power into the fibre: ',np.mean(power))
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-f0GHz
    
    
    filt = f*0 < fbwGHz
    
    f= f[filt]
    S_V= S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = np.abs(S_V/C) - bgOffset
    DeltaT = S_V/np.max(DC_V)
    
    DeltaW_nm = W_nm - lamb0nm
    DeltanuGHz = nuGHz - nu0GHz
    
    return W_nm, S_V, DeltaT, C, P_in, DeltanuGHz, DeltaW_nm, f, DC_V

#%% general functions

def aMicrodiskOnRes(ka, ki, kex):
    return np.sqrt(kex)*2/(ka + kex + ki)

def aMicrodiskRes2PinFibre(V, I, a):
    return 2*I*nd*V/c_const/a**2

def afromalpha(alpha, L):
    return np.exp(-alpha*L)

def calcQMicrodisk(omega, kappa):
    return omega/kappa

def positive_real_values(arr, tol=1e-12):
    arr = np.asarray(arr)
    mask = (np.abs(arr.imag) < tol) & (arr.real > 0)
    return arr.real[mask]

def TEtalon(Tm, L, alpha):
    return Tm**2 * np.exp(-alpha*L)/(1-(1-Tm)*np.exp(-alpha*L))**2

def TBulk(L, alpha):
    return np.exp(-alpha*L)


def calcNonlinearTransmissionBulk(L, MW = 'On'):
    def ode(l, I):
        alphaLocalOff = p72AlphaIdeal(ItoPOff(I))/3
        alphaLocalOn = alphaLocalOff + (p72AlphaIdeal(ItoPOn(I)) - p72AlphaIdeal(ItoPOff(I)))/12
        if MW == 'On':
            return -alphaLocalOn*I
        else:
            return -alphaLocalOff*I
            
    
    sol = solve_ivp(ode, (0,L), [I_sOpt], denseOutput = False)
    
    Iout = sol.y[0,-1]
    
    T = Iout/I_sOpt
    return T

def calcBulkSensitivityLin(L, A = 1):    
    C = (L*(alpha1-alpha0))
    P = I_sOpt*A
    return calcSensitivityCW(nuOpt, C, P, lamb_s)

def calcBulkSensitivitySimple(L, A = 1):    
    T0 = TBulk(L, alpha0)  
    T1 = TBulk(L, alpha1) 
    T = T0/2 + T1/2
    C = (T0-T1)/T
    P = I_sOpt*A
    return calcSensitivityCW(nuOpt, C, P*T, lamb_s)

def calcBulkSensitivity(L, A = 1):    
    T0 = calcNonlinearTransmissionBulk(L, MW = 'Off')
    T1 = calcNonlinearTransmissionBulk(L, MW = 'On')
    T = T0/2 + T1/2
    C = (T0-T1)/T
    P = I_sOpt*A
    return calcSensitivityCW(nuOpt, C, P*T, lamb_s)

def PinEtalon(Tm, L, alpha, I, A):
    # return A*I *(1-(1-Tm)*np.exp(-alpha*L))**2/(Tm*np.exp(-alpha*L))
    return A*I /Tm *(1- (1-Tm)*(np.exp(-alpha*L)))**2

def calcBestTm(a0, a1):
    return ((-a0 + a0**2 - (a1-1)*a1 + a0**3*(a1-1)*a1 + a0*a1**3 - a0**2*a1**3)/
            (-3*a0**2*a1 - 3*a0**3*(a1-1)*a1 + 3*a0*a1**2 - 3*a0*a1**3 + 3*a0**2*a1**3))

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
    return 4/(3*np.sqrt(3)) * 1/gamma * nu/C * np.sqrt(h*c_const/lamb/P)

def calcSensitivityCWUncertainty(nu, dnu, C, dC, P, dP, lamb):
    return 4/(3*np.sqrt(3)) * 1/gamma * np.sqrt(
        (dnu/C * np.sqrt(h*c_const/lamb/P))**2 + 
        (nu*dC/C**2 * np.sqrt(h*c_const/lamb/P))**2 +
        (nu*dP/C/2 * np.sqrt(h*c_const/lamb)*(P**(-3/2)))**2        
        )

def calcSensitivityCW_countrate(nu, C,R):
    return 4/(3*np.sqrt(3)) * 1/gamma * nu/C * np.sqrt(1/R)


def calcSensitivityRamsey_countrate(C, R, tread, tdephase, twrite):
    ttotal = tread + tdephase + twrite
    return 1/(2*np.pi*gamma)/C * np.sqrt(1/R/tread) * np.sqrt(ttotal)/tdephase

def calcQ(L, Tm, a):
    return 4*np.pi*nd*L/lamb_s/(1- (1-Tm)**2*a)

def Thermal_fit(Delta, Pi, CT, ke, k, n):

    # % Calculate intracavity photon number 
    A = -288*CT**3*k**2*Delta + 1728*CT**4*ke*Pi - 128*CT**3*Delta**3;
    B = 12*CT**2*k**2 - 16*CT**2*Delta**2;
    D = (A + np.sqrt(4*B**3 + A**2))**(1/3);  

    N1 = 2*Delta/(3*CT) - B/(6*2**(2/3)*CT**2*D) + D/(12*2**(1/3)*CT**2);
    N2 = 2*Delta/(3*CT) - (1 + 1j*np.sqrt(3))*B/(6*2**(2/3)*CT**2*D) + (1 - 1j*np.sqrt(3))*D/(12*2**(1/3)*CT**2);
    N3 = 2*Delta/(3*CT) - (1 - 1j*np.sqrt(3))*B/(6*2**(2/3)*CT**2*D) + (1 + 1j*np.sqrt(3))*D/(12*2**(1/3)*CT**2);

    N1 = np.real(N1);
    N2 = np.real(N2);
    N3 = np.real(N3);

    if n == 1:
        a = np.sqrt(ke)/(-1j*(Delta - CT* N1) + k/2); 
        N = N1;
    elif n == 2:
        a = np.sqrt(ke)/(-1j*(Delta - CT* N2) + k/2); 
        N = N2;
    elif n == 3:
        a = np.sqrt(ke)/(-1j*(Delta - CT* N3) + k/2);

        N = N3;

    return a, N

def calcTransmission(kex1, kex2, ki1, ki2, Delta, g = 0, tau = 1):
    a1 = np.sqrt(kex1)/((ki1 + kex1)/2 - 1j*(Delta - g/2))
    a2 = np.sqrt(kex2)/((ki2 + kex2)/2 - 1j*(Delta + g/2))
    return np.real(np.abs(tau - np.sqrt(1/4*kex1)*a1 - np.sqrt(1/4*kex2)*a2)**2)

def calcTransmissionSinglet(kex1, ki1, Delta, tau = 1):
    a1 = np.sqrt(kex1)/((ki1 + kex1)/2 - 1j*(Delta))
    return np.abs(tau - np.sqrt(kex1)*a1)**2

def solveAlphaFuncofIRIntensity(N = 11):
    I_smin = 2
    I_smax = 12
        
    C, power, nu, I_IR, p7On, p7Off =     solvePopulationIR(saveBool = saveBool,
                                    N_IR = N,
                                    OmegaR = OmegaROptRadps,
                                    I_smin = I_smin,
                                    I_smax = I_smax,
                                    I_p =I_pOpt,
                                    plotBool = False,
                                    logBoolI = True)
    
    
    plt.figure()
    plt.semilogx(I_IR,p7Off)
    plt.semilogx(I_IR,p7On)
    plt.semilogx(I_IR,p7On-p7Off)
    plt.show()
    
    np.savez(os.path.join(_REFFILES_DIR, "p7_I_IR_lookup.npz"),
         I_IR=I_IR,
         p7On=p7On,
         p7Off=p7Off)

    return I_IR, p7On, p7Off

def get_p7_interp(filename=None, onBool = True):
    if filename is None:
        filename = os.path.join(_REFFILES_DIR, "p7_I_IR_lookup.npz")
    
    if os.path.exists(filename):
        data = np.load(filename)
        I_IR = data["I_IR"]
        p7On = data["p7On"]
        p7Off = data["p7Off"]
    else:
        # --- expensive computation ---
        I_IR, p7On = solveAlphaFuncofIRIntensity()
        idx = np.argsort(p7On)
        I_IR = I_IR[idx]
        p7On = p7On[idx]
        p7Off = p7Off[idx]
        np.savez(filename, I_IR=I_IR, p7On=p7On , p7Off=p7Off )

    if onBool:
        return interp1d(I_IR, p7On,
                        kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate')
    else:
        return interp1d(I_IR, p7Off,
                        kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate')
    
ItoPOn = get_p7_interp(onBool = True)
ItoPOff = get_p7_interp(onBool = False)


def kappa2alpha(kappa):
    return kappa/2/np.pi*nd/c_const
def alpha2kappa(alpha):
    return alpha*2*np.pi*c_const/nd

#%% other plotting functions

def plotVisibleSpectrometerData(saveBool = False):
    

    fig, ax = plot_custom(
       PRQuantumWidthFull*0.7, 0.5/1.618*PRQuantumWidthFull, r"NV emission wavelength (nm)", r"NV PL intensity (au)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/Spectrometer_Data/'
    filenames = ['PL_20240702_fine_grating_550nm_900nm_greenLaser_noMW_noTelecom.csv']
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/DNV-QG3-B14/Spectrometer_Data/'
    filenames = ['redo_PL_20240702_fine_grating_550nm_900nm_greenLaser_MW_noTelecom.csv',
                 'redo_PL_20240702_fine_grating_550nm_900nm_greenLaser_noMW_noTelecom.csv']
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/DNV-QG3-B14/Spectrometer_Data/greenLaserMicrowaveTHG/'
    filenames = ['PL_20240702_fine_grating_550nm_900nm_greenLaser_noMW_noTelecom_noT.csv',
                 'PL_20240702_fine_grating_550nm_900nm_greenLaser_MW_noTelecom_noTHG.csv']

    min_x_value = 697       
    max_x_value = 699
    min_y_value = 0
    max_y_value = 1

    offset = [0, 0, 1500]
    wlOff = [0, 0.01]
    scale = [1800, 1800, 40]
    color = ['b', 'r', [0,0,0]]
    NArr = [8, 8, 1]
    areaFiberPL = np.zeros([len(filenames)])
    areaNVPL = np.zeros([len(filenames)])
    i=0
    for filename in filenames:
        # Replace 'file_path.csv' with the actual file path
        df = pd.read_csv(directory + filename)
        filt = np.array(df.Wavelength >= min_x_value) & np.array(df.Wavelength <= max_x_value)
        wl = df.Wavelength[filt]
        N = NArr[i]
        wl = wl[N:-1];
        intensity = (df.Intensity[filt].rolling(window=N).mean() - offset[i])
        intensity = intensity[N:-1]
        
        # if i ==0:
        maxIntensity = max(intensity)
        intensity = intensity/scale[i]
        
        filtFiberPL = np.array(wl >= 600) & np.array(wl <= 800) & np.array(intensity <= 0.02)
        filtNVPL = np.array(wl >= 600) & np.array(wl <= 800) & np.array(intensity >= 0.02)
        areaFiberPL[i] = np.trapz(intensity[filtFiberPL], wl[filtFiberPL])
        areaNVPL[i] = np.trapz(intensity[filtNVPL], wl[filtNVPL])
        ax.plot(wl-wlOff[i], intensity, linestyle = '-',color = color[i], lw = lw/2, zorder = i)
        i +=1

    ax.fill_between(wl, intensity, intensity*0, color = color[1], alpha = 0.1, lw = 0)

    area = areaNVPL;# - areaFiberPL;
    area = area/max(area)
    print('Collection efficiency of NVPL: ',area[0])
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.07
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')

    ax.spines['left'].set_bounds(min_y_value,max_y_value)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels


    plt.show()
    
def plotVisibleSpectrometerDiffData(saveBool = False):
    

    fig, ax = plot_custom(
       PRQuantumWidthFull*0.7, 0.5/1.618*PRQuantumWidthFull, r"NV emission wavelength (nm)", r"NV PL intensity (au)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/Spectrometer_Data/'
    filenames = ['PL_20240702_fine_grating_550nm_900nm_greenLaser_noMW_noTelecom.csv']
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/DNV-QG3-B14/Spectrometer_Data/'
    filenames = ['redo_PL_20240702_fine_grating_550nm_900nm_greenLaser_MW_noTelecom.csv',
                 'redo_PL_20240702_fine_grating_550nm_900nm_greenLaser_noMW_noTelecom.csv']
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/DNV-QG3-B14/Spectrometer_Data/greenLaserMicrowaveTHG/'
    filenames = ['PL_20240702_fine_grating_550nm_900nm_greenLaser_noMW_noTelecom_noT.csv',
                 'PL_20240702_fine_grating_550nm_900nm_greenLaser_MW_noTelecom_noTHG.csv']

    min_x_value = 600       
    max_x_value = 900
    min_y_value = -0.1
    max_y_value = 0.5

    scale = [1800, 1800]
    color = ['b', 'r']
             
    def readDatFile(directory, filename, scale):
        df = pd.read_csv(directory + filename)
        filt = np.array(df.Wavelength >= min_x_value) & np.array(df.Wavelength <= max_x_value)
        wl = df.Wavelength[filt]
        intensity = (df.Intensity[filt])
        intensity = intensity/scale
        return wl, intensity
    
    wlOff, intOff = readDatFile(directory, filenames[1], scale[1])
    wlOn, intOn = readDatFile(directory, filenames[0], scale[0])
    
    ax.plot(wlOff[100: -100], intOn[100: -100] - intOff[105: -100], linestyle = '-',color = 'k', lw = 0.2/0.3528, zorder = 1)

    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.07
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')

    ax.spines['left'].set_bounds(min_y_value,max_y_value)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels


    plt.show()

def plotIRSpectrometerData(saveBool = False):
    
    def readDatFile(directory, filename,scalar=1):
        dfBack = pd.read_csv(directory + filename)
        dfBack[['Wavelength', 'S1']] = dfBack.iloc[:, 0].str.split('\t', expand=True)
        dfBack['Wavelength'] = pd.to_numeric(dfBack['Wavelength'], errors='coerce')
        dfBack['S1'] = pd.to_numeric(dfBack['S1'], errors='coerce')
        
        filt = np.array((dfBack.Wavelength) >= min_x_value) & np.array(dfBack.Wavelength <= max_x_value)
        wl = dfBack.Wavelength[filt]
        wl = wl.values
        intensity = (dfBack.S1[filt])
        intensity = intensity.values/intensity.values[0]*scalar
        return wl, intensity

    fig, ax = plot_custom(
       PRQuantumWidthFull*0.7, 0.5/1.618*PRQuantumWidthFull, r"NV emission wavelength (nm)", r"NV PL intensity (au)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/B14_4p2_IR_Emission_Spectrum/'
    filenameBack = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra_Bypass-FiberPL.dat'
    filenameOn = 'MW_ON_1000-1400_100s_Exposure_150umSlit_Spectra.dat'
    filenameOff = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra.dat'

    min_x_value = 1000       
    max_x_value = 1450
    min_y_value = 0
    max_y_value = 4

    offset = [0, 700, 1500]
    scale = [1.05, 1, 40]
    color = [[1,0,0], [0.5,0,0], [0,0,0]]
    
    
    wlBack, intensityBack = readDatFile(directory, filenameBack, 0.8)
    wlOn, intensityOn= readDatFile(directory, filenameOn)
    wlOff, intensityOff = readDatFile(directory, filenameOff)
    
    intensityOn = intensityOn - intensityBack
    intensityOff = intensityOff - intensityBack
    
    ax.plot(wlOn, intensityOn, linestyle = '-',color = 'r', lw = 0.2/0.3528, zorder = 0)
    ax.plot(wlOff, intensityOff, linestyle = '-',color = 'b', lw = 0.2/0.3528, zorder = 0)



    # area = areaNVPL;# - areaFiberPL;
    # area = area/max(area)
    # print('Collection efficiency of NVPL: ',area[0])
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.07
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')

    ax.spines['left'].set_bounds(min_y_value,max_y_value)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels


    plt.show()
    
def plotIRSpectrometerDiffData(saveBool = False):
    
    def readDatFile(directory, filename,scalar=1):
        dfBack = pd.read_csv(directory + filename)
        dfBack[['Wavelength', 'S1']] = dfBack.iloc[:, 0].str.split('\t', expand=True)
        dfBack['Wavelength'] = pd.to_numeric(dfBack['Wavelength'], errors='coerce')
        dfBack['S1'] = pd.to_numeric(dfBack['S1'], errors='coerce')
        
        filt = np.array((dfBack.Wavelength) >= min_x_value) & np.array(dfBack.Wavelength <= max_x_value)
        wl = dfBack.Wavelength[filt]
        wl = wl.values
        intensity = (dfBack.S1[filt])
        intensity = intensity.values/intensity.values[0]*scalar
        return wl, intensity

    fig, ax = plot_custom(
       PRQuantumWidthFull*0.7, 0.5/1.618*PRQuantumWidthFull, r"NV emission wavelength (nm)", r"NV PL intensity (au)" #"coincidence rate (mHz/mW)"
       , labelPadX = 3, labelPadY = 10, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/B14_4p2_IR_Emission_Spectrum/'
    filenameBack = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra_Bypass-FiberPL.dat'
    filenameOn = 'MW_ON_1000-1400_100s_Exposure_150umSlit_Spectra.dat'
    filenameOff = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra.dat'

    min_x_value = 1000       
    max_x_value = 1450
    min_y_value = -1
    max_y_value = 2

    offset = [0, 700, 1500]
    scale = [1.05, 1, 40]
    color = [[1,0,0], [0.5,0,0], [0,0,0]]
    
    
    wlBack, intensityBack = readDatFile(directory, filenameBack, 0.8)
    wlOn, intensityOn= readDatFile(directory, filenameOn)
    wlOff, intensityOff = readDatFile(directory, filenameOff)
    
    intensityOn = intensityOn - intensityBack
    intensityOff = intensityOff - intensityBack
    
    ax.plot(wlOn, intensityOn-intensityOff, linestyle = '-',color = 'k', lw = 0.2/0.3528, zorder = 0)
    # ax.plot(wlOff, intensityOff, linestyle = '-',color = 'b', lw = 0.2/0.3528, zorder = 0)



    # area = areaNVPL;# - areaFiberPL;
    # area = area/max(area)
    # print('Collection efficiency of NVPL: ',area[0])
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.07
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')

    ax.spines['left'].set_bounds(min_y_value,max_y_value)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels


    plt.show()



def plotEtalonSensitivity(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.45, 0.34*PRQuantumWidthFull, 
       r"cavity length, $L$ (m)", 
       r'mirror transmissivity, $T_{\text{m}}$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    
    
    lw = 1
    
    N = 400
    
    # mirror transmissivity array
    Tmin = -5
    Tmax = 0
    Tm = np.logspace(Tmin, Tmax, N)
    
    # cavity length array
    Lmin = -6
    Lmax = 0
    L = np.logspace(Lmin,Lmax,N)
    
    
    Tmmat, Lmat = np.meshgrid(Tm, L)
    
    a0 = afromalpha(alpha0, L)
    a1 = afromalpha(alpha1, L)
    TmBest = -calcBestTm(a0, a1)
    filt = np.logical_and(TmBest < 1, TmBest > 0, np.isfinite(TmBest))
    filt = np.logical_and(L > 1e-4, L < 0.1)
    
    T0 = TEtalon(Tmmat, Lmat, alpha0)  
    T1 = TEtalon(Tmmat, Lmat, alpha1) 
    T = T0/2 + T1/2
    C = T0-T1
    P = PinEtalon(Tmmat, Lmat, alpha, I_sOpt, A)
    R = T*P/(hbar*omegas)
    
    Q =  calcQ(Lmat, Tmmat, a0/2 + a1/2)
    
    eta = calcSensitivityCW_countrate(nuOpt, C, R)
    etaMinarg = np.argmin(eta)
    i, j = np.unravel_index(etaMinarg, eta.shape)
    
    
    print('min eta: ', eta[i,j])
    print('length: ', Lmat[i,j])
    print('contrast: ', C[i,j])
    print('Tm: ', Tmmat[i,j])
    print('Q: ', Q[i,j])
    print('power: ', P[i,j]*T[i,j])
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 1, 256))
    )
    p=ax.contourf(np.log10(Lmat), np.log10(Tmmat), np.log10(eta), 25, cmap = 'viridis',vmin=np.min(np.log10(eta)), vmax=np.max(np.log10(eta)))
    
    # ax.scatter(np.log10(Lmat[i,j]), np.log10(Tmmat[i,j]), marker = '*', color = 'red', clip_on=False, zorder = 10)
    ax.scatter(np.log10((np.log(alpha0) - np.log(alpha1))/(alpha0-alpha1)), np.log10(1), marker = '*', color = 'red', clip_on=False, zorder = 10)

    ticks = [np.max(np.log10(eta)), np.min(np.log10(eta))]
    cbar = fig.colorbar(
        ScalarMappable(norm=p.norm, cmap=p.cmap),
        ax = ax
    )
    # cbar = fig.colorbar(p, ax=ax, orientation='vertical', label = r'',ticks=ticks)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks], fontsize = numfontsize)
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks], fontsize = numfontsize)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$10^{{{val:.1f}}}$" for val in ticks], fontsize = numfontsize)
    cbar.set_label(label = r'sensitivity, $\eta$ (T Hz$^{-1/2}$)', fontsize= labelfontsize, labelpad = -10)
    
    plt.tight_layout()

 
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "2DsensitivityIR.png", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotEtalonSensitivityKappa(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.45, 0.34*PRQuantumWidthFull, 
       r"cavity length, $L$ (m)", 
       r'mirror transmissivity, $T_{\text{m}}$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    
    
    lw = 1
    
    N = 400
    
    # mirror transmissivity array
    kexmin = np.log10(c_const/(2*1*nd))
    kexmax = 12
    kex = np.logspace(kexmin, kexmax, N)
    
    
    # cavity length array
    Lmin = -6
    Lmax = 0
    L = np.logspace(Lmin,Lmax,N)
    
    
    kexmat, Lmat = np.meshgrid(kex, L)
    
    Tmmat = kexmat*2*Lmat*nd/c_const
    
    a0 = afromalpha(alpha0, L)
    a1 = afromalpha(alpha1, L)
    TmBest = -calcBestTm(a0, a1)
    filt = np.logical_and(TmBest < 1, TmBest > 0, np.isfinite(TmBest))
    filt = np.logical_and(L > 1e-4, L < 0.1)
    
    T0 = TEtalon(kexmat, Lmat, alpha0)  
    T1 = TEtalon(Tmmat, Lmat, alpha1) 
    T = T0/2 + T1/2
    C = T0-T1
    P = PinEtalon(Tmmat, Lmat, alpha, I_sOpt, A)
    R = T*P/(hbar*omegas)
    eta = calcSensitivityCW_countrate(nuOpt, C, R)
    etaMinarg = np.argmin(eta)
    i, j = np.unravel_index(etaMinarg, eta.shape)
    
    
    print('min eta: ', eta[i,j])
    print('length: ', Lmat[i,j])
    print('contrast: ', C[i,j])
    print('Tm: ', Tmmat[i,j])
    print('power: ', P[i,j]*T[i,j])
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 1, 256))
    )
    p=ax.contourf(np.log10(Lmat), np.log10(kexmat), np.log10(eta), 25, cmap = 'viridis')
    
    # ax.scatter(np.log10(Lmat[i,j]), np.log10(Tmmat[i,j]), marker = '*', color = 'red', clip_on=False, zorder = 10)
    # ax.scatter(np.log10((np.log(alpha0) - np.log(alpha1))/(alpha0-alpha1)), np.log10(1), marker = '*', color = 'red', clip_on=False, zorder = 10)

    ticks = [np.max(np.log10(eta)), np.min(np.log10(eta))]
    cbar = fig.colorbar(
        ScalarMappable(norm=p.norm, cmap=p.cmap),
        ax = ax
    )
    # cbar = fig.colorbar(p, ax=ax, orientation='vertical', label = r'',ticks=ticks)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks], fontsize = numfontsize)
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks], fontsize = numfontsize)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$10^{{{val:.1f}}}$" for val in ticks], fontsize = numfontsize)
    cbar.set_label(label = r'sensitivity, $\eta$ (T Hz$^{-1/2}$)', fontsize= labelfontsize, labelpad = -10)
    
    plt.tight_layout()

 
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "LengthSensitivityPres.png", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()

def plotMicrodiskTheorySensitivity(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.45, 0.34*PRQuantumWidthFull, 
       r"cavity length, $L$ (m)", 
       r'$\kappa_{\text{ex}}/\kappa_{\text{abs}}$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    
    def Tmicrodisk(ka, ki, kex):
        return (ka + ki - kex)**2/(ka + kex + ki)**2
    
    def kex_opt(ka1, ka0):
        ka0 = np.asarray(ka0, dtype=complex)
        ka1 = np.asarray(ka1, dtype=complex)

        sqrt1 = np.sqrt(ka0**2 + 34*ka0*ka1 + ka1**2)

        numerator = (
            -(-ka0 - ka1)**3
            - 24 * ka0 * (-ka0 - ka1) * ka1
            - 8 * (-ka0**2 * ka1 - ka0 * ka1**2)
        )


        sol1 = (
            (ka0 + ka1) / 4.0
            - 0.25 * sqrt1
            - 0.5 * np.sqrt(
                            0.5 * (-ka0 - ka1)**2
                            + 4 * ka0 * ka1
                            - numerator / (2.0 * sqrt1)
                        )
        )
        sol2 = (
            (ka0 + ka1) / 4.0
            - 0.25 * sqrt1
            + 0.5 * np.sqrt(
                            0.5 * (-ka0 - ka1)**2
                            + 4 * ka0 * ka1
                            - numerator / (2.0 * sqrt1)
                        )
        )
        sol3 = (
            (ka0 + ka1) / 4.0
            + 0.25 * sqrt1
            - 0.5 * np.sqrt(
                            0.5 * (-ka0 - ka1)**2
                            + 4 * ka0 * ka1
                            + numerator / (2.0 * sqrt1)
                        )
        )
        sol4 = (
            (ka0 + ka1) / 4.0
            + 0.25 * sqrt1
            + 0.5 * np.sqrt(
                            0.5 * (-ka0 - ka1)**2
                            + 4 * ka0 * ka1
                            + numerator / (2.0 * sqrt1)
                        )
        )
        
        return sol1, sol2, sol3, sol4


    
    A = 0.5e-6**2
    
    
    Lmin = -9
    Lmax = -5
    
    lw = 1
    
    N = 400
    L = np.logspace(Lmin,Lmax,N)
    
    # mirror transmissivity array
    # kimin = 0
    # kimax = 10
    # ki = np.logspace(kimin, kimax, N)
    ki = 0e9
    
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    
    # cavity length array
    kexmin = np.log10(ki +kabs) - 2
    kexmax = np.log10(ki +kabs) + 2
    kex = np.logspace(kexmin,kexmax,N)
    
    
    kexOpt = positive_real_values(kex_opt(kabs1, kabs0))
    
    
    Lmat, kexmat = np.meshgrid(L, kex)
    
    Q = calcQMicrodisk(omegas, ki + kabs + kexmat)
    aMicrodisk = aMicrodiskOnRes(kabs, ki, kex)
   
    T0 = Tmicrodisk(kabs0, ki, kexmat)  
    T1 = Tmicrodisk(kabs1, ki, kexmat)
    T = T0/2 + T1/2
    C = T0-T1
    P = aMicrodiskRes2PinFibre(L*A, I_sOpt, aMicrodisk)
    R = T*P/(hbar*omegas)
    eta = np.abs(np.real(calcSensitivityCW_countrate(nu, C, R)))
    # etaMinarg = np.argmin(eta)
    idx = np.argpartition((eta).ravel(), 2)[:2]
    rows, cols = np.unravel_index(idx, C.shape)
    
    
    # print('Tm: ', Tmmat[i,j])
    # print('power: ', P[i,j]*T[i,j])
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 1, 256))
    )
    p=ax.contourf(np.log10(Lmat), np.log10(kexmat/(ki +kabs)), np.log10(np.abs(eta)), 25, cmap = 'viridis')#,vmin=np.min(np.log10(eta)), vmax=np.max(np.log10(eta)))
    print(kexOpt/kabs)
    # p=ax.pcolor(np.log10(Lmat), np.log10(kexmat/(ki +kabs)), np.log10(np.abs(eta)), cmap = 'viridis')#,vmin=np.min(np.log10(eta)), vmax=np.max(np.log10(eta)))
    
    # ax.plot(np.log10(L), np.log10(c_const/2/L/nd*0.01), 'r', lw = 0.2/0.3528)
    # ax.plot(np.log10(L), np.log10(L*0+(kabs0 + kabs1)/2 + ki), 'r', lw = 0.2/0.3528)
    # ax.plot(np.log10(L), np.log10(L*0+ kexOpt[0]), 'r', lw = 0.2/0.3528)
    # ax.plot(np.log10(L), np.log10(L*0+ np.sqrt(kabs0*kabs1)), 'r', lw = 0.2/0.3528)
    # ax.plot(np.log10(L), np.log10(L*0+ ((kabs0 + kabs1)/2 + ki)/(np.exp(-1)/2)), 'r', lw = 0.2/0.3528)
    # ax.plot(np.log10(L), np.log10(L*0 + kexOpt[0]/(ki +kabs) ), 'r--', lw = 0.2/0.3528)
    # for i, j in zip(rows, cols):
    #     ax.scatter(np.log10(Lmat[i,j]), np.log10(kexmat[i,j]/(ki +kabs)), marker = '*', color = 'red', clip_on=False, zorder = 10)
        
        # print('min eta: ', eta[i,j])
        # print('length: ', Lmat[i,j])
        # print('Q: ', Q[i,j])
        # print('contrast: ', C[i,j])
        # print('kex: ', kexmat[i,j])
    # ax.scatter(np.log10((np.log(alpha0) - np.log(alpha1))/(alpha0-alpha1)), np.log10(1), marker = '*', color = 'red', clip_on=False, zorder = 10)

    ticks = [np.max(np.log10(eta)), np.min(np.log10(eta))]
    cbar = fig.colorbar(
        ScalarMappable(norm=p.norm, cmap=p.cmap),
        ax = ax
    )
    # cbar = fig.colorbar(p, ax=ax, orientation='vertical', label = r'',ticks=ticks)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks], fontsize = numfontsize)
    ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks], fontsize = numfontsize)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$10^{{{val:.1f}}}$" for val in ticks], fontsize = numfontsize)
    cbar.set_label(label = r'sensitivity, $\eta$ (T Hz$^{-1/2}$)', fontsize= labelfontsize, labelpad = -10)
    
    plt.tight_layout()

 
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "2DsensitivityIRMicrocavity.png", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotNarrowTransmission(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.2, 0.2*PRQuantumWidthFull, 
       r"$\lambda$ (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False, tickpad = 4
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.2um_0/20260202/1040_1042nm_1nmps_averaged_norm.mat'
    
    
    data = loadmat(filePath, squeeze_me=True)
    # dataParams = loadmat(filePathParams, squeeze_me=True)
    
    W = data['W']
    TOut = data['TOut']
    
    # params = dataParams['params']
    lambmin = 1040.2
    lambmax = 1041.4
    filt = np.logical_and(W< lambmax, W>lambmin)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]
    ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = 3, zorder = 10, color = colorDarkRed)
    
    plt.tight_layout()
    (Tmin, Tmax) = ax.get_ylim()
    
    lamb = np.linspace(lambmin, lambmax, 1000)*1e-9
    nu = c_const/lamb
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0
    
    g = 0
    tau =1.004524991429497*np.exp(1j* -0.057891967231220)
    kex1 = nu0/2.509576482281477e+04
    kex2 = kex1
    ki1 = nu0/6.763434893840459e+03
    ki2 = ki1
    
    Delta = nu - nu0 
    print(nu0/ki1)
    fit= calcTransmission(kex1, kex2, ki1, ki2, Delta, g = 0, tau = tau)
    fits= calcTransmissionSinglet(kex1, ki1, Delta, tau = tau)
    
    
    
    ax.plot(lamb*1e9, fit, lw = 0.2/0.3528, color = colorRed)
    # ax.plot(lamb*1e9, fits, lw = 0.2/0.3528, color = 'k')
    Tmin = 0.3
    ax.set_xlim([lambmin-0.2,lambmax])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    ax.set_ylim([Tmin-0.1,1])
    ax.spines['left'].set_bounds(Tmin,1)
    ax.set_xticks([lambmin, lambmax])
    ax.set_yticks([Tmin, 1])
    ax.set_xticklabels([rf"${val:.1f}$" for val in [lambmin, lambmax]])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
    ax.yaxis.set_label_coords(-0.3, 0.55)
    ax.xaxis.set_label_coords(0.58, -0.25)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NarrowTransmission_4p2um.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
    



    
def plotWideTransmission(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthHalf, PRQuantumWidthHalf*0.5, 
       r"laser wavelength (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 0, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.2um_0/20260202/1030_1070nm_5nmps_averaged_norm.mat'
    filePathAbs =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260129/ODMR_20260129_CW__16_12dBAtt_4.2um_0_1__2.7768GHz_0MHz_001.mat'
    data = loadmat(filePath, squeeze_me=True)
    dataAbs = loadmat(filePathAbs, squeeze_me=True)
    
    W = data['W']
    TOut = data['TOut']
    
    lambmin = 1030
    lambmax = 1070
    
    filt = np.logical_and(W< lambmax, W>lambmin)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]
    skip = 1
    ax.plot(W[1::skip], TOut[1::skip], lw = 0.2/0.358, color = colorDarkRed)
    # ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = ms/2, zorder = 10, color = 'k')
    
    plt.tight_layout()
    
    ax.set_xlim([lambmin-1.2,lambmax])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)

    ax.set_ylim([0.25,1.01])
    ax.spines['left'].set_bounds(0.3,1)
    ax.set_xticks([lambmin, 1042, lambmax])
    ax.set_yticks([0.3,1])
    # ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
            
    ax.set_xticklabels([rf"${val:.0f}$" for val in [lambmin,1042, lambmax]])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "WideTransmission_4p2um.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

def plot2DData(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.3, 0.25*PRQuantumWidthFull, 
       r"wavelength, $\lambda$ (nm)", 
       r'MW detuning, $\Delta \omega_{\text{MW}}/2\pi$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    fig1, ax1 = plot_custom(
       PRQuantumWidthFull*0.2, 0.25*PRQuantumWidthFull, 
       r"$\Delta T$ ($\times 10^{-4}$)", 
       r'',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    fig2, ax2 = plot_custom(
       PRQuantumWidthFull*0.3, 0.2*PRQuantumWidthFull, 
       r'',
       r"", 
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    lw = 0.2/0.3528
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75, 1040.95
    WAbs = data['laserWLsRead'].item()
    WAbs = WAbs[0,:]
    filt = np.logical_and(WAbs< lambmax, WAbs>lambmin)

    WAbs= WAbs[filt]
    signalAbs = data['signal'].item()
    contrast = data['contrast'].item()
    signalAbs= signalAbs[:,filt]/np.max(signalAbs/contrast)*1e4
    f = data['freqsODMR'].item()*1e-6-2776.76
    
    
    filt = np.logical_and(f< 4, f>-4)

    f= f[filt]
    signalAbs= signalAbs[filt, :]
    
    X, Y = np.meshgrid(WAbs, f)
    
    signalY = np.mean(signalAbs[38:42,:], 0)
    signalX = np.mean(signalAbs[:, 480:520], 1)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    p=ax.pcolormesh(WAbs, Y, signalAbs, cmap=truncated_cmap, shading='auto')
    
    # cbar = fig.colorbar(p, ax=ax, orientation='horizontal', pad=0.18, label = r'$\Delta T$ ($\times 10^{-4}$)')
    # cbar.set_ticks([np.min(signalAbs), 0, np.max(signalAbs)])
    
    # cbar.set_ticklabels([rf"${val:.2f}$" for val in [np.min(signalAbs), 0, np.max(signalAbs)]])
    
    ax.plot([1040.854,1040.854],[-4,4], '--k', lw = 1)
    ax.plot([1040.75,1040.95],[0,0], '--k', lw = 1)
    
    ax1.plot(signalX, f, 'r-', lw = 0.2/0.3528)
    ax2.plot(WAbs, signalY, 'r-', lw = 0.2/0.3528)
    
    plt.tight_layout()
    ax.set_xlim([lambmin,lambmax])
    ax.set_ylim([-4,4])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    # ax.set_ylim([Tmin-0.1,1])
    # ax.spines['left'].set_bounds(Tmin,1)
    ax.set_xticks([lambmin, 1040.85, lambmax])
    ax.set_yticks([-4, -2, 0, 2, 4])
    # ax.set_yticks([Tmin, 1])
    ax.set_xticklabels([rf"${val:.2f}$" for val in [lambmin, 1040.85,lambmax]])
    ax.yaxis.set_label_position("right")
    # ax.set_yticklabels([rf"${val:.2f}$" for val in ax.get_yticks()])
    
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright = True)
    # ax.tick_params(axis='x', which='both', top=False, bottom=True)
    # ax.spines["bottom"].set_visible(True)
    
    ax.set_xlim([lambmin,lambmax])
    
    sigMin, sigMax = np.min(signalY), np.max(signalY)
    ax1.set_ylim([-4,4])
    # sigMin, sigMax = np.min(signalX), np.max(signalX)
    ax1.spines['top'].set_bounds(sigMin, sigMax)
    ax1.set_xticks([sigMin, sigMax])
    ax1.set_xticklabels([rf"${val:.2f}$" for val in ax1.get_xticks()])
    ax1.set_yticks([])
    ax1.xaxis.set_label_position("top")
    ax1.tick_params(axis='x', which='both', top=True, bottom = False, labeltop = True, labelbottom = False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    
    ax2.set_xlim([lambmin,lambmax])
    sigMin, sigMax = np.min(signalY), np.max(signalY)
    ax2.spines['left'].set_bounds(sigMax, sigMin)
    ax2.set_ylim([sigMax, sigMin])
    ax2.set_yticks([sigMin, sigMax])
    ax2.set_yticklabels([rf"${val:.2f}$" for val in ax2.get_yticks()])
    ax2.set_xticks([])
    # ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis='y', which='both', left = True, right = False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
 
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "2DSignal_4p2um.png", dpi=300, bbox_inches="tight",transparent=True
        )
        fig1.savefig(
            (_FIGURES_DIR + "/") + "yLine.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        fig2.savefig(
            (_FIGURES_DIR + "/") + "xLine.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
#%% main plotting functions

#%%% Figure 1
def plotZeemanSplitting(saveBool = False):
    # ----------------------------
    # Parameters
    # ----------------------------
    D = 2.87e9                    # Zero-field splitting [Hz]
    gamma_e = 28.024e9            # Electron gyromagnetic ratio [Hz/T]

    thetaB_deg = 2.0              # field angle relative to NV axis [deg]
    thetaB = np.deg2rad(thetaB_deg)

    B = np.linspace(0, 0.09, 1000)   # magnetic field [T]
    # ----------------------------
    # Spin-1 matrices
    # Basis: |+1>, |0>, |-1>
    # ----------------------------
    sqrt2 = np.sqrt(2)

    Sx = (1/sqrt2) * np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=complex)

    Sy = (1/sqrt2) * np.array([
        [0, -1j, 0],
        [1j, 0, -1j],
        [0, 1j, 0]
    ], dtype=complex)

    Sz = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, -1]
    ], dtype=complex)

    # For spin-1, zero-field splitting is usually D * Sz^2
    Sz2 = Sz @ Sz

    # ----------------------------
    # Arrays for results
    # ----------------------------
    energies = np.zeros((len(B), 3))
    f01 = np.zeros(len(B))
    f02 = np.zeros(len(B))
    f12 = np.zeros(len(B))

    # ----------------------------
    # Solve Hamiltonian
    # ----------------------------
    for i, Bmag in enumerate(B):
        # Put B in x-z plane
        Bx = Bmag * np.sin(thetaB)
        By = 0.0
        Bz = Bmag * np.cos(thetaB)

        H = D * Sz2 + gamma_e * (Bx * Sx + By * Sy + Bz * Sz)

        # Eigenvalues in ascending order
        evals, evecs = np.linalg.eigh(H)
        energies[i, :] = np.real(evals)

        # Transition frequencies between eigenstates
        f01[i] = evals[1] - evals[0]
        f02[i] = evals[2] - evals[0]
        f12[i] = evals[2] - evals[1]

    # ----------------------------
    # Plot eigenenergies
    # ----------------------------
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.35, 0.5*PRQuantumWidthHalf, 
       r"$B$", 
       r'',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False, spineWidth = 0.15
    )
    
    color = [colorDarkGreen, colorLightGreen, colorGreen, colorBlue]
    for n in range(3):
        ax.plot(B * 1e3, energies[:, n] / 1e9, lw=0.2/0.358, color = color[n])

    # ax.set_xlabel()

    min_x_value = 0
    max_x_value = np.max(B)*1000
    min_y_value = 0
    max_y_value = 6

    ax.set_xticks([0])
    ax.set_yticks([0, 2.87])

    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.015
    limScaley = 0.08
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=numfontsize,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.set_xlim([min_x_value-12, max_x_value + rangex*limScale])

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')



    ax.spines['left'].set_bounds(min_y_value,max_y_value)

    plt.tight_layout()
    plt.show()
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "ZeemanSplitting.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        
def plotSpinContrastMechanisms(saveBool = False):
   
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.65, 0.37*PRQuantumWidthHalf, 
       r"$\Delta \omega/2\pi$ (MHz)", 
       r'',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False, spineWidth = 0.15
    )
    
    lw = 0.2/0.3528
    pops, dets = solvePopulationDetuningLineplot(saveBool = False, 
                                    OmegaR = OmegaROpt,          
                                    I_p = I_pOpt,
                                    I_s = I_sOpt,
                                    bw = 0.02e9*2*np.pi,
                                    Npoints = 401,
                                    fastBool = False,
                                    plotBool = False,
                                    returnBool = True )
    p = pops
    dets /=2*np.pi
    pGreen = p[0] + p[1] + p[2]
    pPL = p[3] + p[4] + p[5]
    pIRPL = p[6]
    pIRAbs = p[7]
    
    pGreen = (pGreen/pGreen.min()-1) + 1
    pPL = pPL/pPL.max()
    pIRPLSingle = pIRPL/pIRPL.min()
    pIRAbs = -(pIRAbs/pIRAbs.min()-1) + 1
    
    def singletToHyperfine(dets, p, hyperfine=hyperfine):
        # Create interpolation function
        f = interp1d(dets, p, bounds_error=False, fill_value='extrapolate')
    
        # Keep same x-axis (or you could shrink it later if you want)
        detsNew = dets.copy()
    
        # Sum shifted versions: center, +hf, -hf
        pNew = (
            f(detsNew) +
            f(detsNew - hyperfine) +
            f(detsNew + hyperfine)
        )
    
        return detsNew, (pNew-3)/3.3+1
    
    detsNew, pGreen = singletToHyperfine(dets, pGreen)
    detsNew, pPL = singletToHyperfine(dets, pPL)
    detsNew, pIRPLT = singletToHyperfine(dets, pIRPLSingle)
    detsNew, pIRAbs = singletToHyperfine(dets, pIRAbs)
    
    detsNew /= 1e6
    dets /= 1e6
    
    min_x_value = -10
    max_x_value = 20
    min_y_value = 0.982
    max_y_value = 1.018
    
    filt = np.logical_and(dets < max_x_value, dets > min_x_value)
    # dets = dets[filt]
    
    
    lw = 0.2/0.3528
    
    # ax.plot(detsNew[filt], pGreen[filt], 'g', lw = lw)
    ax.plot(detsNew[filt], pPL[filt], 'r', lw = lw)
    ax.plot(dets[filt], pIRPLT[filt], color = 'k', lw = lw)
    ax.plot(dets[filt], (pIRPLSingle[filt]-1)/3.3+1, color = 'grey', linestyle = '--', lw = lw)
    # ax.plot(dets, pIRPL, color = 'k', lw = lw, linestyle = '--', zorder = -1)
    ax.plot(dets[filt], pIRAbs[filt], colorDarkRed, lw = lw)
    
    
    # ax.plot(40-detsNew[filt], pPL[filt], 'r', lw = lw)
    # ax.plot(40-detsNew[filt], pIRPLT[filt], color = 'k', lw = lw)
    # ax.plot(40-detsNew[filt], (pIRPLSingle[filt]-1)/3.3+1, color = 'grey', linestyle = '--', lw = lw)
    # # ax.plot(dets, pIRPL, color = 'k', lw = lw, linestyle = '--', zorder = -1)
    # ax.plot(40-detsNew[filt], pIRAbs[filt], colorDarkRed, lw = lw)
    

    ax.set_xticks([min_x_value, 0, max_x_value])
    ax.set_yticks([1])
    # ax.set_yticklabels([r'1 + $C$', r'$1$', r'1 - $C$'])

    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.06
    limScaley = 0.1
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=numfontsize,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMin])
    ax.set_xlim([min_x_value-1, max_x_value + rangex*limScale])

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')



    ax.spines['left'].set_bounds(min_y_value,max_y_value)

    plt.tight_layout()
    plt.show()
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "statePopulation.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        

#%%% Figure 2

def solveSensitivityVaryIRPowerInset(saveBool = False,
                            N_IR = 21,
                            OmegaR = OmegaROpt,
                            I_p = I_pOpt,
                            P_smin = -6,
                            P_smax = 5,
                            bw = 0.01e9*2*np.pi,
                            Npoints = 100,
                            A = 10e-6**2,
                            Larr = 125e-3*np.array([1e-4]),
                            plotSimple = False): #1e-4,1e-2, 1
    
    
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.7, PRQuantumWidthHalf*0.6,  
        r"$\log P/A$", 
        r'',
        labelPadX = 3, labelPadY = 0, commonY = False, commonX = True, wSpaceSetting=0,
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
        # print(alpha0)
        nu = np.zeros([N_IR])
        detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
        
        
        
        PoA_min_list = []
        eta_min_list = []
        
        for i, I_s in enumerate(tqdm(I_IR, desc="IR intensity")):
            if I_s > 1e13:
                break
            
            popOn =         nv_steady_state(
                                            **default_params,
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            )
            popOff = nv_steady_state(
                                            **default_params,
                                            Omega = 0, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            )
                
            R_on[i] = (popOn[7])
            R_off[i] = (popOff[7])
            C[i] = -(R_off[i]-R_on[i])
    
            while True:
                detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
                Cdet = np.zeros(Npoints)
                for k, det in enumerate(detunings):
                    params = default_params.copy()
                    params["detuning"] = det
                    popDet =    nv_steady_state(
                                                    **params,
                                                    Omega = OmegaR, #rad/s
                                                    I_s = I_s,   #rad/s
                                                    I_p = I_p, #W/m^2
                                                    )
                    Cdet[k] = -(R_off[i] - (popDet[7]))
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                # plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
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
        # print(C)
        
        lamb = 1042e-9
        eta = calcSensitivityCW(nu, C, Parr*(TOff + TOn)/2, lamb)
        eta0 = calcSensitivityCW(nu[0], C[0], Parr*(TOff[0] + TOn[0])/2, lamb)
        # print(eta)
        
        eta_min_idx = np.argmin(eta)   # index of minimum eta
        PoA_min = (I_IR[eta_min_idx])     # (optional) the minimum eta itself
        print(PoA_min, '  ')
        
        PoA_min_list.append(PoA_min)
        
        
        ax.loglog(I_IR, eta, lw = 0.2/0.358, color = colors[j] )
        # ax.loglog(I_IR, eta0, 'k--', lw = 0.2/0.358)

    if plotSimple == True:
        for j, L in enumerate(Larr):
            Parr = np.logspace(P_smin, P_smax, N_IR)
            I_IR = Parr/A
            p7 = np.zeros([N_IR])
            R_on = np.zeros([N_IR])
            R_off = np.zeros([N_IR])
            C = np.zeros([N_IR])
            M = rho_NV
            alpha0 = M*sigma_s
            # print(alpha0)
            nu = np.zeros([N_IR])
            detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
            
            
            PoA_min_list = []
            eta_min_list = []
            
            for i, I_s in enumerate(tqdm(I_IR, desc="IR intensity")):
                popOn =         nv_steady_state(
                                                **simple_params,
                                                Omega = OmegaR, #rad/s
                                                I_s = I_s,   #rad/s
                                                I_p = I_p, #W/m^2
                                                verboseResults = False
                                                )
                popOff = nv_steady_state(
                                                **simple_params,
                                                Omega = 0, #rad/s
                                                I_s = I_s,   #rad/s
                                                I_p = I_p, #W/m^2
                                                verboseResults = False
                                                )
                    
                R_on[i] = (popOn[7])
                R_off[i] = (popOff[7])
                C[i] = -(R_off[i]-R_on[i])
        
                while True:
                    detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
                    Cdet = np.zeros(Npoints)
                    for k, det in enumerate(detunings):
                        params = simple_params.copy()
                        params["detuning"] = det
                        popDet =    nv_steady_state(
                                                        **params,
                                                        Omega = OmegaR, #rad/s
                                                        I_s = I_s,   #rad/s
                                                        I_p = I_p, #W/m^2
                                                        )
                        Cdet[k] = -(R_off[i] - (popDet[7]))
                        
                    
                    peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                    # plt.plot(detunings/1e9/(2*np.pi),Cdet)
                    # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
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
            # print(C)
            
            lamb = 1042e-9
            eta = calcSensitivityCW(nu, C, Parr*(TOff + TOn)/2, lamb)
            eta0 = calcSensitivityCW(nu[0], C[0], Parr*(TOff[0] + TOn[0])/2, lamb)
            # print(eta)
            
            eta_min_idx = np.argmin(eta)   # index of minimum eta
            PoA_min = (I_IR[eta_min_idx])     # (optional) the minimum eta itself
            print(PoA_min, '  ')
            
            PoA_min_list.append(PoA_min)
            
            
            ax.loglog(I_IR, eta, lw = 0.2/0.358, color = 'k' , linestyle = '--', zorder=-1)
            # ax.loglog(I_IR, eta0, 'k--', lw = 0.2/0.358)

    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/lab-analysis-software/barclay-lab/analysisScripts/ODMR/DNVODMR/CW_IR/'
    
    filenameFit = 'CW_ODMR_IR_fit_results.csv';
    filenameModel = 'CW_ODMR_IR_fit_models.csv';
    filenameSignal = 'CW_ODMR_IR_signal_data.csv';

    dfFit = pd.read_csv(directory + filenameFit)
    
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0
    omega0 = nu0*2*np.pi
    
    kex = nu0/2.509576482281477e+04
    kc = nu0/6.763434893840459e+03
    
    ng = nd
    Veff = 18*(lamb0/ng)**3
    
    powerW = dfFit['Power_uW'].values*1e-6
    powerErrW = dfFit['PowerErr_uW'].values*1e-9
    Ncav = P2ResNcav(powerW, etaFibreIR, omega0, kex, kc)
    I = Ncav2I(Ncav, omega0, ng, Veff)
    # convert power which is attenuation to Rabi frequency
    
    v_MHz = dfFit['v_kHz'].values/1000
    vMin_MHz = dfFit['vMin_kHz'].values/1000
    vMax_MHz = dfFit['vMin_kHz'].values/1000
    C_au = dfFit['C_au'].values*100
    CMin_au = dfFit['CMin_au'].values*100
    CMax_au = dfFit['CMax_au'].values*100
    
    vErr = (np.abs(v_MHz-vMin_MHz) + np.abs(v_MHz-vMax_MHz))/2;
    CErr = (np.abs(C_au-CMin_au) + np.abs(C_au-CMax_au))/2;
    CovErr = np.sqrt(CErr**2 / v_MHz**2 + vErr**2 * C_au**2/v_MHz**4)*np.sqrt(powerW) 
    rat = 4/(3*np.sqrt(3)) * 1/gamma * np.abs(v_MHz*1e6/C_au*np.sqrt(h*c_const/lamb_s/powerW))
    rat /= rat.min()
    rat *= 5e-9
    numSamples = len(powerW)
    colorArray = sns.color_palette(palette='Reds', n_colors=int(np.ceil(1*numSamples))+4)
    colorArray = colorArray[3:-1]
    s = 3
    for i in range(numSamples):
        if i == 0:
            continue
        ax.errorbar(I[i],rat[i],  yerr=rat[i]*0.1, xerr = I[i]*0.1,
                    capthick=lw/2, elinewidth=lw/2, capsize=0, fmt='*', markersize = s, color = colorRed,  mec='None')




    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.yaxis.set_ticks_position('left')
    # ax.tick_params(axis='y', which='both', left=True, right=False)
    # ax.tick_params(axis='x', which='both', top=False, bottom=True)

    # min_x_value=1e6
    # max_x_value = 1e14
    # min_y_value=1e-14
    # max_y_value = 1e-6
    # # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xticks([I_sOpt, I_sat])
    # ax.set_xticklabels([r'$I_{\text{opt}}$',r'$I_{\text{sat}}$'])
    # ax.set_ylim([min_y_value/4, max_y_value*4])
    # ax.set_xlim([min_x_value, max_x_value])

    # ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    # ax.spines['left'].set_bounds(min_y_value,max_y_value)
    # ax.xaxis.set_label_position("bottom")
    # ax.xaxis.set_ticks_position('bottom')

    ax.set_xscale('log')
    ax.set_yscale('log')

    # ax.spines['left'].set_bounds(min_y_value,max_y_value)
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NaiveSensitivity.svg", dpi=300, bbox_inches="tight",transparent=True
            )
    plt.show()

def solveSensitivityVaryIRPower(saveBool = False,
                            N_IR = 101,
                            OmegaR = OmegaROpt,
                            I_p = I_pOpt,
                            P_smin = -4,
                            P_smax = 5,
                            bw = 0.01e9*2*np.pi,
                            Npoints = 100,
                            A = 10e-6**2,
                            Larr = 1e-3*np.array([1])): # 1e-4,1e-2,
    
    
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.55, PRQuantumWidthHalf*0.6,  
        r"$\log I_{\text{s}}$", 
        r'',
        labelPadX = 3, labelPadY = 0, commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False, spineWidth = 0.15
    )
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/lab-analysis-software/barclay-lab/analysisScripts/ODMR/DNVODMR/CW_IR/'
    
    filenameFit = 'CW_ODMR_IR_fit_results.csv';
    filenameModel = 'CW_ODMR_IR_fit_models.csv';
    filenameSignal = 'CW_ODMR_IR_signal_data.csv';

    dfFit = pd.read_csv(directory + filenameFit)
    
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0
    omega0 = nu0*2*np.pi
    
    kex = nu0/2.509576482281477e+04
    kc = nu0/6.763434893840459e+03
    
    ng = nd
    Veff = 18*(lamb0/ng)**3
    
    powerW = dfFit['Power_uW'].values*1e-6*np.sqrt(etaFibreIR)
    powerErrW = dfFit['PowerErr_uW'].values*1e-9
    Ncav = P2ResNcav(powerW, etaFibreIR, omega0, kex, kc)
    I = Ncav2I(Ncav, omega0, ng, Veff)
    # convert power which is attenuation to Rabi frequency
    
    v_MHz = dfFit['v_kHz'].values/1000
    vMin_MHz = dfFit['vMin_kHz'].values/1000
    vMax_MHz = dfFit['vMin_kHz'].values/1000
    C_au = dfFit['C_au'].values*100
    CMin_au = dfFit['CMin_au'].values*100
    CMax_au = dfFit['CMax_au'].values*100
    
    vErr = (np.abs(v_MHz-vMin_MHz) + np.abs(v_MHz-vMax_MHz))/2;
    CErr = (np.abs(C_au-CMin_au) + np.abs(C_au-CMax_au))/2;
    CovErr = np.sqrt(CErr**2 / v_MHz**2 + vErr**2 * C_au**2/v_MHz**4)*np.sqrt(powerW) 
    rat = 4/(3*np.sqrt(3)) * 1/gamma * np.abs(v_MHz*1e6/C_au*np.sqrt(h*c_const/lamb_s/powerW))
    rat /= rat.min()
    rat *= 7e-9
    numSamples = len(powerW)
    colorArray = sns.color_palette(palette='Reds', n_colors=int(np.ceil(1*numSamples))+4)
    colorArray = colorArray[3:-1]
    s = 3
    lw = 0.16/0.3528
    for i in range(numSamples):
        if i == 0:
            continue
        ax.errorbar(I[i],rat[i]/np.min(rat),  yerr=rat[i]*0.1, xerr = I[i]*0.1,
                    capthick=lw/2, elinewidth=lw/2, capsize=lw/2, fmt='.', markersize = s, color = 'k',  mec='None')


    
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
        # print(alpha0)
        nu = np.zeros([N_IR])
        detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
        
        
        
        PoA_min_list = []
        eta_min_list = []
        
        for i, I_s in enumerate(tqdm(I_IR, desc="IR intensity")):
            if I_s > 1e13:
                break
            
            popOn =         nv_steady_state(
                                            **default_params,
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            )
            popOff = nv_steady_state(
                                            **default_params,
                                            Omega = 0, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            )
                
            R_on[i] = (popOn[7])
            R_off[i] = (popOff[7])
            C[i] = -(R_off[i]-R_on[i])
    
            while True:
                detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
                Cdet = np.zeros(Npoints)
                for k, det in enumerate(detunings):
                    params = default_params.copy()
                    params["detuning"] = det
                    popDet =    nv_steady_state(
                                                    **params,
                                                    Omega = OmegaR, #rad/s
                                                    I_s = I_s,   #rad/s
                                                    I_p = I_p, #W/m^2
                                                    )
                    Cdet[k] = -(R_off[i] - (popDet[7]))
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                # plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
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
        # print(C)
        
        lamb = 1042e-9
        eta = calcSensitivityCW(nu, C, Parr*(TOff + TOn)/2, lamb)
        eta0 = calcSensitivityCW(nu[0], C[0], Parr*(TOff[0] + TOn[0])/2, lamb)
        # print(eta)
        
        eta_min_idx = np.argmin(eta)   # index of minimum eta
        PoA_min = (I_IR[eta_min_idx])     # (optional) the minimum eta itself
        print(PoA_min, '  ')
        
        PoA_min_list.append(PoA_min)
        
        
        ax.loglog(I_IR, eta/np.nanmin(eta), lw = lw, color = colors[j] )
        # ax.loglog(I_IR, eta0, 'k--', lw = 0.2/0.358)
    etaComplexModel = eta
    for j, L in enumerate(Larr):
        Parr = np.logspace(P_smin, P_smax, N_IR)
        I_IR = Parr/A
        p7 = np.zeros([N_IR])
        R_on = np.zeros([N_IR])
        R_off = np.zeros([N_IR])
        C = np.zeros([N_IR])
        M = rho_NV
        alpha0 = M*sigma_s
        # print(alpha0)
        nu = np.zeros([N_IR])
        detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
        
        
        PoA_min_list = []
        eta_min_list = []
        
        for i, I_s in enumerate(tqdm(I_IR, desc="IR intensity")):
            popOn =         nv_steady_state(
                                            **simple_params,
                                            Omega = OmegaR, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            )
            popOff = nv_steady_state(
                                            **simple_params,
                                            Omega = 0, #rad/s
                                            I_s = I_s,   #rad/s
                                            I_p = I_p, #W/m^2
                                            )
                
            R_on[i] = (popOn[7])
            R_off[i] = (popOff[7])
            C[i] = -(R_off[i]-R_on[i])
    
            while True:
                detunings = np.linspace(detuning - bw, detuning + bw, Npoints)
                Cdet = np.zeros(Npoints)
                for k, det in enumerate(detunings):
                    params = simple_params.copy()
                    params["detuning"] = det
                    popDet =    nv_steady_state(
                                                    **params,
                                                    Omega = OmegaR, #rad/s
                                                    I_s = I_s,   #rad/s
                                                    I_p = I_p, #W/m^2
                                                    )
                    Cdet[k] = -(R_off[i] - (popDet[7]))
                    
                
                peaks, FWHM = solvePeakWidthAndHeight(detunings/1e9/(2*np.pi), Cdet)
                # plt.plot(detunings/1e9/(2*np.pi),Cdet)
                # print('FWHM ', FWHM, ' bw ', bw/1e9/(2*np.pi))
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
        # print(C)
        
        lamb = 1042e-9
        eta = calcSensitivityCW(nu, C, Parr*(TOff + TOn)/2, lamb)
        eta0 = calcSensitivityCW(nu[0], C[0], Parr*(TOff[0] + TOn[0])/2, lamb)
        # print(eta)
        
        eta_min_idx = np.argmin(eta)   # index of minimum eta
        PoA_min = (I_IR[eta_min_idx])     # (optional) the minimum eta itself
        print(PoA_min, '  ')
        
        PoA_min_list.append(PoA_min)
        
        
        ax.loglog(I_IR, eta/np.nanmin(etaComplexModel), lw = lw, color = colors[j+2] , linestyle = '--', zorder=-1)
        # ax.loglog(I_IR, eta0, 'k--', lw = 0.2/0.358)


    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    min_x_value=1e6
    max_x_value = 1e14
    min_y_value=1e-12*1e10
    max_y_value = 1e-6*1e10
    # ax.set_xticks([])
    ax.set_yticks([1e-2, 1e0, 1e2, 1e4])
    # ax.set_yticklabels([r'$-2$',r'$0$', r'$2$', r'$4$'])
    ax.set_yticks([], minor = True)
    ax.set_xticks([I_sOpt, I_sat])
    ax.set_xticklabels([r'$I_{\text{opt}}$',r'$I_{\text{sat}}$'])
    ax.set_ylim([min_y_value/4, max_y_value])
    ax.set_xlim([min_x_value/6, max_x_value])

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
  
def plotSensitivityRelation(saveBool = False,
                            OmegaR = OmegaROpt,
                            I_p = I_pOpt,
                            bw = 0.01e9*2*np.pi,
                            Npoints = 11,
                            A = 10e-6**2,
                            Larr = 0.1*np.array([1e-4,1e-2,1])):# 
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.5, PRQuantumWidthHalf*0.6,  
        r"$Q_{\text{cav}}$", 
        r'',
        labelPadX = 3, labelPadY = 9 , commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize-1,
        widthBool =False, spineWidth = 0.15
    )
    
    colors = [colorLightRed, colorRed, colorDarkRed, 'k', 'r']
    
    Qi = np.logspace(0, 9, 100)
    omega0 = omegas/2/np.pi
    kip = omega0/Qi
    
    kabs0 = calcEnergyLossRate(nd, alpha0)
    kabs1 = calcEnergyLossRate(nd, alpha1)
    
    # If you still want ki defined relative to the *mean absorption*:
    kabs_mean = 0.5 * (kabs0 + kabs1)
    
    
    
    kcav = kip + kabs_mean/4.5/10
    kex = kcav.min()/10
    kex = 1e9
    # print(kcav)
    
    etaUC0 = (kex + kcav)/np.sqrt(kex)*45
    etaSimpleUC = (kip)/np.sqrt(kex)*45
    
    # ax.loglog(Qi, etaSimpleUC, lw = 0.2/0.358, color =colors[0], linestyle = '--' )
    
    # kcav = kip + kabs_mean/10
    # kex = kcav/10
    
    # etaUC = (kex + kcav)/np.sqrt(kex)*10
    # etaSimpleUC = (kip)/np.sqrt(kex)*10
    
    # ax.loglog(Qi, etaUC, lw = 0.2/0.358, color =colors[1])
    # ax.loglog(Qi, etaSimpleUC, lw = 0.2/0.358, color =colors[1], linestyle = '--' )
    
    kcav = kip + kabs_mean/1
    # kex = kcav.min()/10
    
    print(kex)
    
    etaUC = (kex + kcav)/np.sqrt(kex)
    etaSimpleUC = (kip)/np.sqrt(kex)
    
    ax.loglog(Qi, etaUC0/etaUC.min(), lw = 0.2/0.358, color =colors[0])
    ax.loglog(Qi, etaUC/etaUC.min(), lw = 0.2/0.358, color =colors[2])
    # ax.loglog(omega0/kex*np.ones(2), [1e0, 1e9], lw = 0.2/0.358, color = 'k')
    
    # ax.loglog(Qi, etaSimpleUC, lw = 0.2/0.358, color =colors[2], linestyle = '--' )
    
    
    plt.tight_layout()
        
        
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    min_x_value=1
    max_x_value = 1e9
    min_y_value=0.3
    max_y_value = 1e7
    ticks = [1, 1e3, 1e6, 1e9]
    ax.set_xticks(ticks)
    ax.set_yticks([1,1e2,1e4, 1e6])
    # ax.set_yticklabels([r'$0$', r'$1$', r'$2$', r'$3$', r'$4$'])
    ax.set_yticks([], minor = True)
    ax.set_ylim([min_y_value, max_y_value*4])
    
    # Create labels like 10^x
    # labels = [r"$10^{%.0f}$" % int(np.round(np.log10(t))) for t in ticks]
    # ax.set_xticklabels(labels)
    
    # ax.set_ylim([min_y_value/4, max_y_value*4])
    ax.set_xlim([min_x_value/6, max_x_value*4])

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.spines['left'].set_bounds(1,max_y_value*2)
    # ax.xaxis.set_label_position("bottom")
    # ax.xaxis.set_ticks_position('bottom')

        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "SensitivityRelation.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotSensitivityComparison(saveBool = False,
                            OmegaR = OmegaROpt,
                            I_p = I_pOpt,
                            bw = 0.01e9*2*np.pi,
                            Npoints = 11,
                            A = 10e-6**2,
                            Larr = 0.1*np.array([1e-4,1e-2,1])):# 
    fig, ax = plot_custom(
       PRQuantumWidthHalf, PRQuantumWidthHalf*0.6,  
        r"$L$ (m)", 
        r"$\eta\sqrt{A}$ (Tm/$\sqrt{\text{Hz}})$",
        labelPadX = 3, labelPadY = 9 , commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize-1,
        widthBool =False, spineWidth = 0.15
    )
    
    LArr_m = np.logspace(-8, 0, 101)
    Tm = np.logspace(-8, 0, 1001)
    Tm_min_list = []
    eta_min_list = []
    etaDiamondIdealCavity = []
    powerIdealCavity = []
    A = 1e-9
    
    for j, L in enumerate(LArr_m):
        T0 = TEtalon(Tm, L, alpha0)  
        T1 = TEtalon(Tm, L, alpha1) 
        T = T0/2 + T1/2
        C = (T0-T1)/T
        P = PinEtalon(Tm, L, alpha, I_sOpt, A)
        Pdet = I_sOpt*A*Tm*np.exp(L*alpha)
        eta = calcSensitivityCW(nuOpt, C, Pdet, lamb_s)
        eta_min = eta.min()
        powerIdealCavity.append(P[np.argmin(eta)])
        etaDiamondIdealCavity.append(eta_min)
        idx = np.argmin(eta)
        Tm_min_list.append(Tm[idx])

    ax.plot(LArr_m, np.array(etaDiamondIdealCavity), lw = 0.2/0.358)
    
    LArr_m = np.logspace(-8, 0.5, 101)
    
    etaBulk = []    
    etaBulkSimple = []    
    etaBulkLin = []    
    
    for j, L in enumerate(LArr_m):
        etaBulkSimple.append(calcBulkSensitivitySimple(L, A))
        etaBulk.append(calcBulkSensitivity(L, A))
        etaBulkLin.append(calcBulkSensitivityLin(L, A))
    
    
   
    ax.plot(LArr_m, np.array(etaBulk), lw = 0.2/0.358)
    ax.plot(LArr_m, np.array(etaBulkSimple), lw = 0.2/0.358 )
    ax.plot(LArr_m, np.array(etaBulkLin), lw = 0.2/0.358, color = 'k' , linestyle = '--')
    
    
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    # ax.loglog(Qi, etaUC0/etaUC.min(), lw = 0.2/0.358, color =colors[0])
    # ax.loglog(Qi, etaUC/etaUC.min(), lw = 0.2/0.358, color =colors[2])
    # ax.loglog(omega0/kex*np.ones(2), [1e0, 1e9], lw = 0.2/0.358, color = 'k')
    
    # ax.loglog(Qi, etaSimpleUC, lw = 0.2/0.358, color =colors[2], linestyle = '--' )
    
    
    plt.tight_layout()
        
        
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    min_x_value=LArr_m.min()
    max_x_value = LArr_m.max()
    min_y_value=0.3
    max_y_value = 1e7
    # ticks = [1, 1e3, 1e6, 1e9]
    # ax.set_xticks(ticks)
    # ax.set_yticks([1,1e2,1e4, 1e6])
    # ax.set_yticklabels([r'$0$', r'$1$', r'$2$', r'$3$', r'$4$'])
    # ax.set_yticks([], minor = True)
    # ax.set_ylim([min_y_value, max_y_value*4])
    
    # Create labels like 10^x
    # labels = [r"$10^{%.0f}$" % int(np.round(np.log10(t))) for t in ticks]
    # ax.set_xticklabels(labels)
    
    # ax.set_ylim([min_y_value/4, max_y_value*4])
    # ax.set_xlim([min_x_value/6, max_x_value*4])

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    # ax.spines['left'].set_bounds(1,max_y_value*2)
    # ax.xaxis.set_label_position("bottom")
    # ax.xaxis.set_ticks_position('bottom')

        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "SensitivityRelation.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
  
def plotEtalonSensitivity1D(saveBool = False,
                            OmegaR = OmegaROpt,
                            I_p = I_pOpt,
                            bw = 0.01e9*2*np.pi,
                            Npoints = 11,
                            A = 10e-6**2,
                            Larr = 0.1*np.array([1e-4,1e-2,1])):# 
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.5, PRQuantumWidthHalf*0.55,  
        r"$\mathcal{F}$", 
        r'',
        labelPadX = 3, labelPadY = 0 , commonY = False, commonX = True, wSpaceSetting=0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False, spineWidth = 0.15
    )
    
    colors = [colorLightBlue, colorBlue, colorDarkBlue, 'k', 'r']
    
    N = 40000
    
    # mirror transmissivity array
    Tmin = -7
    Tmax = 0
    Tm = np.logspace(Tmin, Tmax, N)
    
    Tm_min_list = []
    eta_min_list = []
    scale = 1
    alpha0Local = alpha0*scale
    alpha1Local = alpha1*scale
    alphaLocal = alpha*scale
    
    for j, L in enumerate(Larr):
        T0 = TEtalon(Tm, L, alpha0Local)  
        T1 = TEtalon(Tm, L, alpha1Local) 
        T = T0/2 + T1/2
        T = TEtalon(Tm, L, alphaLocal) 
        C = (T0-T1)/T
        P = PinEtalon(Tm, L, alphaLocal, I_sOpt, A)
        eta = calcSensitivityCW(nuOpt, C, P*T, lamb_s)
        
        
        eta_min_idx = np.argmin(eta)   # index of minimum eta
        Tm_min = Tm[eta_min_idx]       # corresponding Tm value
        eta_min = eta[eta_min_idx]     # (optional) the minimum eta itself
        # print(eta_min, '  ', Tm_min)
        
        Tm_min_list.append(Tm_min)
        eta_min_list.append(eta_min)
        
        Fin = np.pi*np.sqrt(1-Tm)/(Tm)
        ax.loglog(Fin, eta, lw = 0.2/0.358, color = colors[j] )
        plt.tight_layout()
        
        
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)

    min_x_value=0.01
    max_x_value = 1e8
    min_y_value=3e-14
    max_y_value = 1e-6
    ticks = [1, 1e4, 1e8]
    ax.set_xticks(ticks)
    ax.set_yticks([])
    ax.set_ylim([min_y_value/4, max_y_value*4])
    
    # Create labels like 10^x
    # labels = [r"$10^{%.0f}$" % int(np.round(np.log10(t))) for t in ticks]
    # ax.set_xticklabels(labels)
    
    # ax.set_ylim([min_y_value/4, max_y_value*4])
    ax.set_xlim([min_x_value/4, max_x_value*4])

    # ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    # ax.spines['left'].set_bounds(min_y_value,max_y_value)
    # ax.xaxis.set_label_position("bottom")
    # ax.xaxis.set_ticks_position('bottom')

        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "TmSensitivityPres.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotEtalonSensitivityCavityLength(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.5, 0.2*PRQuantumWidthFull, 
       r'cavity length, $L$ (m)',
       r"$\eta$ (T Hz$^{-1/2}$)",
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    A = 24e-6**2
    
    lw = 1
    
    N = 100
    
    # mirror transmissivity array
    T = 1
    
    # cavity length array
    Lmin = -6
    Lmax = 0
    Lm = np.logspace(Lmin, Lmax, N)
    
    a0 = afromalpha(alpha0, Lm)
    a1 = afromalpha(alpha1, Lm)
    
    T0 = TEtalon(T, Lm, alpha0)  
    T1 = TEtalon(T, Lm, alpha1) 
    Tavg = T0/2 + T1/2
    C = T0-T1
    P = PinEtalon(T, Lm, alpha, I_sOpt, A)
    R = Tavg*P/(hbar*omegas)
    eta = calcSensitivityCW_countrate(nu, C, R)
    etaMinarg = np.argmin(np.abs(T))
    
    print('min eta: ', eta[etaMinarg])
    print('contrast: ', C[etaMinarg])
    print('L: ', Lm[etaMinarg])
    print('power: ', P[etaMinarg]*Tavg[etaMinarg])
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 1, 256))
    )
    p=ax.loglog(Lm, eta)
    
    ax.set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_xticks([],minor=True)
    ax.set_xlim([1e-6,1])
    ax.set_ylim([1e-14,1e-4])
    plt.tight_layout()
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "LengthSensitivityPres.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
#%%% Figure 3
def plotIRSpectrometerCombined(saveBool = False):
    
    def readDatFile(directory, filename,scalar=1, sensitivityBool = False):
        dfBack = pd.read_csv(directory + filename)
        dfBack[['Wavelength', 'S1']] = dfBack.iloc[:, 0].str.split('\t', expand=True)
        dfBack['Wavelength'] = pd.to_numeric(dfBack['Wavelength'], errors='coerce')
        dfBack['S1'] = pd.to_numeric(dfBack['S1'], errors='coerce')
        
        filt = np.array((dfBack.Wavelength) >= min_x_value) & np.array(dfBack.Wavelength <= max_x_value)
        wl = dfBack.Wavelength[filt]
        wl = wl.values
        intensity = (dfBack.S1[filt])
        intensity = intensity.values/intensity.values[0]*scalar
        if sensitivityBool:
            return wl, intensity, (dfBack.S1[filt])
            
        else:
            return wl, intensity

    fig, ax = plot_custom(
       PRQuantumWidthHalf*1.1, PRQuantumWidthHalf*0.8, 
       r"emission wavelength (nm)", 
       [r"PL intensity, $I_{\text{PL}}$ (au)", r"$\Delta I_{\text{PL}}$ (au)"] 
       , labelPadX = -2, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True, widthRatios = [1], heightRatios = [3,2]
    )
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/B14_4p2_IR_Emission_Spectrum/'
    filenameBack = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra_Bypass-FiberPL.dat'
    filenameOn = 'MW_ON_1000-1400_100s_Exposure_150umSlit_Spectra.dat'
    filenameOff = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra.dat'

    min_x_value = 1000       
    max_x_value = 1300
    min_y_value = np.array([0,-1])
    max_y_value = np.array([1,1])

    offset = [0, 700, 1500]
    scale = [1.05, 1, 40]
    color = [[1,0,0], [0.5,0,0], [0,0,0]]
    
    
    wlBack, intensityBack = readDatFile(directory, filenameBack, 0.8)
    wlOn, intensityOn= readDatFile(directory, filenameOn)
    wlOff, intensityOff = readDatFile(directory, filenameOff)
    temp, temp, counts = readDatFile(directory, filenameOff, sensitivityBool=True)
    
    filt = wlOff > 1042
    spectrumcountrate = counts[filt]/100
    countrate = spectrumcountrate.sum()
    
    sensIRPL = calcSensitivityCW_countrate(nuOpt, 0.2, countrate)
    print('IR PL sensitivity: ', sensIRPL)
    
    
    intensityOn = intensityOn - intensityBack
    intensityOff = intensityOff - intensityBack
    
    maxInt = np.max([intensityOn, intensityOff])
    print(maxInt)
    
    intensityOn = intensityOn/maxInt
    intensityOff = intensityOff/maxInt
    
    ax[1].plot(wlOn, (intensityOn-intensityOff)/np.max(intensityOn-intensityOff), linestyle = '-',color = 'k', lw = 0.2/0.3528, zorder = 0)

    ax[0].plot(wlOn, intensityOn, linestyle = '-',color = colorDarkRed, lw = 0.2/0.358*1.5, zorder = 0)
    ax[0].plot(wlOn, intensityOff, linestyle = '-',color = colorLightRed, lw = 0.2/0.358*1.5, zorder = 0)
    # ax.plot(wlOff, intensityOff, linestyle = '-',color = 'b', lw = 0.2/0.3528, zorder = 0)



    # area = areaNVPL;# - areaFiberPL;
    # area = area/max(area)
    # print('Collection efficiency of NVPL: ',area[0])
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.03
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    for i, axx in enumerate(ax):
        axx.spines['bottom'].set_bounds(min_x_value,max_x_value)
        axx.xaxis.set_label_position("bottom")
        axx.xaxis.set_ticks_position('bottom')
        
        axx.spines["right"].set_visible(False)
        axx.spines["top"].set_visible(False)
        axx.spines["bottom"].set_visible(False)
        axx.yaxis.set_ticks_position('left')
        axx.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        axx.tick_params(axis='x', which='both', top=False, bottom=False)
    
        axx.set_xticks([])
        axx.spines['left'].set_bounds(min_y_value[i],max_y_value[i])


        axx.set_ylim([min_y_value[i] - limScaleMin[i], max_y_value[i]])
        axx.tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
        axx.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels

    
    ax[1].spines["bottom"].set_visible(True)
    ax[1].set_xticks([1000, 1042, 1300])
    ax[1].set_yticks([-1,0,1])
    ax[0].set_yticks([0,1])
    ax[1].tick_params(axis='x', which='both', top=False, bottom=True)
    ax[0].tick_params(axis='x', which='both', top=False, bottom=False)
    ax[0].tick_params(labelbottom=False)
    ax[1].yaxis.labelpad = 2
    fig.subplots_adjust(hspace=0.1)
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "IREmissionSpectrometry.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        
    plt.show()

def plotIRSpectrometerODMR(saveBool = False):
    min_x_value = 1000
    max_x_value=1500
    def readDatFile(directory, filename,scalar=1 ):
        dfBack = pd.read_csv(directory + filename)
        dfBack[['Wavelength', 'S1']] = dfBack.iloc[:, 0].str.split('\t', expand=True)
        dfBack['Wavelength'] = pd.to_numeric(dfBack['Wavelength'], errors='coerce')
        dfBack['S1'] = pd.to_numeric(dfBack['S1'], errors='coerce')
        
        filt = np.array((dfBack.Wavelength) >= min_x_value) & np.array(dfBack.Wavelength <= max_x_value)
        wl = dfBack.Wavelength[filt]
        wl = wl.values
        intensity = (dfBack.S1[filt])
        intensity = intensity.values/intensity.values[0]*scalar
        return wl, intensity

    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/B14_4p2_IR_Emission_Spectrum/20250903/'
    directoryBack = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsOpticalSpectra/B14/B14_4p2_IR_Emission_Spectrum/'
    filenameBack = 'MW_OFF_1000-1400_100s_Exposure_150umSlit_Spectra_Bypass-FiberPL.dat'
    
    wlBack, intensityBack = readDatFile(directoryBack, filenameBack, 0.8)
    
    # Get list of .dat files
    files = [f for f in os.listdir(directory) if f.endswith('.dat')]
    files.sort()    
    
    N = len(files)
    
    countsTotalNorm = np.zeros(N)
    countsTotal = np.zeros(N)
    
    freqs = np.linspace(2.856, 2.886, 16)
    
    # plt.figure()
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('counts (au)')
    
    # Background data (replace with your loaded background arrays)
    
    intensityBack = intensityBack / intensityBack[0] * 1500 - 100
    
    # Remove duplicate wavelengths
    wlBack, idx = np.unique(wlBack, return_index=True)
    intensityBack = intensityBack[idx]
    
    # plt.plot(wlBack, intensityBack)
    
    # Interpolator
    background_interp_func = interp1d(wlBack, intensityBack, kind='linear', fill_value='extrapolate')
    
    for k, file in enumerate(files):
        # print(f"Now processing {os.path.basename(file)}")
    
        # Load .dat file
        wl, counts = readDatFile(directory, file, 0.8)
    
        background_interp = background_interp_func(wl)
        # counts = counts - background_interp
    
        filtNorm = (wl <= 1035) & (wl >= 900)
        filt = (wl > 1035)
    
        countsTotalNorm[k] = np.sum(counts[filtNorm])
        countsTotal[k] = np.sum(counts[filt])
    
        # plt.plot(wl, counts)

# Second figure

    fig, ax = plot_custom(
        PRQuantumWidthHalf*0.5, PRQuantumWidthHalf*0.35, 
        r"$\omega_{\text{MW}}/2\pi$ (GHz)", 
        r"$\Delta I_{\text{PL}}$ (au)",
        labelPadX = 2, labelPadY = 7, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
        spineColor = 'black', tickColor = 'black', textColor = 'black',
        nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
        widthBool =False
    )
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    ax.set_xticks([2.87])
    ax.set_yticks([1,1.2])
    ax.set_ylim([1,1.2])
    # plt.figure()
    ax.scatter(freqs, countsTotal / countsTotalNorm / np.min(countsTotal / countsTotalNorm), color = colorDarkRed, s = 1)
    ax.plot(freqs, countsTotal / countsTotalNorm / np.min(countsTotal / countsTotalNorm), color = colorDarkRed, lw = 0.2/0.358)
    print(countsTotal)
    
    
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "IREmissionSpectrometryODMR.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
#%%% Figure 4

def plotChangeInTransmissionCartoon(saveBool = False):
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*1, 0.37*PRQuantumWidthFull, 
       [r"$\Delta \omega_{\text{opt}}$",r"$\Delta \omega_{\text{MW}}$"], 
       [r"$T$ (au)",r"$T$ (au)"],
       labelPadX = 3, labelPadY = 3, commonY = True, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow =2, ncol = 2, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False
    )
    lw = 1
    

    lamb0 = 1040.77e-9
    nu0 = c_const/lamb0
    FWHM = 30e9
    bw = FWHM*4
    nu = np.linspace(nu0 - bw, nu0+bw, 1000)
    
    f0 = 2.8e9
    fbw = hyperfine*4
    f = np.linspace(f0 - fbw, f0+fbw, 1000)
    gammaMW = 0.5e6
    ax = axArr[0,0]
    
    g = 0
    kex1 = FWHM/4
    kex2 = kex1
    ki1 = kex1*2
    ki2 = ki1*3
    Delta = nu - nu0 
    print(nu0/ki1)
    T= calcTransmission(kex1, kex2, ki1, ki1, Delta, g = 0, tau = 1)
    T2= calcTransmission(kex1, kex2, ki2, ki2, Delta, g = 0, tau = 1)
    T3 = microwaveDetuning1(f, gammaMW, f0, (np.min(T2) - np.min(T)), np.min(T2))
    
    ax.plot(nu, T, lw = 0.2/0.358, color = colorLightRed)
    ax.plot(nu, T2, lw = 0.2/0.358, color = colorDarkRed)
    axArr[0,1].plot(nu, T3-np.min(T3)+ np.min(T), lw = 0.2/0.358, color = colorLightBlue)
    
    ax = axArr[1,0]
    
    g = 0
    kex1 = FWHM/4
    kex2 = kex1
    ki1 = kex1*4
    ki2 = ki1*1.15
    Delta = nu - nu0 
    print(nu0/ki1)
    T= calcTransmission(kex1, kex2, ki1, ki1, Delta, g = 0, tau = 1)
    T2= calcTransmission(kex1, kex2, ki2, ki2, Delta, g = 0, tau = 1)
    T3 = microwaveDetuning1(f, gammaMW, f0, (np.min(T2) - np.min(T)), np.min(T2))
    
    ax.plot(nu, T, lw = 0.2/0.358, color = colorLightRed)
    ax.plot(nu, T2, lw = 0.2/0.358, color = colorDarkRed)
    axArr[1,1].plot(nu, T3-np.min(T3)+ np.min(T2), lw = 0.2/0.358, color = colorLightBlue)
    
    Tmin = 0
    
    for ax in axArr[0,:]:
        ax.set_xlim([nu0 - bw*1.15,nu0 + bw])
        ax.spines['bottom'].set_bounds(nu0 - bw, nu0 + bw)
        
        ax.set_ylim([Tmin-0.05,1])
        ax.spines['left'].set_bounds(Tmin,1)
        ax.set_xticks([nu0 ])
        ax.set_yticks([Tmin, 1])
        ax.set_xticklabels([r'$0$'])
        ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        ax.spines["bottom"].set_visible(True)
    
    for ax in axArr[1,:]:
        ax.set_xlim([nu0 - bw*1.15,nu0 + bw])
        ax.spines['bottom'].set_bounds(nu0 - bw, nu0 + bw)
        
        ax.set_ylim([Tmin-0.05,1])
        ax.spines['left'].set_bounds(Tmin,1)
        ax.set_xticks([nu0 ])
        ax.set_yticks([Tmin, 1])
        ax.set_xticklabels([r'$0$'])
        ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        ax.spines["bottom"].set_visible(True)
        
        
    ax = axArr[0,0]
    ax.set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal)
    ax.yaxis.set_label_coords(-0.14, 1.1)
    axArr[1,0].set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal
    axArr[1,0].yaxis.set_label_coords(-0.14, 1.1)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax = axArr[0,1]
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelleft=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax = axArr[1,1]
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelleft=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
        
    fig.subplots_adjust(wspace=0.01)
    fig.subplots_adjust(hspace=0.35)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "TransmissionCartoon.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plotChangeInTransmissionCartoon2(saveBool = False):
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*1.1, 0.18*PRQuantumWidthFull, 
       [r"$\Delta \omega_{\text{opt}}$",r"$\Delta \omega_{\text{MW}}$",r"$\Delta \omega_{\text{opt}}$",r"$\Delta \omega_{\text{MW}}$"], 
       r"$T$ (au)",
       labelPadX = 5, labelPadY = 3, commonY = True, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow =1, ncol = 4, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False
    )
    lw = 1
    

    lamb0 = 1040.77e-9
    nu0 = c_const/lamb0
    FWHM = 30e9
    bw = FWHM*2
    nu = np.linspace(nu0 - bw, nu0+bw, 1000)
    
    f0 = 2.8e9
    fbw = hyperfine*4
    f = np.linspace(f0 - fbw, f0+fbw, 1000)
    gammaMW = 0.5e6
    ax = axArr[0]
    
    g = 0
    kex1 = FWHM/4
    kex2 = kex1
    ki1 = kex1*2
    ki2 = ki1*3
    Delta = nu - nu0 
    print(nu0/ki1)
    T= calcTransmission(kex1, kex2, ki1, ki1, Delta, g = 0, tau = 1)
    T2= calcTransmission(kex1, kex2, ki2, ki2, Delta, g = 0, tau = 1)
    T3 = microwaveDetuning1(f, gammaMW*3, f0, (np.min(T2) - np.min(T)), np.min(T2))/1.15
    
    ax.plot(nu, T, lw = 0.2/0.358, color = colorLightRed)
    ax.plot(nu, T2, lw = 0.2/0.358, color = colorDarkRed)
    axArr[1].plot(nu, T3-np.min(T3)+ np.min(T), lw = 0.2/0.358, color = colorBlue)
    
    ax = axArr[2]
    g = 0
    kex1 = FWHM/4
    kex2 = kex1
    ki1 = kex1*4
    ki2 = ki1*1.15
    Delta = nu - nu0 
    print(nu0/ki1)
    T= calcTransmission(kex1, kex2, ki1, ki1, Delta, g = 0, tau = 1)
    T2= calcTransmission(kex1, kex2, ki2, ki2, Delta, g = 0, tau = 1)
    T3 = microwaveDetuning1(f, gammaMW*3, f0, (np.min(T2) - np.min(T)), np.min(T2))/1.15
    
    ax.plot(nu, T, lw = 0.2/0.358, color = colorLightRed)
    ax.plot(nu, T2, lw = 0.2/0.358, color = colorDarkRed)
    axArr[3].plot(nu, T3-np.min(T3)+ np.min(T), lw = 0.2/0.358, color = colorBlue)
    
    
    
    Tmin = 0
    
    for ax in axArr:
        ax.set_xlim([nu0 - bw*1.15,nu0 + bw])
        ax.spines['bottom'].set_bounds(nu0 - bw, nu0 + bw)
        
        ax.set_ylim([Tmin-0.05,1])
        ax.spines['left'].set_bounds(Tmin,1)
        ax.set_xticks([nu0 ])
        ax.set_yticks([Tmin, 1])
        ax.set_xticklabels([r'$0$'])
        ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        # ax.spines["bottom"].set_visible(True)
    
    
        
    axArr[0].set_ylabel(axArr[0].get_ylabel(), rotation=0)   # horizontal
    axArr[0].yaxis.set_label_coords(0.08, 1.1)
    axArr[0].set_xlim([nu0 - bw*1.3,nu0 + bw])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    for ax in axArr[[1,2,3]]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelleft=False)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(labelbottom=True)
        ax.tick_params(axis='x', which='both', bottom=True, top=False)
    
        
    fig.subplots_adjust(wspace=0.1)
    fig.subplots_adjust(hspace=0.35)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "TransmissionCartoon2.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

#%%% Figure 5
def plotNarrowTransmissionPaper(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.2, 0.2*PRQuantumWidthFull, 
       r"$\lambda$ (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 1, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False, tickpad = 4
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.2um_0/20260202/1040_1042nm_1nmps_averaged_norm.mat'
    
    
    data = loadmat(filePath, squeeze_me=True)
    # dataParams = loadmat(filePathParams, squeeze_me=True)
    
    W = data['W']
    TOut = data['TOut']
    
    # params = dataParams['params']
    lambmin = 1040.2
    lambmax = 1041.4
    filt = np.logical_and(W< lambmax, W>lambmin)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]
    ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = 3, zorder = 10, color = colorDarkRed)
    
    plt.tight_layout()
    (Tmin, Tmax) = ax.get_ylim()
    
    lamb = np.linspace(lambmin, lambmax, 1000)*1e-9
    nu = c_const/lamb
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0
    
    g = 0
    tau =1.004524991429497*np.exp(1j* -0.057891967231220)
    kex1 = nu0/2.509576482281477e+04
    kex2 = kex1
    ki1 = nu0/6.763434893840459e+03
    ki2 = ki1
    
    Delta = nu - nu0 
    print(nu0/ki1)
    fit= calcTransmission(kex1, kex2, ki1, ki2, Delta, g = 0, tau = tau)
    fits= calcTransmissionSinglet(kex1, ki1, Delta, tau = tau)
    
    
    
    ax.plot(lamb*1e9, fit, lw = 0.2/0.3528, color = colorRed)
    # ax.plot(lamb*1e9, fits, lw = 0.2/0.3528, color = 'k')
    Tmin = 0.3
    ax.set_xlim([lambmin,lambmax])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    ax.set_ylim([Tmin,1])
    ax.spines['left'].set_bounds(Tmin,1)
    ax.set_xticks([lamb0*1e9])
    ax.set_yticks([Tmin, 1])
    ax.set_xticklabels([rf"${val:.1f}$" for val in ax.get_xticks()])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
    
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=True, bottom=False, labeltop = True, labelbottom = False)
    ax.spines["bottom"].set_visible(True)
    # ax.yaxis.set_label_coords(-0.3, 0.55)
    # ax.xaxis.set_label_coords(0.58, -0.25)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NarrowTransmission_4p2um_paper.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plotZeroFieldODMR(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.6, 0.3*PRQuantumWidthFull, 
       r"$\omega_{\text{MW}}/2\pi$ (MHz)", 
       r'$\Delta T$ ($\times10^{-4}$)',
       labelPadX = 4, labelPadY = 6, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False, tickpad = 4
    )
    lw = 1

    filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260127/ODMR_20260127_CW__16_18dBAtt_4.2um_0_1_2.87GHz_30MHz_003.mat'
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    # lambmin, lambmax = 1040.75, 1040.95
    W_nm = np.squeeze(data['laserWLsRead'].item())
    W_nm = W_nm[0,0,:,:]
    S_V = np.squeeze(data['signal'].item())
    C = np.squeeze(data['contrast'].item())
    DC_V = S_V/C
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    deltaT = S_V/np.max(DC_V)

    f = freqsODMR*1e-6-2870
    
    # cmap = plt.cm.viridis
    # n = np.shape(signalAbs)[1]
    # for i in range(n):
    #     deltaT = np.mean(-contrast[:, i, 480:520], 1)
    #     Tmax = np.mean(T[:, i, :], 0)
        
    #     color = cmap(i / (n - 1))   # normalize index 0→1
    #     ax.plot(f, deltaT, color=color, lw=0.5)
        # ax.plot(WAbs[1,:], Tmax, color=color, lw=0.5)
    
    # plt.show()
    idx = 9
    signalXZero = -np.mean(deltaT[:, idx, 400:450], 1)*10000
    power = np.mean(power[:, idx], 0)
    print(power)
    ax.plot(f, signalXZero, color=colorBlue, lw=0.2/0.3528)
    ax.scatter(f, signalXZero, color=colorDarkBlue, marker = '.', s = 4, edgecolors='none', zorder = 10)
    
    
    filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75, 1040.95
    WAbs = data['laserWLsRead'].item()
    WAbs = WAbs[:,:]
    filt = np.logical_and(WAbs< lambmax, WAbs>lambmin)

    WAbs= WAbs[filt]
    signalAbs = data['signal'].item()
    contrast = data['contrast'].item()
    
    
    S_V = data['signal'].item()
    C = data['contrast'].item()
    DC_V = S_V/C
    deltaT = S_V/np.max(DC_V)*1e4

    
    
    f = data['freqsODMR'].item()*1e-6-2776.76-30
    
    X, Y = np.meshgrid(WAbs, f)
    
    # idx = -3
    signalX = -np.mean(deltaT[:, 460:540], 1)
    
        
    ax.scatter(f, signalX, color=colorDarkBlue, marker = '.', s = 4, edgecolors='none', zorder = 10)
    
    p0 = [nuOpt*1.05, -29.9e6, 0.145*10, 0]
    popt, pcov = curve_fit(microwaveDetuning3, f*1e6, signalX, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    signal = microwaveDetuning3(f*1e6, *popt)
    
    f = np.linspace(-40, -20, 1000)
    signal = microwaveDetuning3(f*1e6, nuOpt*1.05, -29.9e6, 0.145*10, 0)
    
    print('Fitted linewidth MHz: ', popt[0]/1e6, ' pm ', perr[0]/1e6)
    
    
    ax.plot(f, signal, color=colorBlue, lw=0.2/0.3528)
    
    # color = cmap(i / (n - 1))   # normalize index 0→1
    
    
    fmin = -40
    fmax = 15
    Tmin = 0
    Tmax = np.max(signalXZero)
    # ax.plot(lamb*1e9, fit, lw = 0.2/0.3528, color = colorRed)
    # # ax.plot(lamb*1e9, fits, lw = 0.2/0.3528, color = 'k')
    # Tmin = 0.3
    ax.set_xlim([fmin-5,fmax])
    ax.spines['bottom'].set_bounds(fmin,fmax)
    
    ax.set_ylim([Tmin-0.3,Tmax])
    ax.spines['left'].set_bounds(Tmin,Tmax)
    ax.set_xticks([-29.9,0])
    ax.set_yticks([Tmin, np.max(signalX), Tmax])
    ax.set_xticklabels([rf"${val:.0f}$" for val in [2777, 2870]])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
    # ax.yaxis.set_label_coords(-0.3, 0.55)
    # ax.xaxis.set_label_coords(0.58, -0.25)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "ZeroFieldODMR_4p2um.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plotWideTransmissionAndSignal(saveBool = False):
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*0.45, 0.3*PRQuantumWidthFull, 
       r"$\lambda$ (nm)", 
       [r'$T$ (au)',r'$\Delta T$ ($\times 10^{-4}$)'],
       labelPadX = 4, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =True, widthRatios = [1], heightRatios = [1,2] 
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.2um_0/20260202/1030_1070nm_5nmps_averaged_norm.mat'
    filePathAbs =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260129/ODMR_20260129_CW__16_12dBAtt_4.2um_0_1__2.7768GHz_0MHz_001.mat'
    data = loadmat(filePath, squeeze_me=True)
    dataAbs = loadmat(filePathAbs, squeeze_me=True)
    
    W = data['W']
    TOut = data['TOut']-0.03
    TOut /= TOut.max()
    
    
    lambmin = 1040
    lambmax = 1041.5
    
    filt = np.logical_and(W< lambmax, W>lambmin)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]
    # TOut /= -0.007*(W-1041.2)**2+1
    skip = 1
    # axArr[0].plot(W[1::skip], TOut[1::skip], lw = 0.15/0.3528, color = colorDarkRed)
    axArr[0].scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkRed)
    
    dataAbs = dataAbs['data']
    W_nm = dataAbs['laserWLsRead'].item()*1e9
    S_V = dataAbs['signal'].item()*2 # peak amplitude to p2p
    DC_V = dataAbs['contrast'].item() 
    C = S_V/DC_V;   
    deltaT = S_V/np.max(DC_V);
    power = dataAbs['powerIn'].item()
    
    
    print(power)
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
    skip = 1
    W_nm= W_nm[filt]
    S_V= S_V[filt]
    deltaT = deltaT[filt]
    power = power[filt]
    
    contrast = -deltaT/power*np.max(power)*1e4
    
    # axArr[1].plot(W_nm, contrast, lw = 0.15/0.3528, color = colorDarkBlue)
    axArr[1].scatter(W_nm-0.07, contrast, edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkBlue)
    
    
    P_in = 20e-6 * np.sqrt(etaFibreIR) * np.array([1, 1, 1,1,1,1])
    lambModes = np.array([1040.74, 1042.98, 1047.12, 1048.62, 1057.6, 1061.29])
    bwGHzArr = [150,150,100,100,200,100]
    
    Veff = calcVeff()
    g = np.array([5e9, 5.8e9, 20e9, 1e9, 0 ,0])
    t = np.array([0.98,0.947,0.69,0.9,1,0.93])
    phi = np.array([-0.057,0.04,-0.04,-0.4,0,0.3])
    tau =t*np.exp(1j* phi)
    # kex1 = nuModes/np.array([2.6e4])
    kex1 = np.array([1.1e10,48e9,3.5e9,1.5e9,1.1e10,1.2e9])
    kex2 = kex1
    kex2[1] = 36e9
    ki1 = np.array([4e10,15e9])
    ki2 = ki1
    ki2[1] = 13.6e9
    
        
    numin = -300
    numax = 300
    lambArr = np.linspace(1040e-9, 1041.5e-9, 501)
    
    
    # scale = np.array([1,1.6,1,1,1,0.05])*0 + 1e4
    gradScale = np.array([1,1,1,10,1,2])
    # deltaOff = np.array([0,8,-5,1,1,-5])
    P_in[1] = 5e-7
    
    yTop = lambArr * 0
    yBot = lambArr * 0
    for i, lamb in enumerate(lambModes):
        # popt = [kex1[i], kex2[i], ki1[i], ki2[i], g[i], t[i], phi[i],P_in.mean(), 1]     
        if i == 0:
           popt= [ 1.17471071e+10,  1.13956599e+10,  4.21688689e+10,  4.20331073e+10,
              4.59152053e+09 , 8.86694080e-01, -5.64339617e-02 , 8.90467804e-06,
              1]
        else:
           popt= [4.79640598e+10, 3.63207302e+10, 1.50710268e+10, 1.36458848e+10,
            5.84276809e+09 ,9.47568803e-01, 3.98692216e-02, 5.72327012e-06,
            1
            ]
        print('Qcav: ', c_const/lamb/1e-9/popt[2])
                
    
        deltaOmegaGHz = (c_const/lambArr-c_const/lamb/1e-9)*1e-9
        DeltaTFit, TMean,kabs0, kabs1, alphaOff, alphaOn= calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *popt, printBool = True, Veff =Veff)
        TMean/= TMean.max()
        P_det = P_in.mean()*TMean*np.sqrt(etaFibreIR)*etaPostFibre
        
        gradient = np.gradient(TMean*t[i], deltaOmegaGHz)
        yTop += (TMean-1)*t[i]
        yBot += DeltaTFit*1e4+ gradScale[i]*gradient
        
    axArr[0].plot(lambArr/1e-9, (yTop+1)*0.98, lw = 0.2/0.3528, color = colorLightRed)
    
    # axx[i,0].plot(deltaOmegaGHz, gradient, lw = 0.2/0.3528, color = colorRed)

    axArr[1].plot(lambArr/1e-9, yBot, lw = 0.2/0.3528, color = colorBlue)
        # print('Scale: ', popt[-1])
    
    
    plt.tight_layout()
    
    for i,ax in enumerate(axArr):
        ax.set_xlim([lambmin-0.2,lambmax])
        ax.spines['bottom'].set_bounds(lambmin, lambmax)
        
        if i == 0:
            ax.set_ylim([0.25,1.01])
            ax.spines['left'].set_bounds(0.3,1)
            ax.set_xticks([lambmin, lambmax])
            ax.set_yticks([0.3,1])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False, bottom=False)
        else:
            ax.set_ylim([-1.4*2,1.9*2])
            ax.spines['left'].set_bounds(-1.1*2, 3.2)
            ax.set_yticks([-1.1, 0, 1.8])
            
            ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
            ax.tick_params(axis='x', which='both', top=False, bottom=True)
            
        ax.set_xticklabels([rf"${val:.1f}$" for val in [lambmin, lambmax]])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        
    axArr[1].yaxis.labelpad = 2
        
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "WideTransmissionAndAbsSignal_4p2um.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

#%%% Figure 6
def plotODMRFittingCurves(saveBool = False):
    
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*0.6, PRQuantumWidthHalf*0.6, 
       r"$\Delta\omega_{\text{MW}}/2\pi$ (MHz)", 
       r"lock-in signal, $S_{\text{LI}}$ ($\upmu$V)" #"coincidence rate (mHz/mW)"
       , labelPadX = 5, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 1,nrow =1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260201/';
    filename = 'ODMR_20260201_CW__16_12dBAtt_4.2um_0_1__2.7767GHz_16MHz_001.mat';

    data = loadmat(directory + filename, squeeze_me=True)
    data = data['data']
    
    W_nm = np.squeeze(data['laserWLsRead'].item())
    
    S_V = data['signal'].item()
    C = data['contrast'].item()
    DC_V = S_V/C
    DeltaT = S_V/np.max(DC_V)
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-2776.6
    fCont = np.linspace(-10, 10, 1000)
    p0 = [2e6, -1e-5, -7,0]
    p0 = [1e6, 1e5, -10, 1e-7]
    
    S_fit = microwaveDetuningDeriv3(fCont*1e6, *p0)
    
    DC_V_1D = np.mean(DC_V[:, 490:495], 1)
    S_V_1D = np.mean(S_V[:, 490:495], 1)
    
    sigma_laser = DC_V_1D.mean()*np.sqrt(1e-12*1000/2)
    print('sigma_laser: ', sigma_laser)
    
    print('power in: ', power.mean())
    print('power det: ', power.mean()*DC_V_1D.mean()*etaFibreIR*etaPostFibre)
    
    popt, pcov = curve_fit(microwaveDetuningDeriv3, f*1e6, S_V_1D, p0=p0)
    
    offset = popt[1]
    yoffset = popt[-1]
    f = f - offset/1e6
    S_V_1D = S_V_1D + yoffset*2
    
    popt, pcov = curve_fit(microwaveDetuningDeriv3, f*1e6, S_V_1D, p0=p0)
    perr = np.sqrt(np.diag(pcov))  # standard deviations of parameters
    
    S_fit = microwaveDetuningDeriv3(fCont*1e6, *popt)*1e6 # V * 1e6 uV/V
    S_fitDeriv_uVperMHz = microwaveDetuningDDeriv3(fCont*1e6, *popt) * 1e6 * 1e6 # V/Hz *1e6 uV/V *1e6 Hz/MHz
    S_fitDeriv_uVpermT = S_fitDeriv_uVperMHz*1e3*28*1e-3 # uV/MHz * 1e3 MHz/GHz * 28 GHz/T * 1e-3 T/mT
    
    dSdwuVpMHz = microwaveDetuningDDeriv3(0,*popt)*1e6 * 1e6
    dSdwuVpMHzDiff = microwaveDetuningDDeriv3(0,*popt)*1e6 * 1e6 - microwaveDetuningDDeriv3(0,*(popt+perr))*1e6 * 1e6
    dwdBGHzpT = gamma/1e9
    print(dSdwuVpMHz) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    print(dSdwuVpMHzDiff) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    print(dSdwuVpMHz * dwdBGHzpT*1e3/1e6) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    print(dSdwuVpMHzDiff * dwdBGHzpT*1e3/1e6) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    
    
    axArr.scatter(f, S_V_1D*1e6, edgecolors='none', marker = '.', s = 8, zorder = 10, color = colorDarkRed)
    # axArr.scatter(f, DC_V_1D, edgecolors='none', marker = '.', s = 8, zorder = 10, color = colorDarkRed)

    fLin = np.linspace(-1,1,2)*0.35
    
    axArr.plot(fCont, S_fit, color=colorRed, lw=0.2/0.3528)
    axArr.plot(fLin, -fLin*11.4, linestyle = '--',color=colorBlue, lw=0.2/0.3528)
    
    # axArr[1].plot(fCont, S_fitDeriv_uVpermT, color=colorRed, lw=0.2/0.3528)
    ax = axArr
    # for ax in axArr:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax.set_xlim([-11,10])
    ax.spines['bottom'].set_bounds(-10,10)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    # ax = axArr[0]
    
    ax.spines['left'].set_bounds(np.min(S_fit),np.max(S_fit))
    ax.set_yticks([np.min(S_fit), 0, np.max(S_fit)])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
   
    # ax = axArr[1]
    
    # ax.spines['left'].set_bounds(np.min(S_fitDeriv_uVpermT),np.max(S_fitDeriv_uVpermT))
    # ax.set_yticks([np.min(S_fitDeriv_uVpermT), 0, np.max(S_fitDeriv_uVpermT)])
    # ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
    # # ax.set_ylim([-0.2,None])
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "signalConversionCAFQA" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    

    
    
def plotNoiseFigure(saveBool = False):
    
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*1.5, PRQuantumWidthHalf*0.6, 
       [r"frequency, $f$ (Hz)", r"averaging time, $t$ (s)"], 
       [r"magnetic noise (nT/$\sqrt{\text{Hz}}$)",r"Allan deviation, $\sigma_a$ (nT)"] #"coincidence rate (mHz/mW)"
       , labelPadX = 5, labelPadY = 5, commonY = False, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 2,nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize-0.5,
       widthBool = True, widthRatios = [1,0.8], heightRatios = [1]
    )
    
    directoryCW = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260201/'
    filenameCW =  '/TimeSignal_20260201_CW__16_12dBAtt_4.2um_0_1__2.7767GHz_4MHz_007_timeSignal.mat';

    # load data
    mat = loadmat(directoryCW + filenameCW)
    S_V = np.squeeze(mat['signal'])
    S_V -= np.mean(S_V)
    
    time = np.squeeze(mat['time'])
    tau = time[1]-time[0]
    fss = 1/tau
    
    def sig2BCW(signal):
        return signal*1e6/0.318 # V * 1e6 uV/V / 0.318 uV/uT * 1e3 nT/uT
    
    
    B_uT = sig2BCW(S_V)
    xlimAD = 1e3
        
    N = len(B_uT)
    
    # fft, f = fft(time, fss, N)
    f, psd = welch(B_uT, fs=fss, nperseg=2**20)
    fAvg, psdAvg = welch(B_uT, fs=fss, nperseg=2**16)
    
    magNoiseuT = np.sqrt(psd)
    magNoiseAvguT = np.sqrt(psdAvg)
    filt = (f>=1e-2) & (f<=1e3)
    f = f[filt]
    magNoisenT = magNoiseuT[filt]*1000
    filt = (fAvg>=1e-2) & (fAvg<=1e3)
    fAvg = fAvg[filt]
    magNoiseAvgnT = magNoiseAvguT[filt]*1000
    
    sensitivityPSD = np.mean(magNoisenT[f < 1e1])
    print('PSD extracted sensitivity: ', sensitivityPSD, ' nT')
    
    filtAvg = (f>=1e-2) & (f<=1e3)
    
    m_values = np.arange(1, N // 2 - 10)  # All possible m integers up to N/2 - 10
    m_values = np.logspace(np.log10(np.min(m_values)), np.log10(np.max(m_values)), 101)
    avar = np.zeros_like(m_values, dtype=float)
    taum = m_values / fss
    
    # Compute Allan variance
    (t2, ad, ade, adn) = oadev(B_uT, rate=fss, data_type="freq", taus=taum)

    filt = (t2>=1e-2) & (t2<=xlimAD)
    t2 = t2[filt]
    ad = ad[filt]

    def model_func(logt, logsens):
        return -1/2*logt + logsens
    
    def allan_func(t, A, B, C):
        return np.log10((A**2)/t + (B**2) + (C**2)*t)
    
    p0 = [1e-1, 0, 10e-4]
    popt, pcov = curve_fit(allan_func, t2, np.log10(ad**2), p0=p0)
    
    # print(popt)
    sensitivityAD = popt[0]
    print('Allan deviation extracted sensitivity: ', sensitivityAD*np.sqrt(2)*1000, ' nT')
    
    tCont = np.logspace(-2, 3, 1000)
    adFit = allan_func(tCont, *popt)
    adFit = np.sqrt(10**adFit)
    
    ad *= 1000
    adFit *= 1000
  
    lw = 0.2/0.3528
    axArr[0].plot(f, magNoisenT, linestyle = '-', color=colorDarkRed, lw = lw)
    axArr[0].plot(fAvg, magNoiseAvgnT, linestyle = '-', color=colorRed, lw = lw)
    axArr[0].plot([1e-3, 1e3], [sensitivityPSD, sensitivityPSD], linestyle = '--', color='k', lw = lw, zorder = 11)
    axArr[0].set_xscale('log')
    axArr[0].set_yscale('log')
    
    # axArr[1].plot(tCont , adFit, linestyle = '-', color=colorDarkRed, lw = lw)
    axArr[1].plot(np.logspace(-3, 3, 1000) , 1/np.sqrt(np.logspace(-3, 3, 1000))*50, linestyle = '-', color='k', lw = lw)
    axArr[1].scatter(t2 , ad, marker = '.', color=colorRed, s = 12,zorder = 11, edgecolors='none')
    axArr[1].set_xscale('log')
    axArr[1].set_yscale('log')
    
    for ax in axArr:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
   
    plt.tight_layout()
    
    ax = axArr[0]
    
    ax.spines['bottom'].set_bounds(1e-2,5e2)
    ax.set_xlim([np.min(f)*0.65,np.max(f)])
    xticksMinor = ax.get_xticks(minor = True)
    filt = np.logical_and(~(xticksMinor < 1e-2), ~(xticksMinor > 5e2))
    ax.set_xticks(xticksMinor[(filt)], minor = True)
    
    ax.spines['left'].set_bounds(5e1,1e3)
    ax.set_ylim([4.3e1,1e3])
    
    yticks = ax.get_yticks()
    
    filt = np.logical_and((yticks <= 1e3), (yticks >= 5e1))
    
    ax.set_yticks(np.concatenate([yticks[filt],[sensitivityPSD]]), minor = False)
    yticks = ax.get_yticks()
    # print(yticks)
    ax.set_yticklabels([r'$100$', r'$1000$', rf'${sensitivityPSD:.0f}$'])
    
    
    
    ax = axArr[1]
    
    ax.spines['bottom'].set_bounds(np.min(tCont),np.max(tCont))
    ax.set_xlim([np.min(tCont)*0.7,np.max(tCont)])
    xticksMinor = ax.get_xticks(minor = True)
    filt = np.logical_and(~(xticksMinor < 1e-2), ~(xticksMinor > xlimAD))
    ax.set_xticks(xticksMinor[(filt)], minor = True)
    
    ax.spines['left'].set_bounds(np.min(ad),np.max(ad))
    ax.set_ylim([np.min(ad)*0.8,np.max(ad)*1.01])
    yticks = ax.get_yticks()
    
    filt = np.logical_and(~(yticks < np.min(ad)), ~(yticks > np.max(ad)))
    ax.set_yticks(np.concatenate([yticks[filt],np.array([np.min(ad),np.max(ad) ])]), minor = False)
    yticks = ax.get_yticks()
    # print(yticks)
    ax.set_yticklabels([r'$100$', r'$1000$', r'$12.8$', r''])
    
    # ax.set_yticks([np.min(S_fitDeriv_uVpermT), 0, np.max(S_fitDeriv_uVpermT)])
    # ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
    # ax.set_ylim([-0.2,None])
    
    fig.subplots_adjust(wspace=0.4)
    fig.subplots_adjust(hspace=0.35)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "noiseMeasurement" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

#%%% Figure 7
def plotDetuningExample(saveBool = False):
    fig, axx = plot_custom(
       PRQuantumWidthFull*1/4, PRQuantumWidthHalf*1, 
       [ r'',r"$\Delta\omega_{\text{opt}}/2\pi$ (GHz)", r"$\Delta \omega_{\text{MW}}/2\pi$ (MHz)"], 
       [r'',r'', r'$\Delta T$ ($\times 10^{-4}$)'],
       labelPadX = 3, labelPadY = 1, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 3, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =True, widthRatios=[1], heightRatios=[0.5,0.5,1]
    )
    
    
    filePath = filepathTransmission + filenameTransmission4p2um
    
    #%%%% Plot transmission data 
    W_nm, DeltanuGHz, TOut = loadAndFilterTransmissionData(filePath)
    skip = 1
    axx[0].scatter(DeltanuGHz[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkRed)
    
    filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    
    lamb0 = 1040.85 * 1e-9
    nu0 = c_const/lamb0
    nuBW = 200e9 
    f0GHz = 2776.7
    
    #%%%% Plot ODMR line data 
    W_nm, S_V, DeltaT, C, P_in, DeltanuGHz, DeltaW_nm, fExp, DC_V = loadODMRData(filePath, lamb0nm = lamb0 * 1e9, 
                     lambbwnm = 2, f0GHz = f0GHz, nuOptbwGHz = nuBW*1e-9, fbwGHz = 10, 
                     powerScaling = 99/2, boolFreqFiltering = True, idx = 2, bgOffset = 3.5e-3)
    DC_VMean = np.mean(DC_V, 1)
    DC_VMean /= DC_VMean.max()
    
    DeltaT_Smooth = gaussian_filter1d(DeltaT, sigma=20, axis=0)
    
    idx = 54
    scale = 1000
    skip = 20
    x = (c_const/W_nm/1e-9 - nu0)*1e-9
    y = DeltaT_Smooth[:, idx]*2 # peak to p2p
    axx[1].scatter(x[1::skip], y[1::skip]*scale, edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkBlue)
    DeltaTMaxExp = y.max()*scale
    
    (Tmin, Tmax) = axx[0].get_ylim()

    #%%%% Plot transmission fit
    
    numin = nu0 - nuBW
    numax = nu0 + nuBW
    nuArr = np.linspace(numin, numax, 101)
    
    Veff = calcVeff()
    g = 5e9
    tau =1.004524991429497*np.exp(1j* -0.057891967231220)
    kex1 = nu0/2.6e4
    kex2 = kex1
    ki1 = nu0/7.6e3
    ki2 = ki1
    
    Delta = nuArr - nu0 
    
    deltaOmegaGHz = Delta*1e-9
    deltaOmegaGHzMin = deltaOmegaGHz.min()
    deltaOmegaGHzMax = deltaOmegaGHz.max()
    
    #%%%% Plot change in transmission fit with wavelength
    P_in *= np.sqrt(etaFibreIR)*0.8
    p0 = [kex1, kex2, ki1, ki2, g, np.abs(tau), np.angle(tau),P_in.mean()*1.36/1.01, 1]
    DeltaTFit, TMean = calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *p0, Veff =Veff)
    popt=p0    
    print('popt: ', popt)
    
    # DeltaTFit, TMean,kabs0, kabs1= calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *popt, printBool = True, Veff =Veff)
    TMean/= TMean.max()
    P_det = P_in.mean()*TMean*np.sqrt(etaFibreIR)*etaPostFibre
    
    a = np.abs(calcIntracavityField(popt[0], popt[0] + popt[2], popt[1], popt[1]+ popt[3], Delta, popt[4]))
    NcavMax = calcPhotonsInCavity(a.max(), P_in.mean())
    print('Pin: ', P_in.mean())
    print('Max Ncav: ', NcavMax)
    Imax = NtoI1042(NcavMax)
    print('Max intensity MW/cm^2: ', Imax)
    
    # kabs0 = calcEnergyLossRate(nd, alpha0)
    # kabs1 = calcEnergyLossRate(nd, alpha1)
    skip = 106
    Deltak = DeltaT2Deltak(y[1::skip], np.abs(tau), kex1, kex1 + ki1)
    print('Deltak MHz: ', Deltak.max()/1e6)
    print('delta alpha /m: ', kappa2alpha(Deltak.max()*2*np.pi))
    
    DeltaT0, T0, kabs0, kabs1, alphaOff, alphaOn =calcDeltaMicrodiskReturnT(np.array([0]), *popt, printBool = True, Veff =Veff)
    
    print('Deltak model MHz:', (kabs0- kabs1)/-1e6 )
    print('Delta alpha model /m: ', -kappa2alpha((kabs0- kabs1)*2*np.pi))
    print('Delta alpha model /m: ', alphaOn- alphaOff)
    print('alpha off model /m: ', kappa2alpha((kabs0)*2*np.pi))
    
    [kex1, kex2, ki1, ki2, g, tauabs, phi, pin, scaleNew] = popt
    tau = 1*np.exp(1j*phi)
    
    
    #%%%% Predict transmission based on change in transmission
    axx[0].plot(deltaOmegaGHz, TMean, lw = 0.2/0.3528, color = colorRed)
    
    
    #%%%% Plot change in transmission fit with MW det
    fBW = 5e6
    fmin = -fBW
    fmax = fBW
    f = np.linspace(fmin, fmax, 1000)
    amp = DeltaTMaxExp/scale * 1e4*0.75
    TMW = microwaveDetuning3(f, nuOpt, 0, amp, 0)
    TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp/2.8)
    
    
    #%%%% Plot change in transmission with microwave detuning
    idx = 500
    N = 30
    u = fExp-0.1
    v = np.mean(DeltaT_Smooth[idx-N:idx+N, :],0)*1e4*2 # peak to p2p
    skip = 2
    axx[2].scatter(u[1::skip], v[1::skip], edgecolors='none', marker = '.', s =5, zorder = 10, color = colorDarkBlue)
           
    p0 = [1e6, 0, 1, 0]
    lower_bounds = [1e5, -1e6, 1e-2, -1]
    upper_bounds = [2e6, 1e6, 1e1, 1]
    
    popt, pcov = curve_fit(microwaveDetuning3, u*1e6, v, p0=p0,    bounds=(lower_bounds, upper_bounds))
    perr = np.sqrt(np.diag(pcov))
    print('fitted ODMR linewidth (MHz):', popt[0]/1e6 , ' pm ', perr[0]/1e6)
    
    TMax = microwaveDetuning3(0, popt[0], 0, popt[2], popt[3])
    TMin = microwaveDetuning3(hyperfine/2*1.1, popt[0], 0, popt[2], popt[3])
    C = (TMax-TMin)*1e-4/TMean.min()
    dC = C*perr[2]/popt[2]
    print('C:', C , ' pm ', dC)
    print('Tmax/Tmin:', Tmax/Tmin)
    print('Delta T:', (TMax-TMin)*1e-4)
        
    
    etaSNL = calcSensitivityCW(popt[0], C, P_det.min(), lamb_s)

    detaSNL = calcSensitivityCWUncertainty(popt[0], perr[0], C, dC, P_det.min(), P_det.min()*0.2, lamb_s)
    
    print('photon flux /1e10:', P_det.min()/(hbar*omegas)/1e10)
    print('eta nT:', etaSNL*1e9, ' pm ', detaSNL*1e9)
    
    axx[1].plot(deltaOmegaGHz, DeltaTFit*scale, lw = 0.2/0.3528, color = colorBlue)
    
    #%%%% Plot change in transmission fit with MW det 2
    TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp)
    axx[2].plot(f/1e6, TMWsingle1, lw = 0.2/0.3528, color = 'k', linestyle = '--')
    TMW = microwaveDetuning3(f, nuOpt, 0, amp, 4.02066788e-01)
    
    axx[2].plot(f/1e6, TMW, lw = 0.2/0.3528, color = colorBlue)
    
    #%%%% Set plot settings
    ax = axx[0]
    plt.tight_layout()
    (Tmin, Tmax) = axx[0].get_ylim()
    ax.set_xlim([deltaOmegaGHzMin-40,deltaOmegaGHzMax])
    ax.spines['bottom'].set_bounds(deltaOmegaGHzMin, deltaOmegaGHzMax)
    
    ax.set_ylim([Tmin,1.02])
    ax.spines['left'].set_bounds(Tmin, 1)
    ax.spines["bottom"].set_visible(False)
    # ax.set_xticks([-200, 200])
    ax.set_yticks([Tmin,1])
    ax.set_yticklabels([rf"${val:.2f}$" for val in ax.get_yticks()])
    ax.yaxis.set_label_coords(-0.15, 1.1)
    ax.set_xticks([])
    
    ax = axx[1]
    plt.tight_layout()
    (Tmin, Tmax) = axx[1].get_ylim()
    Tmax = np.max(y[1::skip]*scale)
    ax.set_xlim([deltaOmegaGHzMin-40,deltaOmegaGHzMax])
    ax.spines['bottom'].set_bounds(deltaOmegaGHzMin, deltaOmegaGHzMax)
    
    
    ax.set_ylim([Tmin-0.1,Tmax+0.04])
    ax.spines['left'].set_bounds(Tmin, Tmax)
    # ax.spines["bottom"].set_visible(False)
    ax.set_yticks([Tmin,Tmax])
    ax.set_yticklabels([rf"${val:.2f}$" for val in ax.get_yticks()])
    ax.yaxis.set_label_coords(-0.15, 1.1)
    ax.set_xticks([-200, 200])
    
    ax = axx[2]
    plt.tight_layout()
    # (Tmin, Tmax) = axx[2].get_ylim()
       
    ax.yaxis.set_label_coords(-0.18, 1.1)
    
    ax.set_xlim([fmin/1e6-0.7,fmax/1e6])
    ax.spines['bottom'].set_bounds(fmin/1e6,fmax/1e6)
    Tmax *= 10
    ax.set_ylim([-0.3, Tmax+0.2])
    ax.spines['left'].set_bounds(0,Tmax)
    ax.set_yticks([0, 1, Tmax])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
    
    for ax in axx:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal
        ax.xaxis.set_label_coords(0.55,-0.27)
        
    axx[0].yaxis.set_label_coords(-0.16, 1.05)
    
    
    axx[0].text(-220, 1.11, r"$T$ (au)", color=colorRed,
         ha='right', fontsize = labelfontsize-1)
    axx[0].text(-220, 1.29, r"$\Delta T$ (au)", color=colorBlue,
         ha='right', fontsize = labelfontsize-1)

    
    axx[2].yaxis.set_label_coords(-0.04, 1.07)
    axx[1].xaxis.set_label_coords(0.55, -0.45)
    
    
    fig.subplots_adjust(hspace=0.9)
    
    TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp)
    TMW = microwaveDetuning3(f, nuOpt, 0, amp, 0)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "parameterDefinitions.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
#%%% Figure 7 Contd
    
def plotMicrodiskTheorySensitivityDeltaTvsMWDet(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*1/3, PRQuantumWidthHalf * 1, 
       r'$\Delta \omega_{\text{MW}}/2\pi$ (MHz)', 
       [r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)', r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)'],
       labelPadX = 5, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =True,  widthRatios = [1], heightRatios = [1,2]
    )
    
    # define cavity parameters
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    
    
    # define optical frequency detuning array
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0 - 22e9
    lamb0 = c_const/nu0
    
    N = 50
    nuBW = 50e9 #nm
    nu = np.linspace(nu0 - nuBW, nu0 + nuBW, N*2)
    lamb = c_const/nu
    
    # define MW frequency detuning array
    fMW0 = 2776.6e6*0
    fMWBW = 5e6
    fMWmax = -fMWBW
    fMWmin = fMWBW
    fMW = np.linspace(fMWmin, fMWmax, N)
    
    
    ODMR = microwaveDetuning3(fMW, nuOpt, 0, 1, 4.02066788e-01)
    ODMR /= ODMR.max()
    
    DeltaNu = nu - nu0 
    
    # popt = ([11077945096.648079, 11077945096.648079, 37898233225.37501, 37898233225.37501, 
    #          5000000000.0, 1.004524991429497, -0.057891967231220004, 9.071006836633663e-06, 1])
    
    g = 5e9
    tau =1.004524991429497*np.exp(1j* -0.057891967231220)
    kex1 = nu0/2.6e4
    kex2 = kex1
    ki1 = nu0/7.6e3
    ki2 = ki1
    
    #%%%% Plot change in transmission fit with wavelength
    P_in = 9.071006836633663e-06
    popt = [kex1, kex2, ki1, ki2, g, np.abs(tau), np.angle(tau),P_in*1.36/1.01, 1]
    DeltaT, TMean= calcDeltaMicrodiskReturnT(DeltaNu, *popt)
    
    DeltaTMat = np.outer(DeltaT, ODMR)
    
    fMWmat, DeltaNumat = np.meshgrid(fMW, DeltaNu)
    
    
    # load and plot data
    filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-1, 1040.95+1
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 2
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    print('Power into the fibre: ',np.mean(power))
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-2776.7
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = np.abs(S_V/C)
    DeltaT = S_V/np.max(DC_V)*2
    print(np.max(DeltaT))
    
    DeltaT_Smooth = gaussian_filter1d(DeltaT, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    

    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaTMat, DeltaT_Smooth = DeltaTMat*1e4, DeltaT_Smooth*1e4
    
    
    
    DeltaTMin = np.min([DeltaT_Smooth.min(), DeltaTMat.min()])
    DeltaTMax = np.max([DeltaT_Smooth.max(), DeltaTMat.max()])
    
        
    norm = TwoSlopeNorm(
        vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
        vcenter=0,
        vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    )

    
    p = ax[0].pcolormesh(
        X, Y, DeltaT_Smooth,
        cmap='seismic',
        norm=norm,
        shading='auto',
        rasterized=True   
    )
    
    # fig.colorbar(p, ax=ax[0])
    
    # norm = TwoSlopeNorm(
    #     vmin=DeltaTMat.min(),
    #     vcenter=0,
    #     vmax=DeltaTMat.max()
    # )

    p = ax[1].pcolormesh(
        fMWmat/1e6, DeltaNumat/1e9, DeltaTMat,
        cmap='seismic',
        norm=norm,
        shading='gouraud',
        rasterized=True   
    )
    cbar = fig.colorbar(p, ax=ax, location='top')
    cbar.set_label(r'$\Delta T\,(\times10^{-4})$', fontsize = labelfontsize-1)
    cbar.set_ticks([DeltaTMat.min(),DeltaT_Smooth.min(),0,DeltaT_Smooth.max(), DeltaTMat.max()])
    
    
    cbarticks = cbar.get_ticks()
    cbar.set_ticklabels([rf"${val:.1f}$" for val in [DeltaTMat.min(),DeltaT_Smooth.min(),0,DeltaT_Smooth.max(), DeltaTMat.max()]], fontsize = numfontsize-1)
    axw = 0.2/0.3528
    cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    cbar.ax.tick_params(
        direction='in',   # ticks inward
        width=axw,          # tick line width
        length=8         # tick length (optional)
    )
    labels = cbar.ax.get_xticklabels()

    labels[1].set_visible(False)
    labels[3].set_visible(False)

    # print('Max experiment: ', np.max(DeltaT_Smooth))
    
    
    
    # plt.tight_layout()
    
    axs = ax[0]
    
    axs.set_ylim([-nuBW*1e-9, nuBW*1e-9])
    axs.set_ylim([-25, 25])
    axs.set_yticks([-25, 0, 25])

    x0 = 0
    for axx in ax:
        axx.tick_params(
            axis = 'x',
            direction="out",
        )
        axx.tick_params(
            axis = 'y',
            direction="out",
        )
        axx.set_xlim(-4.85, 4.85)
        
    
    if saveBool:
        # fig.savefig(
        #     (_FIGURES_DIR + "/") + "CompareExpToTheory2D.png", dpi=600, bbox_inches="tight",transparent=True
        # )
        
        fig.savefig(
            (_FIGURES_DIR + "/") + "CompareExpToTheory2D.svg", bbox_inches="tight",transparent=True
        )
    
    plt.show()

def plotMicrodiskTheorySensitivitykcavkex(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.37, PRQuantumWidthHalf * 0.97, 
       r'$(\kappa_{\text{i}}+\kappa_{\text{p}})/2\pi$ (Hz)', 
       r'$\kappa_{\text{ex}}/2\pi$ (Hz)', 
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    
    # define cavity parameters
    # kabs0 = c_const/nd*alpha0*2*np.pi
    # kabs1 = c_const/nd*alpha1*2*np.pi
    # kabs = (kabs0 + kabs1)/2*2*np.pi
    N = 200
    ki = np.logspace(7, 12, N)
    kex = np.logspace(7, 12, N)
    
    
    Ki, Kex = np.meshgrid(ki, kex)
    
    ng = nd
    tau = 1
    # lamb0 = lamb_s
    # nu0 = c_const/lamb0
    
    kabs = alpha * c_const/ng
    kabs0 = alpha0 * c_const/ng
    kabs1 = alpha1 * c_const/ng

    T0 = calcTransmissionSinglet(Kex, kabs0 + Ki, 0, tau = tau)
    T1 = calcTransmissionSinglet(Kex, kabs1 + Ki, 0, tau = tau)
    T = (T0 + T1)/2
    DeltaT = T1-T0    
    C = DeltaT/(T)/3.4
    
    Veff = calcVeff()
    
    aMicrodisk = aMicrodiskOnRes(kabs, Ki, Kex)
    P = aMicrodiskRes2PinFibre(Veff, I_sOpt, aMicrodisk)
    P_det = P*T*np.sqrt(etaFibreIR)*etaPostFibre
    eta = np.abs(calcSensitivityCW(nuOpt, C, P_det, lamb_s))
    eta = np.log10(eta)
    
    
    norm = plt.Normalize(vmin=np.min(eta), vmax=-5)
    
    
    # Define your colors (low → high)
    colors = ["#ffffffff", "#ff8a8aff", "#ff1a1aff", "#a80000ff", "#330000ff"]
    
    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", colors)
    
    # p = ax.pcolormesh(
    #     Ki, Kex, eta,
    #     cmap=cmap,
    #     norm =norm,
    #     shading='gouraud',
    #     rasterized=True   
    # )
    
    levels = np.linspace(np.nanmin(eta), np.nanmax(eta), 20 + 1)
    
    # cmap = plt.get_cmap('viridis')
    norm = pltcolors.BoundaryNorm(levels, cmap.N)
    
    p = ax.pcolormesh(
        Ki, Kex, eta,
        cmap=cmap,
        norm=norm,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    
    p = ax.contourf(
            Ki, Kex, eta,
            levels=levels,
            cmap=cmap
        )
    
    cbar = fig.colorbar(p, ax=ax, location='top', extend='max')
    cbar.set_label(r'sensitivity, $\eta\,$ (nT$/\sqrt{\text{Hz}})$', fontsize = labelfontsize-1, labelpad = 7)
    axw = 0.15/0.3528
    cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    cbar.ax.tick_params(
        direction='in',   # ticks inward
        width=axw,          # tick line width
        length=11       , # tick length (optional)
        bottom = True,
        top = False,
        pad = 5
    )
    
    # ax.plot(ki, ki**-0.33*1e12, 'k', lw = 0.2/0.3528)

    ax.plot(ki, ki+ kabs, 'k', lw = 0.2/0.3528)
    
    p0 = [11077945096.648079, 11077945096.648079, 37898233225.37501, 37898233225.37501, 5000000000.0, np.float64(1.004524991429497), np.float64(-0.057891967231220004), np.float64(9.071006836633663e-06), 1.0]
    
    kex = (p0[0] + p0[1])/2
    ki =  (p0[2] + p0[3])/2 - kabs0
    T0 = calcTransmissionSinglet(kex, kabs0 + ki, 0, tau = 1)
    T1 = calcTransmissionSinglet(kex, kabs1 + ki, 0, tau = 1)
    T = (T0 + T1)/2
    DeltaT = (T1-T0    )/3.4
    C = DeltaT/(T)  
    
    Veff = calcVeff()
    

    aMicrodiskHighQ = aMicrodiskOnRes(kabs, (p0[2] + p0[3])/2, (p0[0] + p0[1])/2)
    P = aMicrodiskRes2PinFibre(Veff, I_sOpt, aMicrodiskHighQ)
    P_det = P*T*np.sqrt(etaFibreIR)*etaPostFibre
    print('ideal power : ', P)
    print('ideal C : ', C)
    print('ideal DeltaT : ', DeltaT)
    # print('ideal ODMR linewidth High Q: ', nuOpt)
    etaSNL = np.abs(calcSensitivityCW(nuOpt, C, P_det, lamb_s))
    
    print('etaSNL: ', etaSNL)
    etaSNL = np.log10(etaSNL)
    
    
    ax.scatter( (p0[2] + p0[3])/2 - (kabs0 + kabs1)/2, (p0[0] + p0[1])/2, marker = '*', facecolor = 'gold', s = 25, edgecolors='black', linewidths = 0.15/0.3528)
    
    #High Q mode
    p0 = [1439358456.3236377, 1799198070.4045472, 7575570822.755988, 8224905464.706502, 9000000000.0, np.float64(0.9999999999999999), np.float64(-0.2), np.float64(7.4177169326732685e-06)*0.5, 1]

    
    kex = (p0[0] + p0[1])/2
    ki =  (p0[2] + p0[3])/2 - kabs0
    T0 = calcTransmissionSinglet(kex, kabs0 + ki, 0, tau = 1)
    T1 = calcTransmissionSinglet(kex, kabs1 + ki, 0, tau = 1)
    T = (T0 + T1)/2
    DeltaT = (T1-T0 )/3.4   
    C = DeltaT/(T)
    
    Veff = calcVeff()

    aMicrodiskHighQ = aMicrodiskOnRes(kabs, (p0[2] + p0[3])/2, (p0[0] + p0[1])/2)
    P = aMicrodiskRes2PinFibre(Veff, I_sOpt, aMicrodiskHighQ)
    P_det = P*T*np.sqrt(etaFibreIR)*etaPostFibre
    print('ideal power High Q: ', P)
    print('ideal C High Q: ', C)
    print('ideal DeltaT High Q: ', DeltaT)
    # print('ideal ODMR linewidth High Q: ', nuOpt)
    
    
    etaSNLHighQ = np.abs(calcSensitivityCW(nuOpt, C, P_det, lamb_s))
    
    
    
    print('etaSNLHighQ: ', etaSNLHighQ)
    etaSNLHighQ = np.log10(etaSNLHighQ)
    
    ax.scatter( (p0[2] + p0[3])/2 - (kabs0 + kabs1)/2, (p0[0] + p0[1])/2, marker = '*', facecolor = 'black', s = 10, edgecolors='black', linewidths = 0.15/0.3528)
    
    ax.plot([kabs, kabs], [np.min(Kex), np.max(Kex)], 'k--', lw = 0.2/0.3528, zorder = 10)
    ax.plot([np.min(Ki), np.max(Ki)],[kabs, kabs],  'k--', lw = 0.2/0.3528)
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
    ax.set_ylim([1e7, 1e12])
    ticks = [np.min(eta), etaSNLHighQ, etaSNL]
    cbar.set_ticks(ticks)
    
    labels = cbar.ax.get_xticklabels()
    cbar.set_ticklabels([rf"${val:.0f}$" for val in 10**np.array(ticks)*1e9], fontsize = numfontsize)
    
    # Nudge the 2nd and 3rd labels left/right
    labels[1].set_ha('right')  # shifts it slightly left visually
    labels[2].set_ha('left')   # shifts it slightly right visually
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    plt.tight_layout()
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "2DSensitivity.svg", dpi=600, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotMicrodiskTheoryCompareCalcVsSimple(saveBool = False):
    fig, axArr = plot_custom(
       PRQuantumWidthFull, PRQuantumWidthHalf * 0.97, 
       [r'$(\kappa_{\text{i}}+\kappa_{\text{p}})/2\pi$ (Hz)',r'$(\kappa_{\text{i}}+\kappa_{\text{p}})/2\pi$ (Hz)',r'$(\kappa_{\text{i}}+\kappa_{\text{p}})/2\pi$ (Hz)'], 
       [r'$\kappa_{\text{ex}}/2\pi$ (Hz)',r'$\kappa_{\text{ex}}/2\pi$ (Hz)'], 
       labelPadX = 3, labelPadY = 8, commonY = True, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 3, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    # define cavity parameters
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    N = 200
    ki = np.logspace(7, 12, N)
    kex = np.logspace(7, 12, N)
    
    
    Ki, Kex = np.meshgrid(ki, kex)
    
    g = 0
    ng = 2.41
    tau =1
    lamb0 = 1042e-9
    nu0 = c_const/lamb0
    
    kabs = alpha * c_const/ng
    kabs0 = alpha0 * c_const/ng
    kabs1 = alpha1 * c_const/ng

    T0 = calcTransmissionSinglet(Kex, kabs0 + Ki, 0, tau = tau)
    T1 = calcTransmissionSinglet(Kex, kabs1 + Ki, 0, tau = tau)
    T = (T0 + T1)/2
    DeltaT = T1-T0    
    C = DeltaT/(T) 
    
    
    Veff = calcVeff()
    
    aMicrodisk = aMicrodiskOnRes(kabs, Ki, Kex)
    P = aMicrodiskRes2PinFibre(Veff, I_sOpt, aMicrodisk)
    P_det = P*T*np.sqrt(etaFibreIR)*etaPostFibre
    eta = np.abs(calcSensitivityCW(nuOpt, C, P_det, lamb_s))
    eta = np.log10(eta)
    
    C = np.log10(np.abs(C))
    
    minC = -10
    maxC = 0
    norm = plt.Normalize(vmin=minC, vmax=maxC)
    
    
    # Define your colors (low → high)
    colors = ["#ffffffff", "#ff8a8aff", "#ff1a1aff", "#a80000ff", "#330000ff"]
    
    # Create colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", colors)
    
    # p = ax.pcolormesh(
    #     Ki, Kex, eta,
    #     cmap=cmap,
    #     norm =norm,
    #     shading='gouraud',
    #     rasterized=True   
    # )
    
    # levels = np.linspace(np.nanmin(eta), np.nanmax(eta), 20 + 1)
    
    # # cmap = plt.get_cmap('viridis')
    # norm = pltcolors.BoundaryNorm(levels, cmap.N)
    
    ax = axArr[0,0]
    
    p = ax.pcolormesh(
        Ki, Kex, C,
        cmap=cmap,
        norm=norm,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    
    cbar = fig.colorbar(p, ax=ax, location='right')
    # cbar.set_label(r'', fontsize = labelfontsize-1, labelpad = 7)
    # axw = 0.15/0.3528
    # cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    # cbar.ax.tick_params(
    #     direction='in',   # ticks inward
    #     width=axw,          # tick line width
    #     length=11       , # tick length (optional)
    #     bottom = True,
    #     top = False,
    #     pad = 5
    # )
       
    
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax = axArr[0,2]
    
    mineta = -8
    maxeta = -5
    normeta = plt.Normalize(vmin=mineta, vmax=maxeta)
    p = ax.pcolormesh(
        Ki, Kex, eta,
        cmap=cmap,
        norm=normeta,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    
    cbar = fig.colorbar(p, ax=ax, location='right')
    # cbar.set_label(r'', fontsize = labelfontsize-1, labelpad = 7)
    # axw = 0.15/0.3528
    # cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    # cbar.ax.tick_params(
    #     direction='in',   # ticks inward
    #     width=axw,          # tick line width
    #     length=11       , # tick length (optional)
    #     bottom = True,
    #     top = False,
    #     pad = 5
    # )
       
    
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    ax = axArr[0,1]
    minP = -20
    maxP = 0
    normP = plt.Normalize(vmin=minP, vmax=maxP)
    p = ax.pcolormesh(
        Ki, Kex, np.log(P_det),
        cmap=cmap,
        norm=normP,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    
    cbar = fig.colorbar(p, ax=ax, location='right')
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax = axArr[1,0]
    kabs = alpha * c_const/ng
    kabs0 = alpha0 * c_const/ng
    kabs1 = alpha1 * c_const/ng
    Delk = np.abs(kabs0 - kabs1)
    C = Delk * 4 *Kex/(Ki + Kex + kabs)**2 / (1-2*Kex/(Ki + Kex + kabs)) 
    
    Veff = calcVeff()
    P_det = (1-2*Kex/(Ki + Kex + kabs))**2 * (Ki + Kex + kabs)**2 * (Veff*I_sOpt*nd)/(2*c_const*Kex)  *np.sqrt(etaFibreIR)*etaPostFibre
    # C = Delk * 4 *Kex/(Ki + Kex)**2 / (1-2*Kex/(Ki + Kex)) 
    
    
    # aMicrodisk = aMicrodiskOnRes(kabs, Ki, Kex)
    # P = aMicrodiskRes2PinFibre(Veff, I_sOpt, aMicrodisk)
    # P_det = P*T*np.sqrt(etaFibreIR)*etaPostFibre
    eta = np.abs(calcSensitivityCW(nuOpt, C, P_det, lamb_s))
    
    eta = 4/(3*np.sqrt(3)) * 1/gamma * nuOpt * np.sqrt(h*c_const/lamb_s) / (4*np.sqrt(Kex)*np.abs(kabs0 - kabs1)/(kabs + Kex + Ki) * np.sqrt(I_sOpt*nd*Veff/2/c_const*np.sqrt(etaFibreIR)*etaPostFibre))
    
    eta = np.log10(eta)
    
    C = np.log10(np.abs(C))
    
    
    
    # Define your colors (low → high)
    # p = ax.pcolormesh(
    #     Ki, Kex, eta,
    #     cmap=cmap,
    #     norm =norm,
    #     shading='gouraud',
    #     rasterized=True   
    # )
    
    # levels = np.linspace(np.nanmin(eta), np.nanmax(eta), 20 + 1)
    
    # cmap = plt.get_cmap('viridis')
    # norm = pltcolors.BoundaryNorm(levels, cmap.N)
    
    
    p = ax.pcolormesh(
        Ki, Kex, C,
        cmap=cmap,
        norm=norm,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    cbar = fig.colorbar(p, ax=ax, location='right')
        
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    ax = axArr[1,1]
    p = ax.pcolormesh(
        Ki, Kex, np.log(P_det),
        cmap=cmap,
        norm=normP,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    
    cbar = fig.colorbar(p, ax=ax, location='right')
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax = axArr[1,2]
    p = ax.pcolormesh(
        Ki, Kex, eta,
        cmap=cmap,
        norm=normeta,
        shading='nearest',   # or 'auto'
        rasterized=True
    )
    
    cbar = fig.colorbar(p, ax=ax, location='right')
    ax.tick_params(
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        axis = 'y',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'x',
        direction="out",
    )
    ax.tick_params(
        which = 'minor',
        axis = 'y',
        direction="out",
    )
    
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "2DSensitivity.svg", dpi=600, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    

#%%% Figure 8
def plotLitReview(saveBool=False):
    import pandas as pd
    import numpy as np

    data = [
        {
            "year": 2008,
            "url": "https://www.nature.com/articles/nphys1075#Sec2",
            "sensitivity_nTpRootHz": 1e3,
            "volume_mm3": (100e-6)**3,
            "green_power_uW": None,
            "largest_dimension_nm": 100,
            "type": 'Single',
            "bibtex": "Taylor2008"
        },
        {
            "year": 2008,
            "url": "https://www.nature.com/articles/nature07279",
            "sensitivity_nTpRootHz": 0.5e3,
            "volume_mm3": (100e-6)**3,
            "green_power_uW": None,
            "largest_dimension_nm": 100,
            "type": 'Single',
            "bibtex": "Maze2008"
        },
        {
            "year": 2013,
            "url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.130802#supplemental",
            "sensitivity_nTpRootHz": 50,
            "volume_mm3": (100e-6)**3,
            "green_power_uW": 1.2e3,
            "largest_dimension_nm": None,
            "type": 'Single',
            "bibtex": "Fang2013"
        },
        {
            "year": 2015,
            "url": "https://www.nature.com/articles/nphys3291",
            "sensitivity_nTpRootHz": 290e-3,
            "volume_mm3": 3 * 3 * 0.3,
            "green_power_uW": 50e3,
            "largest_dimension_nm": 300e3,
            "type": 'Ensemble',
            "bibtex": "Clevenson2015"
        },
        {
            "year": 2016,
            "url": "https://www.pnas.org/doi/10.1073/pnas.1601513113",
            "sensitivity_nTpRootHz": 15e-3,
            "volume_mm3": 13 * 200 * 2000 * 1e-9,
            "green_power_uW": 4.5e6,
            "largest_dimension_nm": 2e6,
            "type": 'Ensemble',
            "bibtex": "Barry2016"
        },
        {
            "bibtex": "Pelliccione2016",
            "year": 2016,
            "url": "https://doi.org/10.1038/nnano.2016.68",
            "sensitivity_nTpRootHz": 3e3,
            "volume_mm3": (10e-6)**3,
            "green_power_uW": 30,
            "largest_dimension_nm": 6,
            "type": 'Single'
        },
        {
            "year": 2020,
            "url": "https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.14.044058",
            "sensitivity_nTpRootHz": 310e-3,
            "volume_mm3": 600 * 4650 * 1e-9,
            "green_power_uW": 1e6,
            "largest_dimension_nm": 4650,
            "type": 'Ensemble',
            "bibtex": "Patel2020"
        },
        {
            "year": 2021,
            "url": "https://advanced.onlinelibrary.wiley.com/doi/10.1002/qute.202000111",
            "sensitivity_nTpRootHz": 344e-3,
            "volume_mm3": (0.5*np.pi*0.035**2),
            # "volume_mm3": 0.8*0.8*0.5,
            "green_power_uW": 23.5e3,
            "largest_dimension_nm": 0.5e6,
            "type": 'Ensemble',
            "bibtex": "Sturner2021"
        },
        {
            "bibtex": "Zhang2021",
            "year": 2021,
            "url": "https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.15.064075",
            "sensitivity_nTpRootHz": 17e-3,
            "volume_mm3": 0.5,
            "green_power_uW": 100e3,
            "type": 'Ensemble',
        },
        {
            "year": 2022,
            "url": "https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.17.014009",
            "sensitivity_nTpRootHz": 70e-3,
            "volume_mm3": 200 * 200 * 3 * 1e-9,
            "green_power_uW": 400e3,
            "largest_dimension_nm": 200e3,
            "type": 'Ensemble',
            "bibtex": "HyunShim2022"
        },
        {
            "year": 2023,
            "url": "https://pubs.acs.org/doi/pdf/10.1021/acsphotonics.3c00259",
            "sensitivity_nTpRootHz": 4.2e3, #PNSL sensitivity
            "volume_mm3": 1e-8, #40e-3*0.25e-6
            "green_power_uW": 0.09,
            "largest_dimension_nm": 40e3,
            "type": 'Ensemble',
            "bibtex": "Li2023"
        },
        {
            "year": 2023,
            "url": "https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.19.044042",
            "sensitivity_nTpRootHz": 30e-3,
            "volume_mm3": 1,
            "green_power_uW": 1.8e6,
            "largest_dimension_nm": 1e6,
            "type": 'Ensemble',
            "bibtex": "Graham2023"
        },
        {
            "year": 2024,
            "url": "https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.22.044069",
            "sensitivity_nTpRootHz": 0.46e-3,
            "volume_mm3": 3 * 3 * 0.62,
            "green_power_uW": 12e6,
            "largest_dimension_nm": 3e6,
            "type": 'Ensemble',
            "bibtex": "Barry2024"
        },
        {
            "bibtex": "Sekiguchi2024",
            "year": 2024,
            "url": "https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.21.064010",
            "sensitivity_nTpRootHz": 9.4e-3,
            "volume_mm3": 4e-3, # sensing volume of 4 × 10−3 mm3
            "green_power_uW": 390e3,
            "largest_dimension_nm": 400e3,
            "type": 'Ensemble'
        },
        {
            "year": 2025,
            "url": "https://www.nature.com/articles/s43246-025-00770-x#Sec14",
            "sensitivity_nTpRootHz": 1e3,
            # "volume_mm3": 2.6 * 0.44 * 0.1 * 1e-9,
            "volume_mm3": np.pi*2.8**2 * 0.7 * 1e-9,
            "green_power_uW": 10e3,
            "largest_dimension_nm": 2600,
            "type": 'Ensemble',
            "bibtex": "Katsumi2025"
        },
        {
            "bibtex": "Sorensen2025",
            "year": 2025,
            "url": None,
            "sensitivity_nTpRootHz": 58,
            "volume_mm3": 0.8 * 3.1416 * ((4.4/2) ** 2) * 1e-9,
            "green_power_uW": 250,
            "largest_dimension_nm": 4400,
            "type": 'Ensemble'
        },
        {
            "bibtex": "Acosta2010",
            "year": 2010,
            "url": None,
            "sensitivity_nTpRootHz": 7,
            "volume_mm3": 50*50*300*1e-9,
            "green_power_uW": 8e5,
            "IR_power_W": 8e-3,
            "length_um": 300,
            'area_um2': 50**2,
            "type": 'EnsembleIR',
            "cavityBool": False
        },
        {
            "bibtex": "Jensen2014",
            "year": 2014,
            "url": None,
            "sensitivity_nTpRootHz": 2.5,
            "volume_mm3": 90*90*200*1e-9,
            "green_power_uW": -1,
            "IR_power_W": -1,
            "length_um": 200,
            'area_um2': 90**2,
            "type": 'EnsembleIR',
            "cavityBool": True
        },
        {
            "bibtex": "Chatzidrosos2017",
            "year": 2017,
            "url": None,
            "sensitivity_nTpRootHz": 0.022,
            "volume_mm3": 0.39*np.pi*(76e-3/2)**2,
            "green_power_uW": -1,
            "IR_power_W": -1,
            "length_um": 0.39e3,
            'area_um2': np.pi*(76/2)**2,
            "type": 'EnsembleIR',
            "cavityBool": True
        },
        # {
        #     "bibtex": "Schall2024",
        #     "year": 2024,
        #     "url": None,
        #     "sensitivity_nTpRootHz": 7.5e-3,
        #     "volume_mm3": 0.5*0.063**2,
        #     "green_power_uW": -1,
        #     "IR_power_W": -1,
        #     "largest_dimension_nm": 200e3,
        #     "type": 'EnsembleIR',
        #     "cavityBool": True
        # },
        {
            "bibtex": "OurWork",
            "year": 2025,
            "url": None,
            "sensitivity_nTpRootHz": 253,
            "volume_mm3": 0.8 * np.pi * ((4.2/2) ** 2) * 1e-9,
            "green_power_uW": 250,
            "IR_power_W": 1e-6,
            "length_um": np.pi*2*(4.2/2),
            'area_um2': 0.6**2,
            "type": 'EnsembleIR',
            "cavityBool": True
        },
        {
            "bibtex": "OurWorkImproved",
            "year": 2025,
            "url": None,
            "sensitivity_nTpRootHz": 13*np.sqrt(etaFibreIR*etaPostFibre),
            "volume_mm3": 0.8 * np.pi * ((4.2/2) ** 2) * 1e-9,
            "green_power_uW": 250,
            "IR_power_W": 1e-6,
            "length_um": np.pi*2*(4.2/2),
            'area_um2': 0.6**2,
            "type": 'EnsembleIR',
            "cavityBool": True
        },
    ]

    df = pd.DataFrame(data)

    fig, axArr = plot_custom(
        PRQuantumWidthFull*0.49, 0.6*PRQuantumWidthFull,
        [r"sampled volume, $\delta V$ (mm$^3$)",r"length, $L$ (m)"],
        [r"sensitivity, $\eta$ (nT$\,$Hz$^{-1/2}$)", r"$\eta \sqrt{A}$ (nT$\,$Hz$^{-1/2}$m)"],
        labelPadX=3, labelPadY=10, commonY=False, commonX=False,
        wSpaceSetting=0, hSpaceSetting=0,
        spineColor='black', tickColor='black', textColor='black',
        nrow=2, fontType='BeraSans', axefont=labelfontsize,
        numsize=numfontsize, widthBool=False
    )
    ax = axArr[0]

    s = 20
    
    date_sorted = df.sort_values(by='year')

    for idx, (i, row) in enumerate(date_sorted.iterrows(), start=1):
        print('i = ',rf'$\mathbf{{\mathrm{{{idx}}}}}$', '  ' , row['bibtex'])
        if not(row['type'] == "Single") & (row['bibtex'] != "OurWork")& (row['bibtex'] != "OurWorkImproved"):
            continue
        ax.text(row['volume_mm3'], row['sensitivity_nTpRootHz'],
                rf'$\mathbf{{\mathrm{{{idx}}}}}$', color=colorRed,
                fontsize=numfontsize, ha='center', va='center')
        
    for idx, (i, row) in enumerate(date_sorted.iterrows(), start=1):
        print('i = ',rf'$\mathbf{{\mathrm{{{idx}}}}}$', '  ' , row['bibtex'])
        if not(row['type'] == "Ensemble") & (row['bibtex'] != "OurWork")& (row['bibtex'] != "OurWorkImproved"):
            continue
        ax.text(row['volume_mm3'], row['sensitivity_nTpRootHz'],
                rf'$\mathbf{{\mathrm{{{idx}}}}}$', color=colorRed,
                fontsize=numfontsize, ha='center', va='center')
        
    for idx, (i, row) in enumerate(date_sorted.iterrows(), start=1):
        print('i = ',rf'$\mathbf{{\mathrm{{{idx}}}}}$', '  ' , row['bibtex'])
        if not(row['type'] == "EnsembleIR") & (row['bibtex'] != "OurWork")& (row['bibtex'] != "OurWorkImproved"):
            continue
        ax.text(row['volume_mm3'], row['sensitivity_nTpRootHz'],
                rf'\textbf{{\text{{{idx}}}}}', color=colorDarkRed,
                fontsize=numfontsize, ha='center', va='center')

    # Our work as an asterisk
    filt_our = df['bibtex'] == "OurWork"
    ax.scatter(df[filt_our]['volume_mm3'], df[filt_our]['sensitivity_nTpRootHz'],
               marker='*', color=colorDarkRed, s=s*3, zorder=2, edgecolors='none')
    filt_our = df['bibtex'] == "OurWorkImproved"
    ax.scatter(df[filt_our]['volume_mm3'], df[filt_our]['sensitivity_nTpRootHz'],
                marker='*', color='white', s=s*2, zorder=2, edgecolors=colorDarkRed, lw = lw)

    volumeCont = np.logspace(-14.7, 1, 100)/1e9
    scale=3
    alpha0local = alpha0*scale
    alpha1local = alpha1*scale
    alphalocal = alpha*scale
    num_NV = lambda V: (V * rho_NV)
        
    sens, C, nu, R = solveSensitivityVisSingle(
                                    OmegaR=OmegaROpt_vis,
                                    I_p=I_pOpt_vis,
                                    bw=0.01e9*2*np.pi,
                                    Npoints=101,
                                    lamb=532e-9,
                                    returnAllParams=True
                                    )
    
    eta_CW_psnl = calcSensitivityCW_countrate(nu*1e3, C/4, R*num_NV(volumeCont))*1e9
    
    eta_Ramsey_psnl = calcSensitivityRamsey_countrate(C*2, R*num_NV(volumeCont), tread = 1e-6, tdephase = T2s, twrite = 1e-6)*1e9
    
    eta_sp = lambda V: 1/(gamma*np.sqrt(num_NV(V)*T2s))*1e9

    LArr = volumeCont**(1/3)
    etaDiamondCubeNoCavity = []
    Tm = 1
    for j, L in enumerate(LArr):
        A = L**2
        
        T0 = TEtalon(Tm, L, alpha0local)  
        T1 = TEtalon(Tm, L, alpha1local) 
        T = T0/2 + T1/2
        C = (T0-T1)/T
        P = PinEtalon(Tm, L, alphalocal, I_sOpt, A)
        R = T*P/(hbar*omegas)
        etaDiamondCubeNoCavity.append(calcSensitivityCW(nuOpt, C, P*T, lamb_s)*1e9)

    L0 = 0.2754228703338166 #mm
    etaDiamondAbsorptionLengthNoCavity = []
    Tm = 1
    
    T0 = TEtalon(Tm, L0, alpha0)  
    T1 = TEtalon(Tm, L0, alpha1) 
    T = T0/2 + T1/2
    C = (T0-T1)/T
    
    for j, L in enumerate(LArr):
        A = volumeCont[j]/L0
        P = PinEtalon(Tm, L0, alpha, I_sOpt, A)
        R = T*P/(hbar*omegas)
        etaDiamondAbsorptionLengthNoCavity.append(calcSensitivityCW(nuOpt, C, P*T, lamb_s)*1e9)


    # ax.plot(volumeCont*1e9, etaDiamondCubeNoCavity, lw = 0.2/0.358, color = 'red' )
    ax.plot(volumeCont*1e9, etaDiamondAbsorptionLengthNoCavity, lw = 0.2/0.358, color = colorDarkRed , zorder = -1)
    
    ax.plot(volumeCont*1e9, eta_CW_psnl, color = colorRed, lw = lw, zorder = -1)
    ax.plot(volumeCont*1e9, eta_Ramsey_psnl, color = colorRed ,  linestyle = '--', lw = lw, zorder = -1)
    ax.fill_between(volumeCont*1e9, eta_sp(volumeCont)*1e-10, eta_sp(volumeCont),
                    color = 'k', alpha = 0.3, lw = 0, zorder = -1)
    
    
    # fig.text(0.167, 0.675, r'$T_2^*$ spin projection limit', va='center', rotation=-24, fontsize = labelfontsize, color = 'white')
    # fig.text(0.32, 0.71, r'$\sim \delta V^{-1/2}$', va='center', rotation=-26,alpha = 0.4, fontsize = labelfontsize, color = 'black')
    
    min_x_value = 1e-14
    max_x_value = 1e1
    min_y_value = 1e-4
    max_y_value = 1e4
    
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.07
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=False)
    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    # ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    # ax.set_yticks([min_y_value, 0, max_y_value])
    # ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'], fontsize = numfontsize)
    # ax.set_xticks([-6,0,6])
    
    ax.set_ylim([min_y_value, max_y_value])
    ax.set_xlim([min_x_value, max_x_value])
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # ax.tick_params(axis='x', pad=5)  # Padding for x-axis tick labels
    ax.tick_params(axis='y', pad=3)  # Padding for y-axis tick labels
    # ax.set_ylabel('contrast (\%)', labelpad=0)
    
    ax = axArr[1]
    
    for idx, (i, row) in enumerate(date_sorted.iterrows(), start=1):
        print('i = ',rf'$\mathbf{{\mathrm{{{idx}}}}}$', '  ' , row['bibtex'])
        if not(row['type'] == "EnsembleIR") & (row['bibtex'] != "OurWork")& (row['bibtex'] != "OurWorkImproved"):
            continue
        ax.text(row['length_um']/1e6, row['sensitivity_nTpRootHz']*np.sqrt(row['area_um2'])*1e-6,
                rf'\textbf{{\text{{{idx}}}}}', color=colorDarkRed,
                fontsize=numfontsize, ha='center', va='center')
    
    filt_our = df['bibtex'] == "OurWork"
    ax.scatter(df[filt_our]['length_um']/1e6, df[filt_our]['sensitivity_nTpRootHz']*np.sqrt(df[filt_our]['area_um2'])*1e-6,
               marker='*', color=colorDarkRed, s=s*3, zorder=2, edgecolors='none')
    
    filt_our = df['bibtex'] == "OurWorkImproved"
    ax.scatter(df[filt_our]['length_um']/1e6, df[filt_our]['sensitivity_nTpRootHz']*np.sqrt(df[filt_our]['area_um2'])*1e-6,
               marker='*', color='white', s=s*2, zorder=2, edgecolors=colorDarkRed, lw = lw)
    
    LArr_mm = np.logspace(-3, 3, 101)
    
    etaDiamondNoCavity = []    
    
    for j, L in enumerate(LArr_mm):
        A = 10**2
        L *= 1e-3
        
        T0 = TEtalon(1, L, alpha0local)  
        T1 = TEtalon(1, L, alpha1local) 
        T = T0/2 + T1/2
        C = (T0-T1)/T
        P = PinEtalon(1, L, alphalocal, I_sOpt, A)
        etaDiamondNoCavity.append(calcSensitivityCW(nuOpt, C, P*T, lamb_s)*1e9*np.sqrt(A))
    
    
    etaDiamondIdealCavity = []
    N = 10000
    
    # mirror transmissivity array
    Tmin = -7
    Tmax = 0
    Tm = np.logspace(Tmin, Tmax, N)
    
    Tm_min_list = []
    eta_min_list = []
    
    for j, L in enumerate(LArr_mm):
        L *= 1e-3
        A = 100**2
        T0 = TEtalon(Tm, L, alpha0local)  
        T1 = TEtalon(Tm, L, alpha1local) 
        T = T0/2 + T1/2
        C = (T0-T1)/T
        P = PinEtalon(Tm, L, alphalocal, I_sOpt, A)
        eta = calcSensitivityCW(nuOpt, C, P*T, lamb_s)
        eta_min = eta.min()*1e9*np.sqrt(A)
        etaDiamondIdealCavity.append(eta_min)
        idx = np.argmin(eta)
        Tm_min_list.append(Tm[idx])
        
    idx = np.argsort(np.array(etaDiamondIdealCavity))
    print(np.array(etaDiamondIdealCavity)[idx[0]])
    print(np.array(LArr_mm)[idx[0]])

    ax.plot(LArr_mm*1e-3, etaDiamondIdealCavity, lw = 0.2/0.358, color = 'k', linestyle = '--')
    ax.plot(LArr_mm*1e-3, etaDiamondNoCavity, lw = 0.2/0.358, color = 'k' )
    ax.fill_between(LArr_mm*1e-3, eta_sp(LArr_mm*1e-3*A)*np.sqrt(A)*1e-10, eta_sp(LArr_mm*1e-3*A)*np.sqrt(A),
                    color = 'k', alpha = 0.3, lw = 0)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    min_x_value = 1e-6
    max_x_value = 1e-2
    min_y_value = 1e-10
    max_y_value = 1e-2
    
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value
    
    limScale = 0.07
    limScaley = 0.06
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=False)
    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    # ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    # ax.set_yticks([min_y_value, 0, max_y_value])
    # ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'], fontsize = numfontsize)
    # ax.set_xticks([-6,0,6])
    
    ax.set_ylim([min_y_value, max_y_value])
    ax.set_xlim([min_x_value, max_x_value])
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "litReviewPaper" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )
        # fig2.savefig(
        #     "../figures/" + "litReviewPaperNorm" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        # )
    plt.show()

#%%% Appendix
def plotDetunings(saveBool = False):
    fig, axx = plot_custom(
       PRQuantumWidthHalf, PRQuantumWidthHalf*1.5, 
       [r'']*100, 
       [r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)',r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)',r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)',r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)',r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)',r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)',r'$T$ ', r'$\Delta T$ ($\times 10^{-4}$)']*6,
       labelPadX = 3, labelPadY = 1, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 6, ncol = 2, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False, widthRatios=[1,1], heightRatios=[1,1]
    )
    
    
    #%%%% Plot the different modes
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.2um_0/20260202/1030_1070nm_5nmps_averaged_norm.mat'
    filePathAbs =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260129/ODMR_20260129_CW__16_12dBAtt_4.2um_0_1__2.7768GHz_0MHz_001.mat'
    data = loadmat(filePath, squeeze_me=True)
    dataAbs = loadmat(filePathAbs, squeeze_me=True)
    
    W = data['W']
    TOut = data['TOut']
    
    lambmin = 1035
    lambmax = 1063
    
    filt = np.logical_and(W< lambmax, W>lambmin)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]
    skip = 1
    # axArr[0].plot(W[1::skip], TOut[1::skip], lw = 0.15/0.3528, color = colorDarkRed)
    # ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = ms/2, zorder = 10, color = 'k')

    dataAbs = dataAbs['data']
    W_nm = dataAbs['laserWLsRead'].item()*1e9
    S_V = dataAbs['signal'].item()
    DC_V = dataAbs['contrast'].item() 
    C = S_V/DC_V;   
    deltaT = S_V/np.max(DC_V);
    power = dataAbs['powerIn'].item()
    
    
    print(power)
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
    skip = 1
    W_nm= W_nm[filt]
    S_V= S_V[filt]
    deltaT = deltaT[filt]
    power = power[filt]
    
    contrast = -deltaT/power*np.max(power)*1e3
    
    P_in = 20e-6 * np.sqrt(etaFibreIR) * np.array([1, 1, 1,1,1,1])
    lambModes = np.array([1040.74, 1042.98, 1047.12, 1048.62, 1057.6, 1061.29])
    nuModes = c_const/lambModes/1e-9
    delTMin = [0.2,0.4,0.2,0.05,0.03,0.05]
    delTMax = [0.2,0,0.2,0.1,0.03,0.05]
    bwGHzArr = [150,150,100,100,200,100]
    Tmin = [0.3, 0.3,0.3, 0.7, 0.7, 0.7]
    
    Veff = calcVeff()
    g = np.array([5e9, 5e9, 20e9, 1e9, 0 ,0])
    t = np.array([0.98,0.93,0.69,0.9,1,0.93])
    phi = np.array([-0.057,0.02,-0.04,-0.4,0,0.3])
    tau =t*np.exp(1j* phi)
    # kex1 = nuModes/np.array([2.6e4])
    kex1 = np.array([1.1e10,40e9,3.5e9,1.5e9,1.1e10,1.2e9])
    kex2 = kex1
    ki1 = np.array([4e10,15e9,18e9,5e10,4e10,2e10])
    ki2 = ki1
    
        
    numin = -200
    numax = 200
    Delta = np.linspace(numin, numax, 501)
    
    
    deltaOmegaGHz = Delta
    deltaOmegaGHzMin = deltaOmegaGHz.min()
    deltaOmegaGHzMax = deltaOmegaGHz.max()
    scale = np.array([1,1,1,1,1,0.05])*1e3
    gradScale = np.array([1,1,7,10,1,2])
    deltaOff = np.array([0,8,-5,1,1,-5])
    
    
    for i, lamb in enumerate(lambModes):
        bwGHz = bwGHzArr[i]
        nuDataGHz = c_const/W
        nuModeGHz = c_const/lamb
        
        nuDataGHz -= nuModeGHz
        
        filt = np.logical_and(nuDataGHz< bwGHz, nuDataGHz>-bwGHz)
        
        axx[i,0].scatter(nuDataGHz[filt], TOut[filt], edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkRed)
        
        bwGHz = bwGHzArr[i]
        nuDataGHz = c_const/W_nm
        nuModeGHz = c_const/(lamb+0.06)
        
        nuDataGHz -= nuModeGHz
        
        filt = np.logical_and(nuDataGHz< bwGHz, nuDataGHz>-bwGHz)
        
        axx[i,1].scatter(nuDataGHz[filt]-deltaOff[i], contrast[filt], edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkBlue)
        # axx[i,1].plot(nuDataGHz[filt], contrast[filt], zorder = 10, color = colorDarkBlue)
        
        p0 = [kex1[i], kex2[i], ki1[i], ki2[i], g[i], t[i], phi[i],P_in.mean(), 1]

        # popt, pcov = curve_fit(calcDeltaMicrodiskReturnT, nuDataGHz[filt]*1e9, TOut[filt], p0=p0,  bounds=(lower_bounds, upper_bounds))
        
        
        # DeltaTFitP0, TMean = calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *p0, Veff =Veff)
        popt=p0        
        if i < 2:
            upperScale = 1.2
            lowerScale = 0.9
            if phi[i]<0:
                lower_bounds = [kex1[i]*lowerScale, kex2[i]*lowerScale, ki1[i]*lowerScale, ki2[i]*lowerScale, g[i]*lowerScale, np.abs(tau[i])*lowerScale, np.angle(tau[i])*1.01,P_in[i]*0.1, 0.1]
                upper_bounds = [kex1[i]*upperScale, kex2[i]*upperScale, ki1[i]*upperScale, ki2[i]*upperScale, g[i]*upperScale, np.abs(tau[i])*upperScale, np.angle(tau[i])*0.99,P_in[i]*10, 10]
            else:
                
                lower_bounds = [kex1[i]*lowerScale, kex2[i]*lowerScale, ki1[i]*lowerScale, ki2[i]*lowerScale, g[i]*lowerScale, np.abs(tau[i])*lowerScale, np.angle(tau[i])*0.7,P_in[i]*0.1, 0.1]
                upper_bounds = [kex1[i]*upperScale, kex2[i]*upperScale, ki1[i]*upperScale, ki2[i]*upperScale, g[i]*upperScale, np.abs(tau[i])*upperScale, np.angle(tau[i])*2,P_in[i]*10, 10]
                
            # skip = 106
            def fit_func(Delta, kex1, kex2, ki1, ki2, g, tau, phi, P_in, scale):
                DeltaT, _, _, _ = calcDeltaMicrodiskReturnT(Delta, kex1, kex2, ki1, ki2, g, tau, phi, P_in, scale)
                return DeltaT
            popt, pcov = curve_fit(calcDeltaMicrodisk, (nuDataGHz[filt]-deltaOff[i])*1e9, contrast[filt]/1e3, p0=p0,    bounds=(lower_bounds, upper_bounds))
            print('popt: ', popt)
            
        
        DeltaTFit, TMean,kabs0, kabs1, alphaOff, alphaOn= calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *popt, printBool = True, Veff =Veff)
        TMean/= TMean.max()
        P_det = P_in.mean()*TMean*np.sqrt(etaFibreIR)*etaPostFibre
        
        
        axx[i,0].plot(deltaOmegaGHz, TMean*t[i], lw = 0.2/0.3528, color = colorRed)
        gradient = np.gradient(TMean*t[i], deltaOmegaGHz)
        # axx[i,0].plot(deltaOmegaGHz, gradient, lw = 0.2/0.3528, color = colorRed)
    
        axx[i,1].plot(deltaOmegaGHz, DeltaTFit*scale[i]+ gradScale[i]*gradient, lw = 0.2/0.3528, color = colorBlue)
        print('Scale: ', popt[-1])

    
    for i in range(6):
        # axx[i,0].set_ylim([Tmin[i]-0.07,1])
        # axx[i,0].spines['left'].set_bounds(Tmin[i],1)
        # axx[i,1].set_ylim([-delTMin[i],delTMax[i]])
        # axx[i,1].spines['left'].set_bounds(-delTMin[i],delTMax[i])
        for ax in axx[i,:]:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
            ax.tick_params(axis='x', which='both', top=False, bottom=True)
            
            ax.set_xlim([-bwGHzArr[i],bwGHzArr[i]])
            ax.spines['bottom'].set_bounds(-bwGHzArr[i],bwGHzArr[i])
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.set_ticks_position('bottom')
            # ax.set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal
            ax.xaxis.set_label_coords(0.55,-0.27)
        
    ax = axx[0,0]
    # ax.yaxis.set_label_coords(-0.16, 1.05)
    
    
    # ax.text(-220, 1.09, r"$T$ (au)", color=colorRed,
    #      ha='right', fontsize = labelfontsize-1)
    # ax.text(-220, 1.24, r"$\Delta T$ (au)", color=colorBlue,
    #      ha='right', fontsize = labelfontsize-1)

    
    # axx[1].yaxis.set_label_coords(-0.04, 1.05)
    
    
    fig.subplots_adjust(hspace=0.7)
    
    # TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp)
    # TMW = microwaveDetuning3(f, nuOpt, 0, amp, 0)
    # print('triple/single: ', TMW.max()/TMWsingle1.max())
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "parameterDefinitions.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

    
def plotMicrodiskDeltaTvsMWDetSI(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.25, PRQuantumWidthHalf * 0.8, 
       r'$\Delta \omega_{\text{MW}}/2\pi$ (MHz)', 
       # [r'$S_{\text{LI}}$ ($\upmu$V)',r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)', r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)'],
        [r'',r'', r''],
       labelPadX = 5, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 3, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize,
       widthBool =True,  widthRatios = [1], heightRatios = [0.8,1,1]
    )
    
    pos = ax[0].get_position()


    ax[0].yaxis.set_label_coords(-0.25, 0.5)  # (x, y)
    
    ax[0].set_position([
        pos.x0,
        pos.y0 +0.12,  # move down
        pos.width,
        pos.height
    ])
    
    # define cavity parameters
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    
    fMW0 = 2776.7e6
    fMWBW = 1300e6
    
    fCont = np.linspace(-10e6, 10e6, 1000)
    SLI = microwaveDetuning3(fCont, 900e3, 0, 10.5, 0.3)
    SLIDeriv = microwaveDetuningDeriv3(fCont, 980e3, 0, -5e6, 0.3)
    
    ax[0].plot(fCont*1e-6, SLI, zorder = -1, color = colorRed, lw = lw)
    ax[0].plot(fCont*1e-6, SLIDeriv, zorder = -1, color = colorRed, lw = lw)
    
    
    
    # load and plot data
    # filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260127/ODMR_20260127_CW__16_18dBAtt_4.2um_0_1_2.87GHz_30MHz_002.mat'
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    filePath = filepathODMR+ filenameODMR4p4umBiasLI1
    # filePath = filepathODMR+ filenameODMR4p4umBiasLI2
    
    filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    
    
    fZeroField = 2776.84
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0 - 22e9
    lamb0 = c_const/nu0
    
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-10, 1040.95+10
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 8
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-fZeroField
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = S_V/C
    DeltaT = -S_V/np.max(DC_V)
    
    DeltaT_Smooth = gaussian_filter1d(S_V, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    

    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaT_Smooth = DeltaT_Smooth*1e6
    
    
    
    DeltaTMin = np.min([DeltaT_Smooth.min()])
    DeltaTMax = np.max([DeltaT_Smooth.max()])
    
        
    norm = TwoSlopeNorm(
        vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
        vcenter=0,
        vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    )

    
    p = ax[1].pcolormesh(
        X, Y, DeltaT_Smooth,
        cmap='seismic',
        norm=norm,
        shading='auto',
        rasterized=True   
    )
    
    N = int(np.ceil(np.shape(DeltaT_Smooth)[0]/2)) + 0 
    M = 2
    ax[0].scatter(f, np.mean(DeltaT_Smooth[N-M:N+M,:], 0), marker = '.', s = 2, color = 'k')
    
    # load and plot data
    # filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260127/ODMR_20260127_CW__16_18dBAtt_4.2um_0_1_2.87GHz_30MHz_001.mat'
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    # filePath = filepathODMR+ filenameODMR4p4umBiasLI1
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    
    
    
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0 - 10e9
    lamb0 = c_const/nu0
    
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-10, 1040.95+10
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 8
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-fZeroField
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= -S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = S_V/C
    DeltaT = S_V/np.max(DC_V)
    
    DeltaT_Smoothderiv = gaussian_filter1d(S_V, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    

    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaT_Smoothderiv = DeltaT_Smoothderiv*1e6
    
    
    # N = int(np.ceil(np.shape(DeltaT_Smoothderiv)[0]/2))
    ax[0].scatter(f, np.mean(DeltaT_Smoothderiv[N-M:N+M,:], 0), marker = '.', s = 2, color = 'k')
    
    # DeltaTMin = np.min([DeltaT_Smoothderiv.min()])
    # DeltaTMax = np.max([DeltaT_Smoothderiv.max()])
    
        
    # norm = TwoSlopeNorm(
    #     vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
    #     vcenter=0,
    #     vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    # )

    
    p = ax[2].pcolormesh(
        X, Y, DeltaT_Smoothderiv,
        cmap='seismic',
        norm=norm,
        shading='auto',
        rasterized=True   
    )
    
    ax[1].xaxis.set_label_position('top')
    ax[1].tick_params(axis='x', which='both', top=True, bottom=True, labeltop = True, labelbottom = False)
    ax[1].set_xlabel(ax[2].get_xlabel(), fontsize = labelfontsize -1, labelpad = 6)
    ax[2].tick_params(axis='x', which='both', top=True, bottom=True, labeltop = False, labelbottom = False)
    ax[2].set_xlabel(r'')
    
    ax[0].tick_params(axis='x', which='both', top=True, bottom=True, labeltop = False, labelbottom = True)
    ax[2].set_xlabel(r'')
    
    
    
    
    # cbar = fig.colorbar(p, ax=ax, location='bottom', pad = 0.05)
    # cbar.set_label(r'lock-in signal, $S_{\text{LI}}$ ($\upmu$V)', fontsize = labelfontsize-1)
    # cbar.set_ticks([DeltaT_Smooth.min(), DeltaT_Smoothderiv.min(),0, DeltaT_Smoothderiv.max(),DeltaT_Smooth.max()])
    
    
    # cbarticks = cbar.get_ticks()
    # cbar.set_ticklabels([rf"${val:.0f}$" for val in cbarticks], fontsize = numfontsize)
    # axw = 0.2/0.3528
    # cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    # cbar.ax.tick_params(
    #     direction='in',   # ticks inward
    #     width=axw,          # tick line width
    #     length=12         # tick length (optional)
    # )
    # labels = cbar.ax.get_xticklabels()

    # labels[1].set_visible(False)
    # labels[3].set_visible(False)

    # print('Max experiment: ', np.max(DeltaT_Smooth))
    


    x0 = 0
    for axx in ax[1::]:
        axx.tick_params(
            axis = 'x',
            direction="out",
        )
        axx.tick_params(
            axis = 'y',
            direction="out",
        )
        axx.plot([-20, 20], [0,0], 'k--', lw = 0.2/0.3528)
        axx.set_ylim(-21, 21)
        axx.set_xlim(-4, 4)
        axx.set_xticks([-4,0, 4])
    

    if saveBool:
        # fig.savefig(
        #     (_FIGURES_DIR + "/") + "CompareExpToTheory2D.png", dpi=600, bbox_inches="tight",transparent=True
        # )
        
        fig.savefig(
            (_FIGURES_DIR + "/") + "CompareLockInResultsField.svg", bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotMicrodiskDeltaTvsMWDetZeroFieldSI(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthHalf*1.06, PRQuantumWidthHalf * 0.8, 
       r'$\Delta \omega_{\text{MW}}/2\pi$ (MHz)', 
       # [r'$S_{\text{LI}}$ ($\upmu$V)',r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)', r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)'],
       [r'$S_{\text{LI}}$ ($\upmu$V)',r'', r''],
       labelPadX = 5, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 3, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize,
       widthBool =True,  widthRatios = [1], heightRatios = [0.8,1,1]
    )
    
    pos = ax[0].get_position()


    ax[0].yaxis.set_label_coords(-0.15, 0.5)  # (x, y)
    
    ax[0].set_position([
        pos.x0,
        pos.y0 +0.12,  # move down
        pos.width,
        pos.height
    ])
    
    # define cavity parameters
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    
    fMW0 = 2776.6e6*0
    fMWBW = 13e6
    
    
    # load and plot data
    filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260127/ODMR_20260127_CW__16_18dBAtt_4.2um_0_1_2.87GHz_30MHz_002.mat'
    
    fZeroField = 2870.3
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0 - 26e9
    lamb0 = c_const/nu0
    
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-10, 1040.95+10
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 8
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-fZeroField
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = S_V/C
    DeltaT = -S_V/np.max(DC_V)
    
    DeltaT_Smooth = gaussian_filter1d(S_V, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    

    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaT_Smooth = DeltaT_Smooth*1e6
    
    
    
    DeltaTMin = np.min([DeltaT_Smooth.min()])
    DeltaTMax = np.max([DeltaT_Smooth.max()])
    
        
    norm = TwoSlopeNorm(
        vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
        vcenter=0,
        vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    )

    
    p = ax[1].pcolormesh(
        X, Y, DeltaT_Smooth,
        cmap='seismic',
        norm=norm,
        shading='auto',
        rasterized=True   
    )
    
    N = int(np.ceil(np.shape(DeltaT_Smooth)[0]/2))
    M = 20
    ax[0].scatter(f, np.mean(DeltaT_Smooth[N-M:N+M,:], 0), marker = '.', s = 5, color = 'k')
    
    # load and plot data
    filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260127/ODMR_20260127_CW__16_18dBAtt_4.2um_0_1_2.87GHz_30MHz_001.mat'
    
    
    
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0 - 21e9
    lamb0 = c_const/nu0
    
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-10, 1040.95+10
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 8
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-fZeroField
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= -S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = S_V/C
    DeltaT = S_V/np.max(DC_V)
    
    DeltaT_Smoothderiv = gaussian_filter1d(S_V, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    

    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaT_Smoothderiv = DeltaT_Smoothderiv*1e6
    
    
    N = int(np.ceil(np.shape(DeltaT_Smoothderiv)[0]/2))
    ax[0].scatter(f, np.mean(DeltaT_Smoothderiv[N-M:N+M,:], 0), marker = '.', s = 5, color = 'k')
    
    # DeltaTMin = np.min([DeltaT_Smoothderiv.min()])
    # DeltaTMax = np.max([DeltaT_Smoothderiv.max()])
    
        
    # norm = TwoSlopeNorm(
    #     vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
    #     vcenter=0,
    #     vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    # )

    
    p = ax[2].pcolormesh(
        X, Y, DeltaT_Smoothderiv,
        cmap='seismic',
        norm=norm,
        shading='auto',
        rasterized=True   
    )
    
    ax[1].xaxis.set_label_position('top')
    ax[1].tick_params(axis='x', which='both', top=True, bottom=True, labeltop = True, labelbottom = False)
    ax[1].set_xlabel(ax[2].get_xlabel(), fontsize = labelfontsize -1, labelpad = 6)
    ax[2].tick_params(axis='x', which='both', top=True, bottom=True, labeltop = False, labelbottom = False)
    ax[2].set_xlabel(r'')
    
    ax[0].tick_params(axis='x', which='both', top=True, bottom=True, labeltop = False, labelbottom = True)
    ax[2].set_xlabel(r'')
    
    
    
    
    # cbar = fig.colorbar(p, ax=ax, location='bottom', pad = 0.05)
    # cbar.set_label(r'lock-in signal, $S_{\text{LI}}$ ($\upmu$V)', fontsize = labelfontsize-1)
    # cbar.set_ticks([DeltaT_Smooth.min(), DeltaT_Smoothderiv.min(),0, DeltaT_Smoothderiv.max(),DeltaT_Smooth.max()])
    
    
    # cbarticks = cbar.get_ticks()
    # cbar.set_ticklabels([rf"${val:.0f}$" for val in cbarticks], fontsize = numfontsize)
    # axw = 0.2/0.3528
    # cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    # cbar.ax.tick_params(
    #     direction='in',   # ticks inward
    #     width=axw,          # tick line width
    #     length=12         # tick length (optional)
    # )
    # labels = cbar.ax.get_xticklabels()

    # labels[1].set_visible(False)
    # labels[3].set_visible(False)

    # print('Max experiment: ', np.max(DeltaT_Smooth))
    


    x0 = 0
    for axx in ax[1::]:
        axx.tick_params(
            axis = 'x',
            direction="out",
        )
        axx.tick_params(
            axis = 'y',
            direction="out",
        )
        axx.plot([-20, 20], [0,0], 'k--', lw = 0.2/0.3528)
        axx.set_ylim(-21, 21)
        axx.set_xlim(-10, 10)
        axx.set_xticks([-10, -5, 0, 5, 10])
    

    if saveBool:
        # fig.savefig(
        #     (_FIGURES_DIR + "/") + "CompareExpToTheory2D.png", dpi=600, bbox_inches="tight",transparent=True
        # )
        
        fig.savefig(
            (_FIGURES_DIR + "/") + "CompareLockInResultsZeroField.svg", bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotMicrodiskTheorySensitivityDeltaTvsMWDetSI(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.95, PRQuantumWidthHalf * 1.5, 
       r'MW detuning, $\Delta \omega_{\text{MW}}/2\pi$ (MHz)', 
       [r"", r""],
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True,  widthRatios = [1], heightRatios = [1,4]
    )
    
    # define cavity parameters
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    
    
    # define optical frequency detuning array
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0 - 11e9
    lamb0 = c_const/nu0
    
    nuBW = 100e9 #nm
    nu = np.linspace(nu0 - nuBW, nu0 + nuBW, 1000)
    lamb = c_const/nu
    
    # define MW frequency detuning array
    fMW0 = 2776.6e6*0
    fMWBW = 5e6
    fMWmax = -fMWBW
    fMWmin = fMWBW
    fMW = np.linspace(fMWmin, fMWmax, 1000)
    
    
    ODMRLin = microwaveDetuning3(fMW, 1e6, 0, 1, 0)
    ODMR = microwaveDetuningDeriv3(fMW, 1e6, 0, 1, 0)*1e6
    
    g = 0
    ng = 2.41
    tau =1.004524991429497*np.exp(1j* -0.057891967231220)
    kex = nu0/2.509576482281477e+04
    kex = nu0/1.5e4
    kabs = alpha * c_const/ng
    ki = nu0/6.763434893840459e+03 - kabs
    kabs0 = alpha0 * c_const/ng
    kabs1 = alpha1 * c_const/ng
    
    k0 = ki + kabs0
    k1 = ki + kabs1
    
    DeltaNu = nu - nu0 
    # print(nu0/ki1)
    T0 = calcTransmissionSinglet(kex, k0, DeltaNu, tau = tau)
    T1 = calcTransmissionSinglet(kex, k1, DeltaNu, tau = tau)
    DeltaT = T1-T0
    
    DeltaTMat = np.outer(DeltaT, ODMR)
    
    fMWmat, DeltaNumat = np.meshgrid(fMW, DeltaNu)
    

    # print('Max theory: ', np.max(DeltaTMat)*1e4)
    
    
    # load and plot data
    filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    # filePath = filepathODMR+ filenameODMR4p4umBiasLI1
    # filePath = filepathODMR+ filenameODMR4p4umBiasLI2
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-1, 1040.95+1
    # lambmin, lambmax = 1040, 1043
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    # S_V = data['signal'].item()
    # C = data['contrast'].item()
    # signalAbs= signalAbs[:,filt]/np.max(signalAbs/contrast)*1e4
    # f = data['freqsODMR'].item()*1e-6-2776.76
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 2
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-2776.7
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = S_V/C
    DeltaT = -S_V/np.max(DC_V)
    
    DeltaT_Smooth = gaussian_filter1d(DeltaT, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    

    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaTMat, DeltaT_Smooth = DeltaTMat*1e4, DeltaT_Smooth*1e4
    
    
    
    DeltaTMin = np.min([DeltaT_Smooth.min(), DeltaTMat.min()])
    DeltaTMax = np.max([DeltaT_Smooth.max(), DeltaTMat.max()])
    
        
    norm = TwoSlopeNorm(
        vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
        vcenter=0,
        vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    )

    
    p = ax[0].pcolormesh(
        X, Y, DeltaT_Smooth,
        cmap='seismic',
        norm=norm,
        shading='auto'
    )
    
    # fig.colorbar(p, ax=ax[0])
    
    # norm = TwoSlopeNorm(
    #     vmin=DeltaTMat.min(),
    #     vcenter=0,
    #     vmax=DeltaTMat.max()
    # )

    p = ax[1].pcolormesh(
        fMWmat/1e6, DeltaNumat/1e9, DeltaTMat,
        cmap='seismic',
        norm=norm,
        shading='auto'
    )
    cbar = fig.colorbar(p, ax=ax, location='top')
    cbar.set_label(r'$\Delta T\,(\times10^{-4})$', fontsize = labelfontsize)
    cbar.set_ticks([DeltaTMat.min(),DeltaT_Smooth.min(),0,DeltaT_Smooth.max(), DeltaTMat.max()])
    
    
    cbarticks = cbar.get_ticks()
    cbar.set_ticklabels([rf"${val:.1f}$" for val in [DeltaTMat.min(),DeltaT_Smooth.min(),0,DeltaT_Smooth.max(), DeltaTMat.max()]], fontsize = numfontsize)
    axw = 0.2/0.3528
    cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    cbar.ax.tick_params(
        direction='in',   # ticks inward
        width=axw,          # tick line width
        length=14         # tick length (optional)
    )
    labels = cbar.ax.get_xticklabels()

    labels[1].set_visible(False)
    labels[3].set_visible(False)

    # print('Max experiment: ', np.max(DeltaT_Smooth))
    # print(np.max(DeltaT_Smooth)/np.max(DeltaTMat)*1e4)
    
    
    
    # plt.tight_layout()
    
    axs = ax[0]
    
    axs.set_ylim([-nuBW*1e-9, nuBW*1e-9])
    axs.set_ylim([-25, 25])
    axs.set_yticks([-25, 0, 25])

    x0 = 0
    for axx in ax:
        axx.tick_params(
            axis = 'x',
            direction="out",
        )
        axx.tick_params(
            axis = 'y',
            direction="out",
        )
        axx.set_xlim(-4.85, 4.85)
        
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "CompareExpToTheory2D.png", dpi=600, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    


    
def plotLockinExample(saveBool = False):
    fig, axx = plot_custom(
       PRQuantumWidthFull*0.28, PRQuantumWidthHalf*0.75, 
       [ r"$ \omega_{\text{MW}}/2\pi$ (GHz)", r"$\omega_{\text{MW}}/2\pi$ (GHz)"], 
       [r'$\Delta T$ (au)', r'$\Delta T$ (au)'],
       labelPadX = 3, labelPadY = 1, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =True, widthRatios=[1], heightRatios=[1,1]
    )
    
    fBW = 10e6
    fmin = -fBW
    fmax = fBW
    f = np.linspace(fmin, fmax, 1000)
    gammaMW = nu/1.5
    
    TMW = microwaveDetuning1(f, gammaMW, 0, 0.5, 0)
    TMWOffset = microwaveDetuning1(f, gammaMW, -8e6, 0.5, 0)
    TMWOffset /= TMW.max()
    TMW /= TMW.max()
    
    yLI = np.linspace(0, 1.1, 1000)
    freq = 3.5  # number of periods along the vertical axis
    xLI = square(2 * np.pi * freq * yLI)
    xLI = 4* xLI -4 # width of the square wave
    
    xLI2 = np.linspace(0, 10, 1000)
    yLI2 = square(2 * np.pi * freq/10.1 * xLI2)
    yLI2 =  0.49*yLI2+0.51  # width of the square wave
    
    # ax.plot(lamb*1e9, fits, lw = 0.2/0.3528, color = 'k')
    Tmin = 0.3
    ax = axx[0]
    ax.plot(f/1e6, TMW, lw = 0.2/0.3528, color = 'k')
    # ax.plot(f/1e6, TMWOffset, lw = 0.2/0.3528, color = 'k', linestyle = '--', alpha = 0.8)
    ax.fill_between(f/1e6, TMW, TMW*0,
                    color = 'k', alpha = 0.2, lw = 0)
    # ax.plot(xLI, yLI,lw = 0.2/0.3528, color = colorRed)
    # ax.fill_between([xLI.min(), xLI.max()], [1.1,1.1], [0,0],
    #                 color = colorRed, alpha = 0.2, lw = 0)
    ax.fill_between([xLI2.min(), xLI2.max()], [1,1], [yLI2.min(),yLI2.min()],
                    color = colorBlue, alpha = 0.2, lw = 0)
    ax.plot(xLI2, yLI2,lw = 0.2/0.3528, color = colorBlue)
    
    ax.yaxis.set_label_coords(-0.18, 1.1)
    
    ax.set_xlim([fmin/1e6-2,fmax/1e6])
    ax.spines['bottom'].set_bounds(fmin/1e6,fmax/1e6)
    
    ax.set_ylim([-0.15, 1.1])
    ax.spines['left'].set_bounds(0,TMW.max())
    ax.set_xticks([0])
    ax.set_xticklabels([r'$\Omega_0$'])
    ax.set_yticks([0, 1])
    # ax.set_xticklabels([rf"${val:.0f}$" for val in ax.get_xticks()])
    ax.set_yticklabels([r'$0$', r'$1$'])
    ax = axx[1]
    
    fBW = 3e6
    fmin = -fBW+1e6
    fmax = fBW
    f = np.linspace(fmin, fmax, 1000)
    TMW = microwaveDetuning1(f, gammaMW, 0, 0.5, 0)
    TMWOffset1 = microwaveDetuning1(f, gammaMW, -0.3e6, 0.5, 0)
    TMWOffset2 = microwaveDetuning1(f, gammaMW, +0.3e6, 0.5, 0)
    TMWOffset1 /= TMW.max()
    TMWOffset2 /= TMW.max()
    TMW /= TMW.max()
    
    yLI = np.linspace(0, 0.6, 1000)
    freq = 8.4  # number of periods along the vertical axis
    xLI = np.cos(2 * np.pi * freq * yLI)
    xLI = 0.3* xLI -0.7 # width of the square wave
    
    xLI2 = np.linspace(0, 4, 1000)
    yLI2 = -np.cos(2 * np.pi * freq/6 * xLI2)
    yLI2 =  0.18*yLI2+0.78  # width of the square wave
    xLI2 = xLI2 - 0.42
    
    ax.plot(f/1e6, TMW, lw = 0.2/0.3528, color = 'k')
    ax.plot(f/1e6, TMWOffset1, lw = 0.2/0.3528, color = 'k', linestyle = '--', alpha = 0.8)
    ax.plot(f/1e6, TMWOffset2, lw = 0.2/0.3528, color = 'k', linestyle = '--', alpha = 0.8)
    ax.fill_between(f/1e6, TMW, TMW*0,
                    color = 'k', alpha = 0.2, lw = 0)
    ax.fill_between([xLI.min(), xLI.max()], [yLI.max(), yLI.max()], [0,0],
                    color = colorRed, alpha = 0.2, lw = 0)
    ax.fill_between([xLI2.min(), xLI2.max()], [yLI2.max(), yLI2.max()], [yLI2.min(),yLI2.min()],
                    color = colorBlue, alpha = 0.2, lw = 0)
    
    
    ax.plot(xLI, yLI,lw = 0.2/0.3528, color = colorRed)
    ax.plot(xLI2, yLI2,lw = 0.2/0.3528, color = colorBlue)
    
    # set ylabel positon
    ax.yaxis.set_label_coords(-0.18, 1.1)
    
    ax.set_xlim([fmin/1e6-0.7,fmax/1e6])
    ax.spines['bottom'].set_bounds(fmin/1e6,fmax/1e6)
    
    ax.set_ylim([-0.15, 1.1])
    ax.spines['left'].set_bounds(0,TMW.max())
    # ax.set_xticks()
    ax.set_yticks([0, 1])
    ax.set_xticks([0, hyperfine/1e6])
    ax.set_xticklabels([r'$\Omega_0$', r'$\Omega_0+\Omega_{\text{HF}}$'])
    ax.set_yticklabels([r'$0$', r'$1$'])
    
    for ax in axx:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
        # ax.set_xlim([-11,10])
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal
        ax.xaxis.set_label_coords(0.55,-0.27)
        
    axx[0].yaxis.set_label_coords(-0.16, 1.02)
    
    axx[1].yaxis.set_label_coords(-0.15, 1.02)
    
    
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "LockInTechniques.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plotMicrodiskTheorySensitivityDeltaTvsMWDeHighQ(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*1/3, PRQuantumWidthHalf * 1, 
       r'$\Delta \omega_{\text{MW}}/2\pi$ (MHz)', 
       [r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)', r'$\Delta \omega_{\text{opt}}/2\pi$ (GHz)'],
       labelPadX = 5, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize,
       widthBool =True,  widthRatios = [1], heightRatios = [1,2]
    )
    
    Veff = calcVeffHighQ()
    
    # define cavity parameters
    kabs0 = c_const/nd*alpha0
    kabs1 = c_const/nd*alpha1
    kabs = (kabs0 + kabs1)/2
    
    
    # define optical frequency detuning array
    lamb0 = 1041.319e-9
    nu0 = c_const/lamb0 
    lamb0 = c_const/nu0
    
    N = 50
    nuBW = 50e9 #nm
    nu = np.linspace(nu0 - nuBW, nu0 + nuBW, N*2)
    lamb = c_const/nu
    
    # define MW frequency detuning array
    fMW0 = 2776.6e6*0
    fMWBW = 5e6
    fMWmax = -fMWBW
    fMWmin = fMWBW
    fMW = np.linspace(fMWmin, fMWmax, N)

    
    ODMR = microwaveDetuning3(fMW, 1.85e6, 0, 1, 0)
    ODMR /= ODMR.max()
    
    ng = 2.4
    g = 9.8e9
    tau = np.exp(1j* -0.22)
    kex1 = nu0/1.8e5
    kex2 =  nu0/1.4e5
    ki1 = nu0/45e3
    ki2 = nu0/38e3
        
    g = 9e9
    tau = np.exp(1j* -0.22)
    kex1 = nu0/5e5
    kex2 =  kex1
    ki1 = nu0/30e3
    ki2 = nu0/30e3
    
    g = 8.29195685e+09
    tau = 5.42964177e-01*np.exp(1j* -2.17800024e-01)
    kex1 = 6.29210481e+08
    kex2 =  1.01006776e+09
    ki1 = 8.85624557e+09
    ki2 = 1.16488837e+10
    
    popt = ( [ 6.29210481e+08 , 1.01006776e+09 , 8.85624557e+09,  1.16488837e+10,
      8.29195685e+09,  5.42964177e-01, -2.17800024e-01 , 2.41010163e-05,
      7.79856434e-01])
    
    popt = ([1439358456.3236377, 1799198070.4045472, 7575570822.755988, 8224905464.706502, 9000000000.0, 
             0.9999999999999999, -0.3, 4.120953851485148e-06, 0.3])
    
    popt = [1439358456.3236377, 1799198070.4045472, 7575570822.755988, 8224905464.706502, 9000000000.0, 
            0.9999999999999999, -0.2, 7.4177169326732685e-06, 1]
    
    # kabs = alpha * c_const/ng
    # ki1 = ki1 - kabs
    # ki2 = ki2 - kabs
    # kabs0 = alpha0 * c_const/ng
    # kabs1 = alpha1 * c_const/ng
    
    # k0_1 = ki1 + kabs0
    # k0_2 = ki2 + kabs0
    # k1_1 = ki1 + kabs1
    # k1_2 = ki2 + kabs1
    
    DeltaNu = nu - nu0 
    # # print(nu0/ki1)
    # T0 = calcTransmission(kex1, kex2, k0_1, k0_2, DeltaNu, g, tau)
    # T1 = calcTransmission(kex1, kex2, k1_1, k1_2, DeltaNu, g, tau)
    # DeltaT = T1-T0
    
    DeltaT, TMean= calcDeltaMicrodiskReturnT(DeltaNu, *popt, Veff = Veff)
    
    DeltaTMat = np.outer(DeltaT, ODMR)
    
    fMWmat, DeltaNumat = np.meshgrid(fMW, DeltaNu)
    

    # print('Max theory: ', np.max(DeltaTMat))
    
    
    # load and plot data
    # filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260127/ODMR_20260127_CW__16_18dBAtt_4.2um_0_1_2.87GHz_30MHz_002.mat'
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    filePath = filepathODMR+ filenameODMR4p4umBiasLI1
    # filePath = filepathODMR+ filenameODMR4p4umBiasLI2
    
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI1
    # filePath = filepathODMR+ filenameODMR4p2umBiasLI2
    
    filePath = filepathODMR+ filenameODMR4p4umBiasLI1
    
    
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    lambmin, lambmax = 1040.75-1, 1040.95+1
    W_nm = data['laserWLsRead'].item()
    W_nm = W_nm[0,:]
    filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)

    W_nm= W_nm[filt]
    # S_V = data['signal'].item()
    # C = data['contrast'].item()
    # signalAbs= signalAbs[:,filt]/np.max(signalAbs/contrast)*1e4
    # f = data['freqsODMR'].item()*1e-6-2776.76
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    if len(np.shape(S_V)) == 3:
        
        W_nm = data['laserWLsRead'].item()
        W_nm = W_nm[0,0,0,:]
        filt = np.logical_and(W_nm< lambmax, W_nm>lambmin)
        idx = 2
        S_V = S_V[:, idx, :]
        W_nm = W_nm[filt]
        C = C[:, idx, :]
    S_V = S_V[:,filt]
    C = C[:,filt]
    
    
    power = data['powerIn'].item()
    print('Power into the fibre High Q: ',np.mean(power)*50)
    
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-2776.7
    
    
    filt = np.logical_and(f< fMWBW*1e-6, f>-fMWBW*1e-6)
    
    f= f[filt]
    S_V= S_V.T[:, filt]
    C = C.T[:, filt]
    DC_V = S_V/C
    DeltaT = -S_V/np.max(DC_V)
    
    DeltaT_Smooth = gaussian_filter1d(DeltaT, sigma=20, axis=0)
    
    nuMeasGHz = (c_const/(W_nm*1e-9)- nu0)*1e-9
    
    
    X, Y = np.meshgrid(f, nuMeasGHz)
    
    cmap = plt.cm.hot
    truncated_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc_hot',
        cmap(np.linspace(0, 0.75, 256))
    )
    
    
    # compare the max of experiment and theory and choose the cmap of the bigger range\
    DeltaTMat, DeltaT_Smooth = DeltaTMat*1e4, DeltaT_Smooth*1e4
    
    
    
    DeltaTMin = np.min([DeltaT_Smooth.min(), DeltaTMat.min()])
    DeltaTMax = np.max([DeltaT_Smooth.max(), DeltaTMat.max()])
    
        
    norm = TwoSlopeNorm(
        vmin=-np.max(np.abs([DeltaTMin, DeltaTMax])),
        vcenter=0,
        vmax=np.max(np.abs([DeltaTMin, DeltaTMax]))
    )

    
    p = ax[0].pcolormesh(
        X, Y, DeltaT_Smooth,
        cmap='seismic',
        norm=norm,
        shading='auto',
        rasterized=True   
    )
    
    # fig.colorbar(p, ax=ax[0])
    
    # norm = TwoSlopeNorm(
    #     vmin=DeltaTMat.min(),
    #     vcenter=0,
    #     vmax=DeltaTMat.max()
    # )

    p = ax[1].pcolormesh(
        fMWmat/1e6, DeltaNumat/1e9, DeltaTMat,
        cmap='seismic',
        norm=norm,
        shading='gouraud',
        rasterized=True   
    )
    cbar = fig.colorbar(p, ax=ax, location='top')
    cbar.set_label(r'$\Delta T\,(\times10^{-4})$', fontsize = labelfontsize-1)
    cbar.set_ticks([DeltaTMat.min(),DeltaT_Smooth.min(),0,DeltaT_Smooth.max(), DeltaTMat.max()])
    
    
    cbarticks = cbar.get_ticks()
    cbar.set_ticklabels([rf"${val:.1f}$" for val in [DeltaTMat.min(),DeltaT_Smooth.min(),0,DeltaT_Smooth.max(), DeltaTMat.max()]], fontsize = numfontsize)
    axw = 0.2/0.3528
    cbar.outline.set_linewidth(axw)   # e.g. 2 pt
    cbar.ax.tick_params(
        direction='in',   # ticks inward
        width=axw,          # tick line width
        length=8         # tick length (optional)
    )
    labels = cbar.ax.get_xticklabels()

    labels[1].set_visible(False)
    labels[3].set_visible(False)

    # print('Max experiment: ', np.max(DeltaT_Smooth))
    
    
    
    # plt.tight_layout()
    
    axs = ax[0]
    
    # axs.set_ylim([-nuBW*1e-9, nuBW*1e-9])
    # axs.set_ylim([-25, 25])
    # axs.set_yticks([-25, 0, 25])

    x0 = 0
    for axx in ax:
        axx.tick_params(
            axis = 'x',
            direction="out",
        )
        axx.tick_params(
            axis = 'y',
            direction="out",
        )
        # axx.set_xlim(-4.85, 4.85)
        
    
    if saveBool:
        # fig.savefig(
        #     (_FIGURES_DIR + "/") + "CompareExpToTheory2D.png", dpi=600, bbox_inches="tight",transparent=True
        # )
        
        fig.savefig(
            (_FIGURES_DIR + "/") + "CompareExpToTheory2DHighQ.svg", bbox_inches="tight",transparent=True
        )
    
    plt.show()

    
def plotNarrowTransmissionPaperHighQ(saveBool = False):
    fig, ax = plot_custom(
       PRQuantumWidthFull*0.2, 0.2*PRQuantumWidthFull, 
       r"$\lambda$ (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 1, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False, tickpad = 4
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.4um_0/1041nm_narrow_coupled_nf_12e-3_B14_4p4um_3uWintofibre_10db/'
    filenames = {'1.mat', '2.mat', '3.mat', '4.mat', '5.mat'}
    
    W, DeltanuGHz, TOut = cleanUpAverageData(filePath, filenames, +12e-3)
    
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.4um_0/1041nm_narrow_coupled_nf_12e-3_B14_4p4um_3uWintofibre_10db/'
    filenames = {'greenOn.mat'}
    
    WGreen,DeltanuGHzGreen, TOutGreen = cleanUpAverageData(filePath, filenames, -12e-3)
    
    
    # params = dataParams['params']
    lambmin = 1041.3  
    lambmax = 1041.5
    filt = np.logical_and(W< lambmax, W>lambmin)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]/0.95
    # ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = 1, zorder = 10, color = colorDarkRed)
    # ax.scatter(WGreen[1::skip], TOutGreen[1::skip], edgecolors='none', marker = '.', s = 1, zorder = 10, color = colorBlue)
    
    plt.tight_layout()
    (Tmin, Tmax) = ax.get_ylim()
    
    lamb = np.linspace(lambmin, lambmax, 10000)*1e-9
    nu = c_const/lamb
    lamb0 = 1041.398e-9
    nu0 = c_const/lamb0
    
    g = 9.8e9
    tau = np.exp(1j* -0.22)
    kex1 = nu0/1.8e5
    kex2 =  nu0/1.4e5
    ki1 = nu0/45e3
    ki2 = nu0/38e3
    
    
    g = 9.8e9
    tau = np.exp(1j* -0.22)
    kex1 = nu0/10e5
    kex2 =  nu0/10e5
    ki1 = nu0/50e3
    ki2 = nu0/50e3

    Delta = nu - nu0 
    print(nu0/ki1)
    fit= calcTransmission(kex1, kex2, ki1, ki2, Delta, g = g, tau = tau)
    
    
    
    ax.plot(lamb*1e9, fit, lw = 0.2/0.3528, color = colorRed)
    Tmin = 0.3
    # ax.set_xlim([lambmin,lambmax])
    # ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    # ax.set_ylim([Tmin,1])
    # ax.spines['left'].set_bounds(Tmin,1)
    # ax.set_xticks([lamb0*1e9])
    # ax.set_yticks([Tmin, 1])
    ax.set_xticklabels([rf"${val:.1f}$" for val in ax.get_xticks()])
    # ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
    
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=True, bottom=False, labeltop = True, labelbottom = False)
    ax.spines["bottom"].set_visible(True)
    # ax.yaxis.set_label_coords(-0.3, 0.55)
    # ax.xaxis.set_label_coords(0.58, -0.25)
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "NarrowTransmission_4p2um_paper.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
    

    
def plotCWOptimization(saveBool = False):
    
    fig, axArr = plot_custom(
       PRQuantumWidthHalf, PRQuantumWidthHalf*1.6, 
       [r"green laser power, $P_{\text{p}}$ (mW)", r"Rabi frequency, $\Omega_{{\text{{R}}}}/2\pi$ (MHz)", r"IR laser power, $P_{\text{s}}$ ($\upmu$W)"], 
       [r"$C/\nu$ (au)", r"$C/\nu$ (au)", r"$\sqrt{P_{\text{s}}}C/\nu$ (au)"] #"coincidence rate (mHz/mW)"
       , labelPadX = 5, labelPadY = 6, commonY = False, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 1,nrow = 3, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    

    
    directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/lab-analysis-software/barclay-lab/analysisScripts/ODMR/DNVODMR/CW_IR/'

    filenameFit = 'CW_ODMR_fit_results.csv';
    filenameModel = 'CW_ODMR_fit_models.csv';
    filenameSignal = 'CW_ODMR_signal_data.csv';

    dfFit = pd.read_csv(directory + filenameFit)
    dfODMR = pd.read_csv(directory + filenameSignal)
    dfODMRModel = pd.read_csv(directory + filenameModel)
    
    etaGreenFStoFibreTaper = 0.5*0.1
    
    powerGreen = dfFit['Power_uW'].values/1e6*etaGreenFStoFibreTaper#*np.sqrt(0.15)
    powerGreenErr = dfFit['PowerErr_uW'].values/1e6*etaGreenFStoFibreTaper#*np.sqrt(0.15)
    v_kHz = dfFit['v_kHz'].values
    vMin_kHz = dfFit['vMin_kHz'].values
    vMax_kHz = dfFit['vMin_kHz'].values
    C_au = dfFit['C_au'].values
    CMin_au = dfFit['CMin_au'].values
    CMax_au = dfFit['CMax_au'].values
    
    vErr = (np.abs(v_kHz-vMin_kHz) + np.abs(v_kHz-vMax_kHz))/2;
    CErr = (np.abs(C_au-CMin_au) + np.abs(C_au-CMax_au))/2;
    CovErr = np.sqrt(CErr**2 / v_kHz**2 + vErr**2 * C_au**2/v_kHz**4) 
    

    
    numSamples = len(powerGreen)
    colorArray = sns.color_palette(palette='Reds', n_colors=int(np.ceil(1*numSamples))+5)
    colorArray = colorArray[4:-1]
    
    rat = np.abs(C_au/v_kHz)*1e6
    # Choose a colormap for ax4
    cmap = plt.cm.viridis  # or 'plasma', 'coolwarm', etc.
    
    # Normalize 'rat' values to 0–1 for colormap mapping
    norm = plt.Normalize(vmin=0.05, vmax=0.22)
    s = ms*4
    for i in range(numSamples):
        axArr[0].errorbar(powerGreen[i], rat[i]/0.5*0.9, yerr=CovErr[i]*1e6, xerr = powerGreenErr[i],
                    capthick=lw/2, elinewidth=lw/2, capsize=2, fmt='*', markersize = s, color = colorGreen,  mec='None')

        
    
    filenameFit = 'CW_ODMR_MW_fit_results.csv';
    filenameModel = 'CW_ODMR_MW_fit_models.csv';
    filenameSignal = 'CW_ODMR_MW_signal_data.csv';

    dfFit = pd.read_csv(directory + filenameFit)
    dfODMR = pd.read_csv(directory + filenameSignal)
    dfODMRModel = pd.read_csv(directory + filenameModel)
    
    power = dfFit['Power_uW'].values/-1e6
    powerErr = dfFit['PowerErr_uW'].values
    
    powerMax0dBAtt =  10**3.086
    powermW = 10**((30.86-power)/10)
    addAtt = -2
    power2Rabi = lambda powermW: 6.5 * np.sqrt(powermW/powerMax0dBAtt) * 10**(addAtt/20)# / 12
    
    
    power = dfFit['Power_uW'].values/-1e6
    
    addAtt = 2+8+3
    power2Rabi = lambda power: 6.5 * 10**(-1*(power + addAtt)/20)# / 12
    
    # convert power which is attenuation to Rabi frequency
    
    fRabi = power2Rabi(power)
    fRabiErr = fRabi *0.05 
    
    v_kHz = dfFit['v_kHz'].values
    vMin_kHz = dfFit['vMin_kHz'].values
    vMax_kHz = dfFit['vMin_kHz'].values
    C_au = dfFit['C_au'].values
    CMin_au = dfFit['CMin_au'].values
    CMax_au = dfFit['CMax_au'].values
    
    
    vErr = (np.abs(v_kHz-vMin_kHz) + np.abs(v_kHz-vMax_kHz))/2;
    CErr = (np.abs(C_au-CMin_au) + np.abs(C_au-CMax_au))/2;
    CovErr = np.sqrt(CErr**2 / v_kHz**2 + vErr**2 * C_au**2/v_kHz**4) 
    
    rat = np.abs(C_au/v_kHz)*1e6
    numSamples = len(power)
    colorArray = sns.color_palette(palette='Reds', n_colors=int(np.ceil(1*numSamples))+4)
    colorArray = colorArray[3:-1]
    for i in range(numSamples):
        axArr[1].errorbar(fRabi[i],rat[i]/0.2,  yerr=CovErr[i]*1e6, xerr = fRabiErr[i],
                    capthick=lw/2, elinewidth=lw/2, capsize=2, fmt='*', markersize = s, color = colorBlue,  mec='None')
        


    
    filenameFit = 'CW_ODMR_IR_fit_results.csv';
    filenameModel = 'CW_ODMR_IR_fit_models.csv';
    filenameSignal = 'CW_ODMR_IR_signal_data.csv';

    dfFit = pd.read_csv(directory + filenameFit)
    
    lamb0 = 1040.77393e-9
    nu0 = c_const/lamb0
    omega0 = nu0*2*np.pi
    
    kex = nu0/2.509576482281477e+04
    kc = nu0/6.763434893840459e+03
    
    ng = nd
    Veff = 18*(lamb0/ng)**3
    
    powerW = dfFit['Power_uW'].values*1e-6
    powerErrW = dfFit['PowerErr_uW'].values*1e-6
    Ncav = P2ResNcav(powerW, etaFibreIR, omega0, kex, kc)
    I = Ncav2I(Ncav, omega0, ng, Veff)
    # convert power which is attenuation to Rabi frequency
    
    v_MHz = dfFit['v_kHz'].values/1000
    vMin_MHz = dfFit['vMin_kHz'].values/1000
    vMax_MHz = dfFit['vMin_kHz'].values/1000
    C_au = dfFit['C_au'].values*100
    CMin_au = dfFit['CMin_au'].values*100
    CMax_au = dfFit['CMax_au'].values*100
    
    vErr = (np.abs(v_MHz-vMin_MHz) + np.abs(v_MHz-vMax_MHz))/2;
    CErr = (np.abs(C_au-CMin_au) + np.abs(C_au-CMax_au))/2;
    CovErr = np.sqrt(CErr**2 / v_MHz**2 + vErr**2 * C_au**2/v_MHz**4)*np.sqrt(powerW) 
    rat = np.abs(C_au/v_MHz*np.sqrt(powerW*1e6))*1e3
    numSamples = len(powerW)
    colorArray = sns.color_palette(palette='Reds', n_colors=int(np.ceil(1*numSamples))+4)
    colorArray = colorArray[3:-1]
    for i in range(numSamples):
        axArr[2].errorbar(powerW[i]*1e6,rat[i]/0.06,  yerr=powerErrW[i]/powerW[i]*rat[i]/0.06, xerr = powerErrW[i]*1e6,
                    capthick=lw/2, elinewidth=lw/2, capsize=2, fmt='*', markersize = s, color = colorRed,  mec='None')
        

        
    N = 101
    OmegaRmin = np.log10(0.01e6*2*np.pi)
    OmegaRmax = np.log10(1e6*2*np.pi)
    I_pmin = 7
    I_pmax =9.2
    I_smin = 8
    I_smax = 10.38
    
    C, power, nu, I_IR, p7On, p7Off =     solvePopulationIR(saveBool = saveBool,
                                    N_IR = N,
                                    I_smin = I_smin,
                                    I_smax = I_smax,
                                    OmegaR = OmegaROptRadps,
                                    I_p =I_pOpt,
                                    plotBool = False)

    signal = C/nu*np.sqrt(power)
    signal = signal/np.max(signal)
    axArr[2].plot(I_IR/I_IR.max()*80, signal,zorder = -1,color=colorDarkRed, lw = lw)
        
    I_g, C, nu =    solvePopulationGreen(saveBool = saveBool,
                                    N_Green = N,
                                    I_pmin =I_pmin,
                                    I_pmax = I_pmax,
                                    OmegaR = OmegaROptRadps,
                                    I_s = I_sOpt,
                                    plotBool = False)

    signal = C/nu
    signal = signal/np.max(signal)

    axArr[0].plot(I_g/np.max(I_g)*np.max(powerGreen),signal ,zorder = -1,color=colorDarkGreen, lw = lw)
        
    OmegaRArr, C, nu =    solvePopulationOmegaR(saveBool = saveBool,
                                    N_OmegaR = N,
                                    OmegaRmin = OmegaRmin,
                                    OmegaRmax = OmegaRmax,
                                    I_s = I_sOpt,
                                    I_p =I_pOpt,
                                    plotBool = False)

    signal = C/nu
    signal = signal/np.max(signal)
    axArr[1].plot(OmegaRArr/2/np.pi/1e6, signal,zorder = -1, lw = lw,color=colorDarkBlue)
    
    plt.tight_layout()

    for ax in axArr:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
        ax.set_xlim([0,None])
        ax.set_ylim([0,None])
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position('bottom')

    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "CWOptimizationIR" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plotDetuningExampleHighQ(saveBool = False):
    fig, axx = plot_custom(
       PRQuantumWidthFull*1/4, PRQuantumWidthHalf*0.9, 
       [r"$\Delta\omega_{\text{opt}}/2\pi$ (GHz)", r"$\Delta \omega_{\text{MW}}/2\pi$ (MHz)"], 
       [r'', r'$\Delta T$ ($\times 10^{-4}$)'],
       labelPadX = 3, labelPadY = 1, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False, widthRatios=[1,1], heightRatios=[1,1]
    )
    
    Veff = calcVeffHighQ()
    
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsResonanceSweeps/DNV_B14_QG3/4.4um_0/1041nm_narrow_coupled_nf_12e-3_B14_4p4um_3uWintofibre_10db/'
    filenames = {'1.mat', '2.mat', '3.mat', '4.mat', '5.mat'}
    
    W_nm, DeltanuGHz, TOut = cleanUpAverageData(filePath, filenames, -3.5e-3)
    TOut *= 1.05
    #%%%% Plot transmission data 
    # W_nm, DeltanuGHz, TOut = loadAndFilterTransmissionData(filePath)
    skip = 5
    lamb0 = 1041.41 * 1e-9
    DeltanuGHz -= c_const/lamb0*1e-9
    axx[0].scatter(DeltanuGHz[1::skip]-3, TOut[1::skip], edgecolors='none', marker = '.', s = 2, zorder = 10, color = colorDarkRed)
    
    filePath = filepathODMR+ filenameODMR4p4umBiasLI1
    
    nu0 = c_const/lamb0
    nuBW = 40e9 
    f0GHz = 2776.7
    
    #%%%% Plot ODMR line data 
    W_nm, S_V, DeltaT, C, P_in, DeltanuGHz, DeltaW_nm, fExp, DC_V = loadODMRData(filePath, lamb0nm = lamb0 * 1e9, 
                     lambbwnm = 2, f0GHz = f0GHz, nuOptbwGHz = nuBW*1e-9, fbwGHz = 10, 
                     powerScaling = 99/2, boolFreqFiltering = True, idx = 2, bgOffset = 3e-3)
    
    DC_VMean = np.mean(DC_V, 1)
    DC_VMean /= DC_VMean.max()
    start = 300
    # axx[0].scatter(DeltanuGHz[start:-start:skip]-26.2, DC_VMean[start:-start:skip], edgecolors='none', marker = '.', s = 2, zorder = 10, color = colorDarkRed)
    
    
    DeltaT_Smooth = gaussian_filter1d(DeltaT, sigma=1, axis=0)
    
    idx = 49
    scale = 1000
    skip = 20
    x = (c_const/W_nm/1e-9 - nu0)*1e-9
    y = DeltaT_Smooth[:, idx]*2 # peak to p2p
    z = DC_V[:, idx]
    axx[0].scatter(x[1::skip]-24.5, y[1::skip]*scale, edgecolors='none', marker = '.', s = 5, zorder = 10, color = colorDarkBlue)
    DeltaTMaxExp = y.max()*scale
    
    (Tmin, Tmax) = axx[0].get_ylim()

    #%%%% Plot transmission fit
    
    numin = nu0 - nuBW
    numax = nu0 + nuBW
    nuArr = np.linspace(numin, numax, 100)
    
    g = 9e9
    tau = np.exp(1j* -0.2)
    kex1 = nu0/2e5
    kex2 =  nu0/1.2e5
    ki1 = nu0/36e3
    ki2 = nu0/35e3
    
    # g = 9e9
    # tau = np.exp(1j* -0.3)
    # kex1 = nu0/0.5e5
    # kex2 =  nu0/0.5e5
    # ki1 = nu0/60e3
    # ki2 = nu0/60e3
    
    # g = 8.5e9
    # tau = np.exp(1j* -0.1)
    # kex1 = nu0/5e5
    # kex2 =  nu0/5e5
    # ki1 = nu0/70e3
    # ki2 = nu0/70e3
    
    Delta = nuArr - nu0 
    
    deltaOmegaGHz = Delta*1e-9
    deltaOmegaGHzMin = deltaOmegaGHz.min()
    deltaOmegaGHzMax = deltaOmegaGHz.max()
     
    # print(nu0/ki1)
    fit= calcTransmission(kex1, kex2, ki1, ki2, Delta, g = g, tau = tau)
    
    #%%%% Plot change in transmission fit with wavelength
    P_in *= np.sqrt(etaFibreIR)*1.4/1.01/2
    print('Pin: ', P_in.mean())
    p0 = [kex1, kex2, ki1, ki2, g, np.abs(tau), np.angle(tau),P_in.mean(), 1] # doublet forms standing mode which has twice the peak intensity
    # p0 = [  7.97161704e+08,  2.08375770e+09,  9.19466176e+09 , 4.09822761e+10,
    #   8.78057466e+09 , 6.27425160e-01, -1.01660594e-01,  P_in.mean(),
      # 1.7]
    # p0 = [kex1, kex2, ki1, ki2, g, np.abs(tau), np.angle(tau),P_in.mean()*1, 1]
    DeltaTFitP0, TMean = calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *p0, Veff = Veff, doubletBool=True)
    popt=p0
    upperScale = 100
    lowerScale = 0.001
    # lower_bounds = [kex1*lowerScale, kex2*lowerScale, ki1*lowerScale, ki2*lowerScale, g*lowerScale, np.abs(tau)*lowerScale, np.angle(tau)*1.02,P_in.mean()*0.1, 0.1]
    # upper_bounds = [kex1*upperScale, kex2*upperScale, ki1*upperScale, ki2*upperScale, g*upperScale, np.abs(tau)*upperScale, np.angle(tau)*0.98,P_in.mean()*3, 2]
    # popt, pcov = curve_fit(calcDeltaMicrodisk, x[1::skip]*1e9-24e9, y[1::skip], p0=p0,    bounds=(lower_bounds, upper_bounds))
    
    
    DeltaTFit, TMean= calcDeltaMicrodiskReturnT(deltaOmegaGHz*1e9, *popt, Veff = Veff, doubletBool = True)
    TMean/= TMean.max()
    
    [kex1, kex2, ki1, ki2, g, tauabs, phi, pin, scaleNew] = popt
    print('popt: ', popt)
    # print('Qi1, Qi2: ', omegas/ki1/2/np.pi, omegas/ki2/2/np.pi)
    tau = 1*np.exp(1j*phi)
    
    a = np.abs(calcIntracavityField(popt[0], popt[0] + popt[2], popt[1], popt[1]+ popt[3], Delta, popt[4]))
    NcavMax = calcPhotonsInCavity(a.max(), P_in.mean())
    print('Max Ncav: ', NcavMax)
    Imax = NtoI1042(NcavMax)
    print('Max intensity MW/cm^2: ', Imax)
    print('Qs: ', omegas/popt[2]/(2*np.pi), omegas/popt[3]/(2*np.pi))
    
    
    #%%%% Predict transmission based on change in transmission
    # TMean= calcTransmission(kex1, kex2, ki1, ki2, Delta, g = g, tau = tau)
    axx[0].plot(deltaOmegaGHz, TMean, lw = 0.2/0.3528, color = colorRed)
    
    Deltak = DeltaT2Deltak(DeltaTFit.max(), np.abs(tau), kex1, kex1 + ki1)
    print('Deltak MHz: ', Deltak.max()/1e6)
    print('delta alpha /m: ', kappa2alpha(Deltak.max()*2*np.pi))
    
    DeltaT0, T0, kabs0, kabs1, alphaOff, alphaOn =calcDeltaMicrodiskReturnT(np.array([0]), *popt, printBool = True, Veff =Veff)
    
    print('Deltak model MHz:', (kabs0- kabs1)/-1e6 )
    print('Delta alpha model /m: ', -kappa2alpha((kabs0- kabs1)*2*np.pi))
    print('Delta alpha model /m: ', alphaOn- alphaOff)
    print('alpha off model /m: ', kappa2alpha((kabs0)*2*np.pi))
    
    #%%%% Plot change in transmission fit with MW det
    fBW = 5e6
    fmin = -fBW
    fmax = fBW
    f = np.linspace(fmin, fmax, 1000)
    amp = DeltaTFit.max() * 1e4*0.74
    # print('amp: ',amp)
    TMW = microwaveDetuning3(f, nuOpt, 0, amp, 0)
    TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp)
    
    #%%%% Plot change in transmission with microwave detuning
    idx = np.unravel_index(np.argmax(DeltaT_Smooth), DeltaT_Smooth.shape)
    idx = idx[0]
    N = 5
    u = fExp-0.1
    v = np.mean(DeltaT_Smooth[idx-N:idx+N, :],0)*2 # peak to p2p
    # v /= v.max()
    # v *= TMW.max()
    skip = 2
    axx[1].scatter(u[1::skip]+0.1, v[1::skip]*1e4, edgecolors='none', marker = '.', s =5, zorder = 10, color = colorDarkBlue)
           
    p0 = [1e6, 0, 1e-4, 0]
    lower_bounds = [1e5, -1e6, 1e-7, -1]
    upper_bounds = [3e6, 1e6, 1e1, 1]
    P_det = P_in.mean()*TMean*np.sqrt(etaFibreIR)*etaPostFibre
    
    popt, pcov = curve_fit(microwaveDetuning3, u*1e6, v, p0=p0,    bounds=(lower_bounds, upper_bounds))
    perr = np.sqrt(np.diag(pcov))
    print('fitted ODMR linewidth (MHz):', popt[0]/1e6 , ' pm ', perr[0]/1e6)
    
    TMax = microwaveDetuning3(0, popt[0], 0, popt[2], popt[3])
    TMin = microwaveDetuning3(hyperfine/2*1.2, popt[0], 0, popt[2], popt[3])
    C = (TMax-TMin)/TMean.min()
    dC = C*perr[2]/popt[2]
    print('C:', C , ' pm ', dC)
    print('Tmax/Tmin:', Tmax/Tmin)
        
    
    etaSNL = calcSensitivityCW(popt[0], C, P_det.min(), lamb_s)

    detaSNL = calcSensitivityCWUncertainty(popt[0], perr[0], C, dC, P_det.min(), P_det.min()*0.2, lamb_s)
    
    print('eta High Q:', etaSNL, ' pm ', detaSNL)
    
    ax = axx[0]
    ax.plot(deltaOmegaGHz, DeltaTFit*scale, lw = 0.2/0.3528, color = colorBlue)
    
    
    #%%%% Plot change in transmission fit with MW det 2
    TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp)
    axx[1].plot(f/1e6, TMWsingle1, lw = 0.2/0.3528, color = 'k', linestyle = '--')
    # TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, -hyperfine, amp)
    # axx[1].plot(f/1e6, TMWsingle1, lw = 0.2/0.3528, color = 'k', linestyle = '--')
    # TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, hyperfine, amp)
    # axx[1].plot(f/1e6, TMWsingle1, lw = 0.2/0.3528, color = 'k', linestyle = '--')
    TMW = microwaveDetuning3(f, nuOpt, 0, amp, 4.02066788e-01)
    
    # ax.plot(lamb*1e9, fits, lw = 0.2/0.3528, color = 'k')
    Tmin = -0.3
    ax.set_xlim([deltaOmegaGHzMin-5,deltaOmegaGHzMax])
    ax.spines['bottom'].set_bounds(deltaOmegaGHzMin, deltaOmegaGHzMax)
    
    ax.set_ylim([Tmin-0.1,1.02])
    ax.spines['left'].set_bounds(Tmin,1)
    ax.set_xticks([-40, 40])
    ax.set_yticks([0,DeltaTMaxExp,1])
    ax.set_yticklabels([rf"${val:.2f}$" for val in ax.get_yticks()])
    ax.yaxis.set_label_coords(-0.15, 1.1)
    ax = axx[1]
    
    # DeltaTFit = microwaveDetuning3(f, *popt)
    
    ax.plot(f/1e6, TMW, lw = 0.2/0.3528, color = colorBlue)
    gammaMW = popt[0]
    # print('gammaMW: ', gammaMW)
    # print('popt: ', popt)
    amp =popt[2]
   
    ax.yaxis.set_label_coords(-0.18, 1.1)
    
    ax.set_xlim([fmin/1e6-0.7,fmax/1e6])
    ax.spines['bottom'].set_bounds(fmin/1e6,fmax/1e6)
    
    ax.set_ylim([-0.15, DeltaTMaxExp*10+0.1])
    ax.spines['left'].set_bounds(0,DeltaTMaxExp*10)
    # ax.set_xticks()
    ax.set_yticks([0, 1, DeltaTMaxExp*10])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
    # ax.set_xticklabels([rf"${val:.0f}$" for val in ax.get_xticks()])
    # ax.set_yticklabels([r'$0$', r'$1/2$', r'$1$'])
    
    for ax in axx:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
        # ax.set_xlim([-11,10])
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.set_ticks_position('bottom')
        ax.set_ylabel(ax.get_ylabel(), rotation=0)   # horizontal
        ax.xaxis.set_label_coords(0.55,-0.27)
        

    axx[0].yaxis.set_label_coords(-0.16, 1.05)
    
    axx[1].yaxis.set_label_coords(-0.04, 1.05)
    axx[0].text(-46, 1.11, r"$T$ (au)", color=colorRed,
         ha='right', fontsize = labelfontsize-1)
    axx[0].text(-46, 1.29, r"$\Delta T$ (au)", color=colorBlue,
         ha='right', fontsize = labelfontsize-1)
    
    
    fig.subplots_adjust(hspace=0.7)
    
    TMWsingle1 = lorentzianSingleWithAmplitude(f, nuOpt, 0, amp)
    TMW = microwaveDetuning3(f, nuOpt, 0, amp, 0)
    # print('triple/single: ', TMW.max()/TMWsingle1.max())
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "parameterDefinitionsHighQ.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

def plotODMRFittingCurvesHighQ(saveBool = False):
    
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*0.7, PRQuantumWidthHalf*0.4, 
       r"detuning, $\Delta\omega_{\text{MW}}/2\pi$ (MHz)", 
       r"$S_{\text{LI}}$ ($\upmu$V)",
        labelPadX = 5, labelPadY = 6, commonY = False, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 1,nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    # directory = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260201/';
    # filename = 'ODMR_20260201_CW__16_12dBAtt_4.2um_0_1__2.7767GHz_16MHz_001.mat';
    filePath = filepathODMR+ filenameODMR4p4umBiasLI2
    data = loadmat(filePath, squeeze_me=True)
    data = data['data']
    
    W_nm = np.squeeze(data['laserWLsRead'].item())
    
    S_V = -data['signal'].item()
    C = data['contrast'].item()
    DC_V = S_V/C
    DeltaT = S_V/np.max(DC_V)
    power = data['powerIn'].item()
    freqsODMR = data['freqsODMR'].item()
    f = freqsODMR*1e-6-2776.75
    fCont = np.linspace(-10, 10, 1000)
    p0 = [1e6, 0, -500, 0]
    
    
    S_fit = microwaveDetuningDeriv3(fCont*1e6, *p0)
    cent = 540
    N= 5
    S_V_1D = np.mean(S_V[:, cent - N:cent+N], 1)+0.25e-6
    popt, pcov = curve_fit(microwaveDetuningDeriv3, f*1e6, S_V_1D, p0=p0)
    
    # offset = popt[1]
    # yoffset = popt[-1]
    f = f 
    S_V_1D = S_V_1D 
    
    # popt, pcov = curve_fit(microwaveDetuning3, f*1e6, S_V_1D, p0=p0)
    perr = np.sqrt(np.diag(pcov))  # standard deviations of parameters
    
    S_fit = microwaveDetuningDeriv3(fCont*1e6, *popt) # V * 1e6 uV/V
    # S_fitDeriv_uVperMHz = microwaveDetuningDDeriv3(fCont*1e6, *popt) * 1e6 * 1e6 # V/Hz *1e6 uV/V *1e6 Hz/MHz
    # S_fitDeriv_uVpermT = S_fitDeriv_uVperMHz*1e3*28*1e-3 # uV/MHz * 1e3 MHz/GHz * 28 GHz/T * 1e-3 T/mT
        
    dSdwuVpMHz = microwaveDetuningDDeriv3(0,*popt)*1e6 * 1e6
    dSdwuVpMHzDiff = microwaveDetuningDDeriv3(0,*popt)*1e6 * 1e6 - microwaveDetuningDDeriv3(0,*(popt+perr))*1e6 * 1e6
    dwdBGHzpT = gamma/1e9
    print(dSdwuVpMHz) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    print(dSdwuVpMHzDiff) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    print(dSdwuVpMHz * dwdBGHzpT*1e3/1e6) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    print(dSdwuVpMHzDiff * dwdBGHzpT*1e3/1e6) # V/Hz * 1Hz / 1e-9MHz * 1e6 uV/V
    
    axArr.scatter(f, S_V_1D*1e6, edgecolors='none', marker = '.', s = 8, zorder = 10, color = colorDarkRed)

    fLin = np.linspace(-1,1,2)*0.35
    
    axArr.plot(fCont, S_fit*1e6, color=colorRed, lw=0.2/0.3528)
    # axArr.plot(fLin, -fLin*11.4, linestyle = '--',color=colorBlue, lw=0.2/0.3528)
    
    # axArr[1].plot(fCont, S_fitDeriv_uVpermT, color=colorRed, lw=0.2/0.3528)
    ax = axArr
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax.set_xlim([-11,10])
    ax.spines['bottom'].set_bounds(-10,10)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    
    
    ax.spines['left'].set_bounds(np.min(S_fit)*1e6,np.max(S_fit)*1e6)
    ax.set_yticks([np.min(S_fit)*1e6, 0, np.max(S_fit)*1e6])
    ax.set_yticklabels([rf"${val:.1f}$" for val in ax.get_yticks()])
   
    # ax.spines['left'].set_bounds(np.min(S_fitDeriv_uVpermT),np.max(S_fitDeriv_uVpermT))
    # ax.set_yticks([np.min(S_fitDeriv_uVpermT), 0, np.max(S_fitDeriv_uVpermT)])
    # ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
    # ax.set_ylim([-0.2,None])
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "signalConversionHighQ" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
def plotNoiseFigureHighQ(saveBool = False):
    
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*0.75, PRQuantumWidthHalf*0.55, 
       r"frequency, $f$ (Hz)", 
       r"$\sigma_{B}$ (nT/$\sqrt{\text{Hz}}$)" #"coincidence rate (mHz/mW)"
       , labelPadX = 5, labelPadY = 6, commonY = False, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 1,nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    directoryCW = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260205/'
    filenameCW =  '/TimeSignal_20260205_CW__16_12dBAtt_4.4um_0_1__2.7767GHz_0MHz_001_timeSignal.mat';

    # load data
    mat = loadmat(directoryCW + filenameCW)
    S_V = np.squeeze(mat['signal'])
    S_V -= np.mean(S_V)
    
    time = np.squeeze(mat['time'])
    tau = time[1]-time[0]
    fss = 1/tau
    
    def sig2BCW(signal):
        return signal*1e6/0.318 # V * 1e6 uV/V / 0.318 uV/uT * 1e3 nT/uT
    
    
    B_uT = sig2BCW(S_V)
    xlimAD = 1e3
        
    N = len(B_uT)
    
    # fft, f = fft(time, fss, N)
    f, psd = welch(B_uT, fs=fss, nperseg=2**14)
    fAvg, psdAvg = welch(B_uT, fs=fss, nperseg=2**10)
    
    magNoiseuT = np.sqrt(psd)
    magNoiseAvguT = np.sqrt(psdAvg)
    filt = (f>=1e-2) & (f<=1e3)
    f = f[filt]
    magNoisenT = magNoiseuT[filt]*1000
    filt = (fAvg>=1e-2) & (fAvg<=1e3)
    fAvg = fAvg[filt]
    magNoiseAvgnT = magNoiseAvguT[filt]*1000
    
    sensitivityPSD = np.mean(magNoisenT[f < 1e1])
    print('PSD extracted sensitivity: ', sensitivityPSD, ' nT')
    
    filtAvg = (f>=1e-2) & (f<=1e3)
    
    # m_values = np.arange(1, N // 2 - 10)  # All possible m integers up to N/2 - 10
    # m_values = np.logspace(np.log10(np.min(m_values)), np.log10(np.max(m_values)), 101)
    # avar = np.zeros_like(m_values, dtype=float)
    # taum = m_values / fss
    
    # # Compute Allan variance
    # (t2, ad, ade, adn) = oadev(B_uT, rate=fss, data_type="freq", taus=taum)

    # filt = (t2>=1e-2) & (t2<=xlimAD)
    # t2 = t2[filt]
    # ad = ad[filt]

    # def model_func(logt, logsens):
    #     return -1/2*logt + logsens
    
    # def allan_func(t, A, B, C):
    #     return np.log10((A**2)/t + (B**2) + (C**2)*t)
    
    # p0 = [1e-1, 0, 10e-4]
    # popt, pcov = curve_fit(allan_func, t2, np.log10(ad**2), p0=p0)
    
    # print(popt)
    # sensitivityAD = popt[0]
    # print('Allan deviation extracted sensitivity: ', sensitivityAD*np.sqrt(2)*1000, ' nT')
    
    # tCont = np.logspace(-2, 3, 1000)
    # adFit = allan_func(tCont, *popt)
    # adFit = np.sqrt(10**adFit)
    
    # ad *= 1000
    # adFit *= 1000
  
    ax = axArr
    lw = 0.2/0.3528
    ax.plot(f, magNoisenT, linestyle = '-', color=colorDarkRed, lw = lw)
    ax.plot(fAvg, magNoiseAvgnT, linestyle = '-', color=colorRed, lw = lw)
    ax.plot([1e-3, 1e3], [sensitivityPSD, sensitivityPSD], linestyle = '--', color='k', lw = lw, zorder = 11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # axArr[1].plot(tCont , adFit, linestyle = '-', color=colorDarkRed, lw = lw)
    # axArr[1].plot(np.logspace(-3, 3, 1000) , 1/np.sqrt(np.logspace(-3, 3, 1000))*50, linestyle = '-', color='k', lw = lw)
    # axArr[1].scatter(t2 , ad, marker = '.', color=colorRed, s = 12,zorder = 11, edgecolors='none')
    # axArr[1].set_xscale('log')
    # axArr[1].set_yscale('log')
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
   
    plt.tight_layout()
    
    ax = axArr
    
    ax.spines['bottom'].set_bounds(1e-1,5e2)
    ax.set_xlim([1e-1*0.7,np.max(f)])
    xticksMinor = ax.get_xticks(minor = True)
    filt = np.logical_and(~(xticksMinor < 1e-1), ~(xticksMinor > 5e2))
    ax.set_xticks(xticksMinor[(filt)], minor = True)
    
    ax.spines['left'].set_bounds(5e1,1e3)
    ax.set_ylim([41,1e3])
    
    yticks = ax.get_yticks()
    
    filt = np.logical_and((yticks <= 1e3), (yticks >= 5e1))
    
    ax.set_yticks(np.concatenate([yticks[filt],[sensitivityPSD]]), minor = False)
    yticks = ax.get_yticks()
    # print(yticks)
    ax.set_yticklabels([r'$100$', r'$1000$', rf'${sensitivityPSD:.0f}$'])
    
    
    
    # ax = axArr[1]
    
    # ax.spines['bottom'].set_bounds(np.min(tCont),np.max(tCont))
    # ax.set_xlim([np.min(tCont)*0.7,np.max(tCont)])
    # xticksMinor = ax.get_xticks(minor = True)
    # filt = np.logical_and(~(xticksMinor < 1e-2), ~(xticksMinor > xlimAD))
    # ax.set_xticks(xticksMinor[(filt)], minor = True)
    
    # ax.spines['left'].set_bounds(np.min(ad),np.max(ad))
    # ax.set_ylim([np.min(ad)*0.8,np.max(ad)*1.01])
    # yticks = ax.get_yticks()
    
    # filt = np.logical_and(~(yticks < np.min(ad)), ~(yticks > np.max(ad)))
    # ax.set_yticks(np.concatenate([yticks[filt],np.array([np.min(ad),np.max(ad) ])]), minor = False)
    # yticks = ax.get_yticks()
    # # print(yticks)
    # ax.set_yticklabels([r'$100$', r'$1000$', r'$12.8$', r''])
    
    # ax.set_yticks([np.min(S_fitDeriv_uVpermT), 0, np.max(S_fitDeriv_uVpermT)])
    # ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
    # ax.set_ylim([-0.2,None])
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "noiseMeasurementHighQ" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plotSpinContrastMechanismsPL(saveBool = False):
   
    fig, ax = plot_custom(
       PRQuantumWidthHalf*0.7, 0.5*PRQuantumWidthHalf, 
       r"$\omega_{\text{MW}}/2\pi$ (GHz)", 
       r'count rate, $R$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False, spineWidth = 0.15
    )
    
    lw = 0.2/0.3528
    pops, dets = solvePopulationDetuningLineplot(saveBool = False, 
                                    OmegaR = OmegaROpt,          
                                    I_p = I_pOpt,
                                    I_s = I_sOpt,
                                    bw = 0.03e9*2*np.pi,
                                    Npoints = 401,
                                    fastBool = False,
                                    plotBool = False,
                                    returnBool = True )
    p = pops
    dets /=2*np.pi
    pGreen = p[0] + p[1] + p[2]
    pPL = p[3] + p[4] + p[5]
    pIRPL = p[6]
    pIRAbs = p[7]
    
    pGreen = (pGreen/pGreen.min()-1) + 1
    pPL = pPL/pPL.max()
    pIRPLSingle = pIRPL/pIRPL.min()
    pIRAbs = -(pIRAbs/pIRAbs.min()-1) + 1
    
    def singletToHyperfine(dets, p, hyperfine=hyperfine):
        # Create interpolation function
        f = interp1d(dets, p, bounds_error=False, fill_value='extrapolate')
    
        # Keep same x-axis (or you could shrink it later if you want)
        detsNew = dets.copy()
    
        # Sum shifted versions: center, +hf, -hf
        offset = 15e6
        pNew = (
            f(detsNew - offset + hyperfine) +
            f(detsNew - offset) +
            f(detsNew - offset - hyperfine) +
            f(detsNew + offset + hyperfine) +
            f(detsNew + offset) +
            f(detsNew + offset - hyperfine) 
            
        )
    
        return detsNew, (pNew-6)*7+1
    detsNew = dets
    # detsNew, pGreen = singletToHyperfine(dets, pGreen)
    detsNew, pPL = singletToHyperfine(dets, pPL)
    # detsNew, pIRPLT = singletToHyperfine(dets, pIRPLSingle)
    # detsNew, pIRAbs = singletToHyperfine(dets, pIRAbs)
    
    detsNew = detsNew/1e6 + 2870
    dets /= 1e0
    
    min_x_value = 2.840
    max_x_value = 2.900
    min_y_value = 0.925
    max_y_value = 1.0
    
    filt = np.logical_and(detsNew < max_x_value*1e3, detsNew > min_x_value*1e3)
    # dets = dets[filt]
    
    
    lw = 0.2/0.3528
    
    # ax.plot(detsNew[filt], pGreen[filt], 'g', lw = lw)
    ax.plot(detsNew[filt]/1e3, pPL[filt], 'r', lw = lw)
    # ax.plot(dets[filt], pIRPLT[filt], color = 'k', lw = lw)
    # ax.plot(dets[filt], (pIRPLSingle[filt]-1)/3.3+1, color = 'grey', linestyle = '--', lw = lw)
    # ax.plot(dets, pIRPL, color = 'k', lw = lw, linestyle = '--', zorder = -1)
    # ax.plot(dets[filt], pIRAbs[filt], colorDarkRed, lw = lw)
    

    ax.set_xticks([min_x_value, 2.87, max_x_value])
    ax.set_yticks([1])
# 
    rangex = max_x_value - min_x_value
    rangey = max_y_value - min_y_value

    limScale = 0.1
    limScaley = 0.1
    limScaleMin = rangey*limScaley
    limScaleMax = rangey*limScaley

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=numfontsize,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)


    ax.set_ylim([min_y_value - limScaleMin, max_y_value + limScaleMin])
    ax.set_xlim([min_x_value-0.002, max_x_value + rangex*limScale])

    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')



    ax.spines['left'].set_bounds(min_y_value,max_y_value)

    plt.tight_layout()
    plt.show()
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "ODMRContrastSim.svg", dpi=300, bbox_inches="tight",transparent=True
        )

def plotNoiseFigureCAFQA(saveBool = False):
    
    fig, axArr = plot_custom(
       PRQuantumWidthHalf*1.5, PRQuantumWidthHalf*0.8, 
       [r"frequency, $f$ (Hz)", r"averaging time, $t$ (s)"], 
       [r"magnetic noise (nT/$\sqrt{\text{Hz}}$)",r"Allan deviation, $\sigma_a$ (nT)"] #"coincidence rate (mHz/mW)"
       , labelPadX = 5, labelPadY = 6, commonY = False, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       ncol = 2,nrow = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    directoryCW = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260201/'
    filenameCW =  '/TimeSignal_20260201_CW__16_12dBAtt_4.2um_0_1__2.7767GHz_4MHz_007_timeSignal.mat';

    # load data
    mat = loadmat(directoryCW + filenameCW)
    S_V = np.squeeze(mat['signal'])
    S_V -= np.mean(S_V)
    
    time = np.squeeze(mat['time'])
    tau = time[1]-time[0]
    fss = 1/tau
    
    def sig2BCW(signal):
        return signal*1e6/0.318 # V * 1e6 uV/V / 0.318 uV/uT * 1e3 nT/uT
    
    
    B_uT = sig2BCW(S_V)
    xlimAD = 1e3
        
    N = len(B_uT)
    
    # fft, f = fft(time, fss, N)
    f, psd = welch(B_uT, fs=fss, nperseg=2**20)
    fAvg, psdAvg = welch(B_uT, fs=fss, nperseg=2**16)
    
    magNoiseuT = np.sqrt(psd)
    magNoiseAvguT = np.sqrt(psdAvg)
    filt = (f>=1e-2) & (f<=1e3)
    f = f[filt]
    magNoisenT = magNoiseuT[filt]*1000
    filt = (fAvg>=1e-2) & (fAvg<=1e3)
    fAvg = fAvg[filt]
    magNoiseAvgnT = magNoiseAvguT[filt]*1000
    
    sensitivityPSD = np.mean(magNoisenT[f < 1e1])
    print('PSD extracted sensitivity: ', sensitivityPSD, ' nT')
    
    filtAvg = (f>=1e-2) & (f<=1e3)
    
    m_values = np.arange(1, N // 2 - 10)  # All possible m integers up to N/2 - 10
    m_values = np.logspace(np.log10(np.min(m_values)), np.log10(np.max(m_values)), 101)
    avar = np.zeros_like(m_values, dtype=float)
    taum = m_values / fss
    
    # Compute Allan variance
    (t2, ad, ade, adn) = oadev(B_uT, rate=fss, data_type="freq", taus=taum)

    filt = (t2>=1e-2) & (t2<=xlimAD)
    t2 = t2[filt]
    ad = ad[filt]

    def model_func(logt, logsens):
        return -1/2*logt + logsens
    
    def allan_func(t, A, B, C):
        return np.log10((A**2)/t + (B**2) + (C**2)*t)
    
    p0 = [1e-1, 0, 10e-4]
    popt, pcov = curve_fit(allan_func, t2, np.log10(ad**2), p0=p0)
    
    # print(popt)
    sensitivityAD = popt[0]
    print('Allan deviation extracted sensitivity: ', sensitivityAD*np.sqrt(2)*1000, ' nT')
    
    tCont = np.logspace(-2, 3, 1000)
    adFit = allan_func(tCont, *popt)
    adFit = np.sqrt(10**adFit)
    
    ad *= 1000
    adFit *= 1000
  
    lw = 0.2/0.3528
    axArr[0].plot(f, magNoisenT, linestyle = '-', color=colorDarkRed, lw = lw)
    axArr[0].plot(fAvg, magNoiseAvgnT, linestyle = '-', color=colorRed, lw = lw)
    axArr[0].plot([1e-3, 1e3], [sensitivityPSD, sensitivityPSD], linestyle = '--', color='k', lw = lw, zorder = 11)
    axArr[0].set_xscale('log')
    axArr[0].set_yscale('log')
    
    # axArr[1].plot(tCont , adFit, linestyle = '-', color=colorDarkRed, lw = lw)
    axArr[1].plot(np.logspace(-3, 3, 1000) , 1/np.sqrt(np.logspace(-3, 3, 1000))*50, linestyle = '-', color='k', lw = lw)
    axArr[1].scatter(t2 , ad, marker = '.', color=colorRed, s = 12,zorder = 11, edgecolors='none')
    axArr[1].set_xscale('log')
    axArr[1].set_yscale('log')
    
    for ax in axArr:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        
   
    plt.tight_layout()
    
    ax = axArr[0]
    
    ax.spines['bottom'].set_bounds(1e-2,5e2)
    ax.set_xlim([np.min(f)*0.65,np.max(f)])
    xticksMinor = ax.get_xticks(minor = True)
    filt = np.logical_and(~(xticksMinor < 1e-2), ~(xticksMinor > 5e2))
    ax.set_xticks(xticksMinor[(filt)], minor = True)
    
    ax.spines['left'].set_bounds(5e1,1e3)
    ax.set_ylim([4.3e1,1e3])
    
    yticks = ax.get_yticks()
    
    filt = np.logical_and((yticks <= 1e3), (yticks >= 5e1))
    
    ax.set_yticks(np.concatenate([yticks[filt],[sensitivityPSD]]), minor = False)
    yticks = ax.get_yticks()
    # print(yticks)
    ax.set_yticklabels([r'$100$', r'$1000$', rf'${sensitivityPSD:.0f}$'])
    
    
    
    ax = axArr[1]
    
    ax.spines['bottom'].set_bounds(np.min(tCont),np.max(tCont))
    ax.set_xlim([np.min(tCont)*0.7,np.max(tCont)])
    xticksMinor = ax.get_xticks(minor = True)
    filt = np.logical_and(~(xticksMinor < 1e-2), ~(xticksMinor > xlimAD))
    ax.set_xticks(xticksMinor[(filt)], minor = True)
    
    ax.spines['left'].set_bounds(np.min(ad),np.max(ad))
    ax.set_ylim([np.min(ad)*0.8,np.max(ad)*1.01])
    yticks = ax.get_yticks()
    
    filt = np.logical_and(~(yticks < np.min(ad)), ~(yticks > np.max(ad)))
    ax.set_yticks(np.concatenate([yticks[filt],np.array([np.min(ad),np.max(ad) ])]), minor = False)
    yticks = ax.get_yticks()
    # print(yticks)
    ax.set_yticklabels([r'$100$', r'$1000$', r'$12.8$', r''])
    
    # ax.set_yticks([np.min(S_fitDeriv_uVpermT), 0, np.max(S_fitDeriv_uVpermT)])
    # ax.set_yticklabels([rf"${val:.0f}$" for val in ax.get_yticks()])
    # ax.set_ylim([-0.2,None])
    
    if saveBool:
        fig.savefig(
            (_FIGURES_DIR + "/") + "noiseMeasurementCAFQA" + ".svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

#%% other function calls
saveBool = False

#### unused functions
# plotVisibleSpectrometerData(saveBool)
# plotVisibleSpectrometerDiffData(saveBool)
# plotIRSpectrometerData(saveBool)
# plotIRSpectrometerDiffData(saveBool)
# plotEtalonSensitivityCavityLength(saveBool)
# plotEtalonSensitivity(saveBool)
# plotEtalonSensitivityKappa(saveBool)
# plotMicrodiskTheorySensitivity(saveBool)
# plotWideTransmission(saveBool)

#### conference abstract functions
# plot2DData(saveBool)

#%% paper function calls

saveBool = False

## figure 1
plotZeemanSplitting(saveBool = saveBool)
# plotSpinContrastMechanisms(saveBool = saveBool)

## figure 2
# solveSensitivityVaryIRPower(saveBool = saveBool)
# plotEtalonSensitivity1D(saveBool)
# plotSensitivityRelation(saveBool)
# plotSensitivityComparison(saveBool)

## figure 3
# plotIRSpectrometerCombined(saveBool)
# plotIRSpectrometerODMR(saveBool)

## figure 4
# plotChangeInTransmissionCartoon(saveBool)
# plotChangeInTransmissionCartoon2(saveBool)

## figure 5
# plotNarrowTransmissionPaper(saveBool)
# plotZeroFieldODMR(saveBool)
# plotWideTransmissionAndSignal(saveBool)

## figure 6
# plotODMRFittingCurves(saveBool = saveBool)
# plotNoiseFigure(saveBool = saveBool)

# figure 7
# plotDetuningExample(saveBool)
# plotMicrodiskTheorySensitivityDeltaTvsMWDet(saveBool = saveBool)
# plotMicrodiskTheorySensitivitykcavkex(saveBool = saveBool)

## figure 8
# plotLitReview(saveBool)

## Appendix
# plotCWOptimization(saveBool = saveBool)
# only run once
# solveAlphaFuncofIRIntensity(N = 51)

# plotMicrodiskTheorySensitivityDeltaTvsMWDetSI(saveBool = saveBool)
# plotLockinExample(saveBool)
# plotMicrodiskDeltaTvsMWDetZeroFieldSI(saveBool = saveBool)
# plotMicrodiskDeltaTvsMWDetSI(saveBool = saveBool)

## plot Second Higher Q Mode result
# plotDetunings(saveBool)
# plotNarrowTransmissionPaperHighQ(saveBool)
# plotDetuningExampleHighQ(saveBool)
# plotMicrodiskTheorySensitivityDeltaTvsMWDeHighQ(saveBool = saveBool)
# plotODMRFittingCurvesHighQ(saveBool = saveBool)
# plotNoiseFigureHighQ(saveBool = saveBool)


# print((calcEnergyLossRate(nd, alpha1) - calcEnergyLossRate(nd, alpha0))/1e6)
# plotSpinContrastMechanismsPL(saveBool = saveBool)
# plotNoiseFigureCAFQA(saveBool = saveBool)
# plotMicrodiskTheoryCompareCalcVsSimple(saveBool = saveBool)
