# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 16:56:35 2025

@author: nicho
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# import seaborn as sns
from plot_custom import plot_custom
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
from scipy.signal import butter, filtfilt
import re
# import pandas as pd
import os
# import emcee
from scipy import odr
# from scipy import odr

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in multiply")

labelfontsize = 8
numfontsize = 8
textwidth = 174/25.4#8.5 - 2 # in inches
textwidth= 8.5-1.25
OpticaWidthFull = textwidth
OpticaWidthHalf = (8.5-1.25)/2 - 1/6

# === Constants ===
hbar = 1.054e-34
c_const = 299792458.0 #m/s
greenBool = False
savePlotBool = False
ms = 2
msa = 3
lw = 0.5
nd = 2.39
GHzprad = 1e-9/2/np.pi
lambdah = 1358

#%% conversion functions
def wavelength_to_freq(lambda_nm):
    """Convert wavelength (nm) to frequency (THz)."""
    lambda_m = lambda_nm * 1e-9
    return c_const / lambda_m / 1e12

def freq_to_wavelength(freq_THz):
    """Convert frequency (THz) to wavelength (nm)."""
    return (c_const / (freq_THz * 1e12)) * 1e9

def ItoN979(I, lambdacav = 979e-9, ng = 2.41, V0 = 21.3):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI979(N, lambdacav = 979e-9, ng = 2.41, V0 = 21.3):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1047(I, lambdacav = 1047e-9, ng = 2.41, V0 = 18.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI1047(N, lambdacav = 1047e-9, ng = 2.41, V0 = 18.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1267(I, lambdacav = 1267e-9, ng = 2.44, V0 = 12.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI1267(N, lambdacav = 1267e-9, ng = 2.44, V0 = 12.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1322(I, lambdacav = 1322e-9, ng = 2.45, V0 = 11.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI1322(N, lambdacav = 1322e-9, ng = 2.45, V0 = 11.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1383(I, lambdacav = 1383e-9, ng = 2.46, V0 = 10.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI1383(N, lambdacav = 1383e-9, ng = 2.46, V0 = 10.4):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1450(I, lambdacav = 1450e-9, ng = 2.47, V0 = 9.49):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI1450(N, lambdacav = 1450e-9, ng = 2.47, V0 = 9.49):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1524(I,  lambdacav = 1524e-9, ng = 2.49, V0 = 8.64):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/8  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10 

def NtoI1524(N, lambdacav = 1524e-9, ng = 2.49, V0 = 8.64):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10 # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2


def ItoN1604(I, lambdacav = 1604e-9, ng = 2.50, V0 = 7.86):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return (1/2  * c_const * hbar * wcav / (ng * Veff))**-1 *I *1e10

def NtoI1604(N, lambdacav = 1604e-9, ng = 2.50, V0 = 7.86):
    wcav = 2*np.pi*c_const/lambdacav
    Veff = V0 * (lambdacav/nd)**3
    return 1/2 * N * c_const * hbar * wcav / (ng * Veff) *1e-10  # W/m^2 * 1e-6 MW/W * 1e-4m^2/cm^2

#%% analysis functions

# assumes non-coherent driving of the TLS
def saturable_model_logI(logN, k0, logNsat, kother):
    N = 10**logN # convert from log10(I [mW]) → I [W]
    Nsat = 10**logNsat
    return k0 /(1 + N/Nsat) + kother

def saturable_model_logI_odr(beta, N):
    k0, Nsat, kother = beta
    # N = 10**logN
    # Nsat = 10**logNsat
    return k0 / (1 + N / Nsat) + kother


calcdk = lambda dw, Qip, dQip, w: np.sqrt((dw/Qip)**2 + (dQip*w/Qip**2)**2)
calcNcav = lambda etaFibre, powerW, w, Delta, a: np.max(np.sqrt(etaFibre)*(powerW)/(hbar*(w + Delta))*np.abs(a)**2)
calcdNcav = lambda etaFibre, etaUncertainty, powerW, w, Delta, a: (
    np.max(np.sqrt(((etaFibre)**-0.5*etaUncertainty*(powerW))**2 
                   + (np.sqrt(etaFibre)*(powerW*0.05))**2)/(hbar*(w + Delta))*np.abs(a)**2)
 )


def calcTransmission(kex1, a1, kex2, a2, tau = 1):
    return np.abs(tau - np.sqrt(1/4*kex1)*a1 - np.sqrt(1/4*kex2)*a2)**2

def calcField(w, kex1, k1, CT,  Delta, power = 0):
    
    [a1s, N1s] = Thermal_fit(Delta, power/hbar/w, CT, kex1, k1, 1);
    [a2s, N2s] = Thermal_fit(Delta, power/hbar/w, CT, kex1, k1, 2);
    [a3s, N3s] = Thermal_fit(Delta, power/hbar/w, CT, kex1, k1, 3);
    
    
    NArrs = np.abs([N1s, N2s, N3s])
    aArrs = np.array([a1s, a2s, a3s]);
    # print(np.min(NArrs))

    idxs = np.argmin(NArrs, axis=0)
    NArr1 = NArrs[idxs, np.arange(NArrs.shape[1])]
    aArr1 = aArrs[idxs, np.arange(aArrs.shape[1])]

    return aArr1, NArr1
          

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


def plotSI_AllTransmissionAndKappa(saveBool = False):
    
    # === File and plotting settings ===
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_979nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1267nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1322nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1383nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1450nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1525nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1604nm_Thermo_IRPowerSweep.mat'
    ]
    doubletBool = [False, True, True, True, True, False, False, False]
    labels = ['979','1048','1267','1322','1383','1450','1524','1604']
    lambdaArr = np.array([979, 1048, 1267, 1322, 1383, 1450, 1524, 1604])
    
    etaFibre = np.array(      [0.3, 0.3, 0.39, 0.39, 0.33, 0.43, 0.39, 0.27])
    etaUncertainty = np.array([0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02])
    
    xmin = -50
    xmax = 50
    
    # === Colormaps ===
    numFiles = len(filenames)
    markerss = ['.'] * numFiles
    markersa = ['x'] * numFiles
    cpalette = 'hot'
    cpalette_r = 'hot_r'
    
    cmap_full = cm.get_cmap(cpalette)
    # Clip to lower 90% of the color range (darkens top end)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full(np.linspace(0, 0.7, 256))
    )
    
    cmap_full_r = cm.get_cmap(cpalette_r)
    # Clip to lower 90% of the color range (darkens top end)
    cmap_r = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full_r(np.linspace(0.3, 1, 256))
    )
    norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    colorArray_r = [cmap_r(norm(lmbd)) for lmbd in lambdaArr]
    
    # === Set up plots ===
    
    fig2, axArr = plot_custom(
       OpticaWidthFull*0.2, 0.5*OpticaWidthFull, 
       r"$\Delta\omega/2\pi$ (GHz)", 
       [r'']*numFiles,#[r'$T$ (au)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = numFiles, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    widthRatios =  [1,1]
    fig3, axArr2 = plot_custom(
       OpticaWidthFull*0.5, 0.5*OpticaWidthFull, 
       [r"$\mathcal{N}_{\text{cav}}$"]*5*numFiles, 
       [r'', r'']*numFiles*5,#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = numFiles, ncol = 2, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True, widthRatios =widthRatios, heightRatios = [1]*numFiles
    )
    
    axArr3 = axArr2[:,0]
    axArr4 = axArr2[:,1]
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
        dNcavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            dNcava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        dws = 2 * np.pi * c_const * dlambdas / lambdas**2
    
        if doubletBool[i]:
            wa = 2 * np.pi * c_const / lambdaa
            dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
            g = ws - wa
    
            kips = ws / Qips
            kipa = wa / Qipa
            kexs = ws / Qexs
            kexa = wa / Qexa
    
            dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)
    
            ks = kips + kexs
            ka = kipa + kexa
            
            norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            # cpaletteT = 'viridis'
            # cmapT = cm.get_cmap(cpaletteT)  # or 'plasma', 'inferno', etc.
            colorArrayT = [cmap(norm(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], wa[j]]))*GHzprad > xmin,(w-np.mean([ws[j], wa[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j],  Delta - g[j]/2, power = powerW[j])
                
                Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                dNcavs[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                
                [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                              cT[j],  Delta + g[j]/2, power = powerW[j])
                
                Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                dNcava[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                
                T = calcTransmission(kexs[j], a_s, kexa[j], a_a, tau = tau[j]*np.exp(1j*phi[j]))
                
                if j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw)
                    axArr[i].scatter((wMeas-np.mean([ws[j], wa[j]])/1e9)/2/np.pi,TOut/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2, zorder = 10)
            
            step = 2
            axArr3[i].errorbar(Ncavs[1:-1:step], kips[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
                               xerr=dNcavs[1:-1:step],
                        marker=markerss[i], linestyle = 'none', label=labels[i], 
                        color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.1)
            
            axArr3[i].errorbar(Ncava[1:-1:step], kipa[1:-1:step]*GHzprad, yerr=dkipa[1:-1:step]*GHzprad,
                        marker=markersa[i], linestyle = 'none', 
                        xerr=dNcavs[1:-1:step],
                        color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)
            
            axArr4[i].errorbar(Ncavs[1:-1:step], kexs[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
                        marker=markerss[i], linestyle = 'none', label=labels[i], 
                        xerr=dNcavs[1:-1:step],
                        color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.1 )
            
            axArr4[i].errorbar(Ncava[1:-1:step], kexa[1:-1:step]*GHzprad, yerr=dkipa[1:-1:step]*GHzprad,
                        marker=markersa[i], linestyle = 'none', 
                        xerr=dNcava[1:-1:step],
                        color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)
            
            ymin = np.min(np.abs([kips*GHzprad, kipa*GHzprad]))
            ymax = np.max(np.abs([kips*GHzprad, kipa*GHzprad]))
            axArr3[i].spines['right'].set_bounds(ymin,ymax)
            axArr3[i].set_yticks([ymin, ymax])
            axArr3[i].set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
            
            ymin = np.min(np.abs([kexs*GHzprad, kexa*GHzprad]))
            ymax = np.max(np.abs([kexs*GHzprad, kexa*GHzprad]))
            axArr4[i].spines['right'].set_bounds(ymin,ymax)
            axArr4[i].set_yticks([ymin, ymax])
            axArr4[i].set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
        else:
    
            kips = ws / Qips
            kexs = ws / Qexs
            
            dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            
            ks = kips + kexs
            
            normT = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            colorArrayT = [cmap(normT(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], ws[j]]))*GHzprad > xmin,(w-np.mean([ws[j], ws[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j]/2,  Delta, power = powerW[j])
                
                Ncavs[j] = np.max(np.sqrt(etaFibre[i])*(powerW[j])/(hbar*(np.mean([ws, ws]) + Delta))*np.abs(a_s)**2)
                dNcavs[j] = np.max(np.sqrt( ((etaFibre[i])**-0.5*etaUncertainty[i]*(powerW[j]))**2 + (np.sqrt(etaFibre[i])*(powerW[j]*0.05))**2)/(hbar*(np.mean([ws, ws]) + Delta))*np.abs(a_s)**2)
                
              
                
                T = calcTransmission(kexs[j], a_s, kexs[j], a_s, tau = tau[j]*np.exp(1j*phi[j]))
                
                if j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,5) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw )
                    axArr[i].scatter((wMeas-ws[j]/1e9)/2/np.pi,TOut/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2)
               
               
            step = 2
            axArr3[i].errorbar(Ncavs[1:-1:step], kips[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
            xerr=dNcavs[1:-1:step],
                        marker=markerss[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
            axArr4[i].errorbar(Ncavs[1:-1:step], kexs[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
            xerr=dNcavs[1:-1:step],
                        marker=markerss[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
           
           
           
            ymin = np.min(np.abs([kips*GHzprad]))
            ymax = np.max(np.abs([kips*GHzprad]))
            axArr3[i].spines['right'].set_bounds(ymin,ymax)
            axArr3[i].set_yticks([ymin, ymax])
            axArr3[i].set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
            
            ymin = np.min(np.abs([kexs*GHzprad]))
            ymax = np.max(np.abs([kexs*GHzprad]))
            axArr4[i].spines['right'].set_bounds(ymin,ymax)
            axArr4[i].set_yticks([ymin, ymax])
            axArr4[i].set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
    
    # === Add colorbar for wavelength ===
    cmap = plt.cm.hot
    norm = plt.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig2.colorbar(sm, ax=axArr, orientation='horizontal', pad=0.15)
    cbar.set_label(r"input power, $P$ (dBm)", labelpad=5, fontsize = labelfontsize)
    
    pos = axArr[-1].get_position()
    cbar.ax.set_position([pos.x0+0.08, pos.y1 - 0.45, pos.width-0.1, 0.06])  # [x, y, width, height]
    # Move colorbar above (by adjusting position)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_ticks(np.log10([powerMin, powerMax]))
    cbar.set_ticklabels([rf'${l:.1f}$' for l in 10*np.log10(np.array([powerMin, powerMax])*1e3)], fontsize = numfontsize)
    plt.tight_layout()
    
    for ax in axArr:
        ax.set_ylim([None,1])
        ax.set_xlim([-50, 50])
        y_low, y_high = ax.get_ylim()
        
        ax.set_yticks([y_low, 1])
        ax.spines['left'].set_bounds(y_low,1)
        ax.set_ylim([y_low-(1-y_low)/20,1+(1-y_low)/20])
        ax.set_yticklabels([rf"${val:.2f}$" for val in [y_low, 1]])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.spines['bottom'].set_bounds(-50,50)
        
        ax.set_xlim([-60, 50])
    
    axArr[-1].tick_params(axis='x', which='both', top=False, bottom=True)
    axArr[-1].spines["bottom"].set_visible(True)
    
    norm = plt.Normalize(vmin=min(lambdaArr), vmax=max(lambdaArr))
    sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm)
    cbar = fig3.colorbar(sm, ax=axArr4, orientation='horizontal', label=r'wavelength, $\lambda$ (nm)', aspect = 52)
    cbar.set_ticks(lambdaArr)
    cbar.set_ticklabels([rf'${l:.0f}$' for l in lambdaArr], fontsize = numfontsize)
    cbar.set_label(label=r'wavelength, $\lambda$ (nm)', size = labelfontsize)
    
    pos = axArr[-1].get_position()
    cbar.ax.set_position([pos.x0-0.04, pos.y1 - 0.45, pos.width+0.2, 0.06])  # [x, y, width, height]
    
    # Move colorbar above (by adjusting position)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    plt.tight_layout()
    
    
    
    
    for ax in axArr3:
        ax.spines["bottom"].set_bounds([1e0, 1e6])
        ax.set_xlim([0.5, 3e6])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='y', which='both', left=False, right=True, labelright=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xscale('log')
        
    
    axArr3[-1].set_xticks([1, 100, 1e4, 1e6])
    axArr3[-1].tick_params(axis='x', which='major', top=False, bottom=True)
    axArr3[-1].spines["bottom"].set_visible(True)
    
    for ax in axArr4:
        ax.spines["bottom"].set_bounds([1e0, 1e6])
        ax.set_xlim([0.5, 3e6])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='y', which='both', left=False, right=True, labelright=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xscale('log')
        
    axArr4[-1].set_xticks([1, 100, 1e4, 1e6])
    axArr4[-1].tick_params(axis='x', which='major', top=False, bottom=True)
    axArr4[-1].spines["bottom"].set_visible(True)
        
    
    
    fig2.subplots_adjust(hspace=0.4)
    fig3.subplots_adjust(hspace=0.3)
    
    if saveBool:
        fig2.savefig(
            "./figures/" + "SI_transmissionFits.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        fig3.savefig(
            "./figures/" + "SI_kin.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotSI_AllTransmissionAndKappaNew(saveBool = False):
    
    # === File and plotting settings ===
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_979nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1267nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1322nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1383nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1450nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1525nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1604nm_Thermo_IRPowerSweep.mat'
    ]
    doubletBool = [False, True, True, True, True, False, False, False]
    labels = ['979','1048','1267','1322','1383','1450','1524','1604']
    lambdaArr = np.array([979, 1048, 1267, 1322, 1383, 1450, 1524, 1604])
    
    etaFibre = np.array(      [0.3, 0.3, 0.39, 0.39, 0.33, 0.43, 0.39, 0.27])
    etaUncertainty = np.array([0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02])
    
    xmin = -50
    xmax = 50
    
    # === Colormaps ===
    numFiles = len(filenames)
    markerss = ['.'] * numFiles
    markersa = ['x'] * numFiles
    cpalette = 'hot'
    cpalette_r = 'hot_r'
    
    cmap_full = cm.get_cmap(cpalette)
    # Clip to lower 90% of the color range (darkens top end)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full(np.linspace(0, 0.7, 256))
    )
    
    cmap_full_r = cm.get_cmap(cpalette_r)
    # Clip to lower 90% of the color range (darkens top end)
    cmap_r = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full_r(np.linspace(0.4, 0.9, 256))
    )
    norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    colorArray_r = [cmap_r(norm(lmbd)) for lmbd in lambdaArr]
    
    # === Set up plots ===
    
    fig2, axArr = plot_custom(
       OpticaWidthFull*0.9, 0.35*OpticaWidthFull, 
       [r""]*numFiles, #$\Delta\omega/2\pi$ (GHz)
       r'transmission, $T$ (au)',#[r'$T$ (au)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = True, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = numFiles, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    widthRatios =  [1,1]
    fig3, axArr2 = plot_custom(
       OpticaWidthHalf, 0.8*OpticaWidthFull, 
       r'intracavity photons, $\mathcal{N}_{\text{cav}}$', 
       [r'$\kappa_{\text{c}}/2\pi$ (GHz)', r'$\kappa_{\text{ex}}/2\pi$ (GHz)'],#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True, widthRatios = [1], heightRatios = [50, 12]
    )
    
    axArr3 = axArr2[0]
    axArr4 = axArr2[1]
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
        dNcavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            dNcava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        dws = 2 * np.pi * c_const * dlambdas / lambdas**2
    
        if doubletBool[i]:
            wa = 2 * np.pi * c_const / lambdaa
            dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
            g = ws - wa
    
            kips = ws / Qips
            kipa = wa / Qipa
            kexs = ws / Qexs
            kexa = wa / Qexa
    
            dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)
    
            ks = kips + kexs
            ka = kipa + kexa
            
            norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            # cpaletteT = 'viridis'
            # cmapT = cm.get_cmap(cpaletteT)  # or 'plasma', 'inferno', etc.
            colorArrayT = [cmap(norm(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], wa[j]]))*GHzprad > xmin,(w-np.mean([ws[j], wa[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j],  Delta - g[j]/2, power = powerW[j])
                
                Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                dNcavs[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                
                [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                              cT[j],  Delta + g[j]/2, power = powerW[j])
                
                Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                dNcava[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                
                T = calcTransmission(kexs[j], a_s, kexa[j], a_a, tau = tau[j]*np.exp(1j*phi[j]))
                
                if j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw)
                    axArr[i].scatter((wMeas-np.mean([ws[j], wa[j]])/1e9)/2/np.pi,TOut/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2, zorder = 10)
            
            step = 1
            axArr3.errorbar(Ncavs[1:-1:step], kips[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
                               xerr=dNcavs[1:-1:step],
                        marker=markerss[i], linestyle = 'none', label=labels[i], 
                        color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.1)
            
            axArr3.errorbar(Ncava[1:-1:step], kipa[1:-1:step]*GHzprad, yerr=dkipa[1:-1:step]*GHzprad,
                        marker=markersa[i], linestyle = 'none', 
                        xerr=dNcavs[1:-1:step],
                        color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)
            
            axArr4.errorbar(Ncavs[1:-1:step], kexs[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
                        marker=markerss[i], linestyle = 'none', label=labels[i], 
                        xerr=dNcavs[1:-1:step],
                        color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.1 )
            
            axArr4.errorbar(Ncava[1:-1:step], kexa[1:-1:step]*GHzprad, yerr=dkipa[1:-1:step]*GHzprad,
                        marker=markersa[i], linestyle = 'none', 
                        xerr=dNcava[1:-1:step],
                        color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)
            
            # ymin = np.min(np.abs([kips*GHzprad, kipa*GHzprad]))
            # ymax = np.max(np.abs([kips*GHzprad, kipa*GHzprad]))
            # axArr3.spines['right'].set_bounds(ymin,ymax)
            # axArr3.set_yticks([ymin, ymax])
            # axArr3.set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
            
            # ymin = np.min(np.abs([kexs*GHzprad, kexa*GHzprad]))
            # ymax = np.max(np.abs([kexs*GHzprad, kexa*GHzprad]))
            # axArr4.spines['right'].set_bounds(ymin,ymax)
            # axArr4.set_yticks([ymin, ymax])
            # axArr4.set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
        else:
    
            kips = ws / Qips
            kexs = ws / Qexs
            
            dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            
            ks = kips + kexs
            
            normT = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            colorArrayT = [cmap(normT(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], ws[j]]))*GHzprad > xmin,(w-np.mean([ws[j], ws[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j]/2,  Delta, power = powerW[j])
                
                Ncavs[j] = np.max(np.sqrt(etaFibre[i])*(powerW[j])/(hbar*(np.mean([ws, ws]) + Delta))*np.abs(a_s)**2)
                dNcavs[j] = np.max(np.sqrt( ((etaFibre[i])**-0.5*etaUncertainty[i]*(powerW[j]))**2 + (np.sqrt(etaFibre[i])*(powerW[j]*0.05))**2)/(hbar*(np.mean([ws, ws]) + Delta))*np.abs(a_s)**2)
                
              
                
                T = calcTransmission(kexs[j], a_s, kexs[j], a_s, tau = tau[j]*np.exp(1j*phi[j]))
                
                if j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,5) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw )
                    axArr[i].scatter((wMeas-ws[j]/1e9)/2/np.pi,TOut/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2)
               
               
            step = 1
            axArr3.errorbar(Ncavs[1:-1:step], kips[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
            xerr=dNcavs[1:-1:step],
                        marker=markerss[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
            axArr4.errorbar(Ncavs[1:-1:step], kexs[1:-1:step]*GHzprad, yerr=dkips[1:-1:step]*GHzprad,
            xerr=dNcavs[1:-1:step],
                        marker=markerss[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
           
           
           
            # ymin = np.min(np.abs([kips*GHzprad]))
            # ymax = np.max(np.abs([kips*GHzprad]))
            # axArr3.spines['right'].set_bounds(ymin,ymax)
            # axArr3.set_yticks([ymin, ymax])
            # axArr3.set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
            
            # ymin = np.min(np.abs([kexs*GHzprad]))
            # ymax = np.max(np.abs([kexs*GHzprad]))
            # axArr4.spines['right'].set_bounds(ymin,ymax)
            # axArr4.set_yticks([ymin, ymax])
            # axArr4.set_yticklabels([rf"${val:.1f}$" for val in [ymin, ymax]])
    
    # === Add colorbar for wavelength ===
    
    norm = plt.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig2.colorbar(sm, ax=axArr, orientation='vertical', pad=0.15)
    cbar.set_label(r"input power, $P$ (dBm)", labelpad=-10, fontsize = labelfontsize)
    
    pos = axArr[-1].get_position()
    cbar.ax.set_position([pos.x0+0.36, pos.y1 - 0.79, 0.05, 0.78])  # [x, y, width, height]
    # Move colorbar above (by adjusting position)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_ticks(np.log10([powerMin, powerMax]))
    cbar.set_ticklabels([rf'${l:.1f}$' for l in 10*np.log10(np.array([powerMin, powerMax])*1e3)], fontsize = numfontsize)
    plt.tight_layout()
    
    i = -1
    for ax in axArr:
        i += 1
        ax.set_ylim([0,1])
        ax.set_xlim([50,-50])
        y_low, y_high = ax.get_ylim()
        
        ax.spines['left'].set_bounds(y_low,1)
        
        ax.set_xticks([-50, 50])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(True)
        ax.spines["left"].set_visible(False)
        # ax.yaxis.set_ticks_position('left')
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis='x', which='both', top=True, bottom=False, labeltop = True)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.spines['top'].set_bounds(50,-50)    
        
        ax1sec = ax.secondary_xaxis('bottom')
        ax1sec.set_xlabel(r"",fontsize = labelfontsize)
        ax1sec.set_xticks([0])
        ax1sec.set_xticklabels([rf"${labels[i]}$"])
        ax1sec.tick_params(
            axis="x",
            which="major",
            width=0.8,
            length=2,
            labelsize=numfontsize,
            zorder=1,
            direction="in",
            colors='k'
        )
        if i == 0:
            ax1sec.spines['bottom'].set_bounds(50,-100) 
        else:
            
            ax1sec.spines['bottom'].set_bounds(150,-50) 
            
        
        
        ax.set_xlim([70,-50])
    y_low = 0
    axArr[0].set_yticks([y_low, 0.25, 0.5, 0.75, 1]) 
    axArr[0].set_ylim([y_low-(1-y_low)/20,1+(1-y_low)/20])
    axArr[0].set_yticklabels([rf"${val:.2f}$" for val in [y_low, 0.25, 0.5, 0.75, 1]])
    axArr[0].spines["left"].set_visible(True)
    axArr[0].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    # axArr[-1].tick_params(axis='x', which='both', top=False, bottom=True)
    # axArr[-1].spines["bottom"].set_visible(True)
    
    
    
   
    norm = plt.Normalize(vmin=min(lambdaArr), vmax=max(lambdaArr))
    sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm)
    # cbar = fig3.colorbar(sm, ax=axArr4, orientation='horizontal', label=r'wavelength, $\lambda$ (nm)', aspect = 52)
    # cbar.set_ticks(lambdaArr)
    # cbar.set_ticklabels([rf'${l:.0f}$' for l in lambdaArr], fontsize = numfontsize)
    # cbar.set_label(label=r'wavelength, $\lambda$ (nm)', size = labelfontsize)
    
    # pos = axArr[-1].get_position()
    # cbar.ax.set_position([pos.x0-0.04, pos.y1 - 0.45, pos.width+0.2, 0.06])  # [x, y, width, height]
    
    # # Move colorbar above (by adjusting position)
    # cbar.ax.xaxis.set_ticks_position('bottom')
    # cbar.ax.xaxis.set_label_position('bottom')
    plt.tight_layout()
    
    
    
    ax = axArr3
    ax.spines["bottom"].set_bounds([1e0, 1e6])
    ax.spines["left"].set_bounds([0, 50])
    ax.set_ylim([-3, 50])
    ax.set_xlim([0.5, 1e6])
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([0, 10,20,30,40,50]) 
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', top=False, bottom=False)
    ax.spines["bottom"].set_visible(False)
    # ax.set_xscale('log')
    # ax.set_xticks([1e0, 1e2, 1e4, 1e6]) 
        
    
    # axArr3[-1].set_xticks([1, 100, 1e4, 1e6])
    # axArr3[-1].tick_params(axis='x', which='major', top=False, bottom=True)
    # axArr3[-1].spines["bottom"].set_visible(True)
    
    ax = axArr4
    ax.spines["bottom"].set_bounds([1e0, 1e6])
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='major', top=False, bottom=True)
    ax.tick_params(axis='x', which='minor', top=False, bottom=False)
    ax.set_yticks([0,6,12]) 
    ax.set_xlim([0.5, 1e6])
    ax.spines["left"].set_bounds([0, 12])
    ax.set_ylim([-2, 12])
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.yaxis.set_ticks_position('right')
    # ax.yaxis.set_label_position('right')
    ax.spines["bottom"].set_visible(True)
    ax.set_xscale('log')
    ax.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]) 
    
        
    # axArr4[-1].set_xticks([1, 100, 1e4, 1e6])
    # axArr4[-1].tick_params(axis='x', which='major', top=False, bottom=True)
    # axArr4[-1].spines["bottom"].set_visible(True)
        
    
    
    fig2.subplots_adjust(wspace=0.3)
    fig3.subplots_adjust(hspace=0.1)
    
    if saveBool:
        fig2.savefig(
            "./figures/" + "SI_transmissionFitsNew.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        fig3.savefig(
            "./figures/" + "SI_kNew.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()


def plotSI_1048KappasNew(saveBool = False):
    # === File and plotting settings ===
    
    
    etaFibre = np.array(      [0.3, 0.3, 0.39, 0.39, 0.33, 0.43, 0.39, 0.27])
    etaUncertainty = np.array([0.08, 0.08, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02])
    ng1047 = 2.4   
    
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1267nm_Thermo_IRPowerSweep.mat'
    ]
    doubletBool = [True, True]
    labels = ['1048', '1267']
    # lambdaArr = np.array([1047, 1604])
    lambdaArr = np.array([979, 1047, 1267, 1322, 1383, 1450, 1524, 1604])
    
    etaFibre = np.array([0.3,0.39])
    etaUncertainty = np.array([0.08,0.03])
    
    xmin = -50
    xmax = 50
    ms = 1
    msa = 1
    # === Colormaps ===
    numFiles = len(filenames)
    markers = ['.'] * numFiles
    markersa = ['.'] * numFiles
    cpalette_r = 'hot_r'
    
    cmap_full_r = plt.colormaps[cpalette_r]
    cmap_r = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full_r(np.linspace(0.4, 0.9, 256))
    )
    normLambda = mcolors.Normalize(vmin=979, vmax=1604)
    
    
    
    # === Set up plots ===
    fig, axArr = plot_custom(
       OpticaWidthFull*0.22, 0.42*OpticaWidthFull, 
       r"$\mathcal{N}_{\text{cav}}$", 
       r'',#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       # widthBool =True, widthRatios=[1], heightRatios = [8.3, 3.2]
    )
    
    
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    func_map = {
        979: NtoI979,
        1047: NtoI1047,
        1267: NtoI1267,
        1322: NtoI1322,
        1383: NtoI1383,
        1450: NtoI1450,
        1524: NtoI1524,
        1604: NtoI1604,
    }
    
    for i, fname in enumerate(filenames):
        
        
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    for i, fname in enumerate(filenames):
        lamb = lambdaArr[i]
        try:
            NtoI = func_map[lamb]
        except KeyError:
            raise ValueError(f"No N→I function defined for λ = {lamb}")
        
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
        dNcavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            dNcava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
            
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        dws = 2 * np.pi * c_const * dlambdas / lambdas**2
        
        # plt.plot(lambdas)
        
    
        colorArray_r_false = [cmap_r(normLambda(lmbd)) for lmbd in [1047, 1267]]
        wa = 2 * np.pi * c_const / lambdaa
        dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
        g = ws - wa

        kips = ws / Qips
        kipa = wa / Qipa
        kexs = ws / Qexs
        kexa = wa / Qexa

        dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
        dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)

        ks = kips + kexs
        ka = kipa + kexa
        
        
        norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
        
        for j, entry in enumerate(deviceSweepData):
            # skip = int(np.floor(len(np.array(entry['W']))/200))
            
            Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
            [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                          cT[j],  Delta - g[j]/2, power = powerW[j])
            
            Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
            dNcavs[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_s)

            
            [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                          cT[j],  Delta + g[j]/2, power = powerW[j])
            
            Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
            dNcava[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                    
        Ncavs = np.real(Ncavs)
        kips = np.real(kips)
        dNcavs = np.real(dNcavs)
        dkips = np.real(dkips)
        Ncava = np.real(Ncava)
        kipa = np.real(kipa)
        dNcava = np.real(dNcava)
        dkipa = np.real(dkipa)
        
        
        Ncavs = np.abs(Ncavs + Ncava)/2
        kips = np.abs(kipa + kips)/2
        dkips = np.abs(dkipa + dkips)/np.sqrt(2)
        
        
        # initial guesses for parameters [p1, p2, p3]
        p0 = [7.5, 4, 8.7]    
        
        logN = np.log10(Ncavs)
        dlogN = dNcavs / (Ncavs * np.log(10))
        
        popt, pcov = curve_fit(saturable_model_logI, logN, kips*GHzprad, 
                               sigma = dkips*GHzprad ,p0=p0, absolute_sigma=False)
        p1_fits, p2_fits, p3_fits = popt
        perr = np.sqrt(np.diag(pcov))
   
        
        p0 = [7.5, 10**4, 8.7]    
        
        y = kips * GHzprad
        sy = dkips * GHzprad
            
        model = odr.Model(saturable_model_logI_odr)
        data = odr.RealData(Ncavs, y, sx=dNcavs, sy = np.mean(sy))
        
        
        odr_fit = odr.ODR(data, model, beta0=p0, maxit=1000)
        out = odr_fit.run()
        # out.pprint()
        
        popt = out.beta
        p1_fits, p2_fits, p3_fits = popt
        perr = out.sd_beta
        
        print('\n')
        print("Wavelength: ", labels[i])
        print("Fitted parameters:", popt)
        print("Uncertainties:", perr)
        print('Isats:', NtoI(p2_fits))
        p1s = popt[0]
        N = np.logspace(0, 6, 1000)
        # axArr.plot(N, saturable_model_logI(np.log10(N), *popt), lw = lw,  color = 'k', zorder = 10 )
        axArr.plot(N, saturable_model_logI_odr(out.beta,N), lw = 1,  color = colorArray_r_false[i], zorder = 10 )
        axArr.errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
                    marker=markers[i], linestyle = 'none', label=labels[i], 
                    xerr=dNcavs,
                    color=colorArray_r_false[i], ms = ms, lw = lw, markeredgewidth=0.5)
        p0 = [7.5, 4, 8.7]   
        popt, pcov = curve_fit(saturable_model_logI, np.log10(Ncava),  np.real(kipa*GHzprad), p0=p0)
        p1_fita, p2_fita, p3_fita = popt
        perr = np.sqrt(np.diag(pcov))
        
        p0 = [7.5, 10**4, 8.7]    
        
        y = kipa * GHzprad
        sy = dkipa * GHzprad
            
        model = odr.Model(saturable_model_logI_odr)
        data = odr.RealData(Ncavs, y, sx=dNcava, sy = np.mean(sy))
        
        
        odr_fit = odr.ODR(data, model, beta0=p0, maxit=1000)
        out = odr_fit.run()
        # out.pprint()
        
        popt = out.beta
        p1_fita, p2_fita, p3_fita = popt
        perr = out.sd_beta

        p1a = popt[0]
        
        kappaabs = (p1a+p1s)/2 
        dkappaabs =  np.abs(kappaabs - p1a)
        print('Kappa: ', kappaabs)
        print('dKappa: ', dkappaabs)
        print('\n')
        
            
    
    # === Add colorbar for wavelength ===
    
    
    plt.tight_layout()
    # i = 0;
    ax = axArr
    y_low, y_high = ax.get_ylim()
    x_low, x_high = ax.get_xlim()
    y_low = p3_fita
    y_high = p3_fita+kappaabs+dkappaabs
    
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(r'internal cavity loss, $\kappa_{\text{c}}/2\pi$ (GHz)')
    
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='y', which='both', left=True, right=True, labelleft=False, labelright = True)
    ax.spines['bottom'].set_bounds(1e1,1e6)
    ax.set_xlim([1e1,1e6])
    ax.set_xscale('log')
    NArr = np.array([100, 1e3, 1e4,1e5, 1e6])
    ax.set_xticks(NArr)
    
        
    # Create the secondary x-axis
    ax1sec = axArr.secondary_xaxis('top', functions=(NtoI1047, ItoN1047))
    ax1sec.set_xlabel(r"$\langle I\rangle$ (MW/cm$^2$)",fontsize = labelfontsize, color = colorArray_r_false[0])
    
    ax = ax1sec
    xmin = np.min(NtoI1047( NArr))
    xmax = np.max(NtoI1047( NArr))
    xrng = xmax-xmin
    ax.spines['top'].set_bounds(xmin,xmax)
    ax.xaxis.set_ticks_position('top')
    ax.set_xlim([xmin-xrng/10,xmax+xrng/10])

    ax.tick_params(
        axis="x",
        which="major",
        width=0.8,
        length=2.5,
        labelsize=numfontsize,
        zorder=1,
        direction="in",
        top="off",
        colors=colorArray_r_false[0]
    )
    ax.tick_params(
        axis="x",
        which="minor",
        width=0.8,
        length=0,
        labelsize=numfontsize,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )    
    axArr.tick_params(
            axis="x",
            which="minor",
            width=0.8,
            length=0,
            labelsize=numfontsize,
            zorder=1,
            direction="in",
            top="off",
            colors='k'
        )
    fig.subplots_adjust(hspace=0.2)
    
    if saveBool:
        fig.savefig(
            "./figures/" + "SI_kappa1048New.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotAllFitsKappaIn(saveBool = False):
    # === File and plotting settings ===
    
    
    etaFibre = np.array(      [0.3, 0.3, 0.39, 0.39, 0.33, 0.43, 0.39, 0.27])
    etaUncertainty = np.array([0.08, 0.08, 0.03, 0.03, 0.02, 0.01, 0.02, 0.02])
    
    msFits = 4
    

    
    
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_979nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1267nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1322nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1383nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1450nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1525nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1604nm_Thermo_IRPowerSweep.mat'
    ]

    doubletBool = [False, True, True, True, True, False, False, False]
    labels = ['979','1048','1267','1322','1383','1450','1524','1604']
    lambdaArr = np.array([979, 1047, 1267, 1322, 1383, 1450, 1524, 1604])
    
    
    
    p0Arr = np.array([[7, 4e3, 22], 
                      [7, 4e3, 9],
                      [7.5, 1e4, 8.7],
                      [7.5, 1e4, 8.7],
                      [7.5, 1e4, 8.7],
                      [7.5, 1e4, 8.7],
                      [7.5, 1e4, 8.7],
                      [7.5, 1e4, 8.7]])
    
    xmin = -50
    xmax = 50
    ms = 1
    # === Colormaps ===
    numFiles = len(filenames)
    markers = ['d'] * numFiles
    cpalette_r = 'hot_r'
    
    norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    
    cmap_full_r = plt.colormaps[cpalette_r]
    cmap_r = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full_r(np.linspace(0.4, 0.9, 256))
    )
    
    colorArray_r_false = [cmap_r(norm(lmbd)) for lmbd in lambdaArr]
    # norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    # colorArray_r = [cmap_r(norm(lmbd)) for lmbd in lambdaArr]
    
    # === Set up plots ===
    fig, axArr = plot_custom(
       OpticaWidthFull*0.3, 0.47*OpticaWidthFull, 
       r"intracavity photons, $\mathcal{N}_{\text{cav}}$", 
       r'internal cavity loss, $\kappa_{\text{c}}/2\pi$ (GHz)',#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool = False
    )
    
    # === Set up plots ===
    fig2, ax2 = plot_custom(
       OpticaWidthHalf*0.55, 0.24*OpticaWidthFull, 
       r"wavelength, $\lambda$ (nm)", 
       [r'$\alpha_0$ (cm$^{-1})$',r"$I_{\text{sat}}$ (MW/cm$^2$)"],#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool = False
    )
    
    leftaxiscolor = 'r'
    # ax2.set_ylabel(ax2.get_ylabel(), color = leftaxiscolor)
    rightaxiscolor = 'b'
    # ax2sec = ax2.twinx()
    # ax2sec.set_ylabel(r"$I_{\text{sat}}$ (MW/cm$^2$)",fontsize = labelfontsize,color = rightaxiscolor)
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    func_map = {
        979: NtoI979,
        1047: NtoI1047,
        1267: NtoI1267,
        1322: NtoI1322,
        1383: NtoI1383,
        1450: NtoI1450,
        1524: NtoI1524,
        1604: NtoI1604,
    }
    for i, fname in enumerate(filenames):
        lamb = lambdaArr[i]
        try:
            NtoI = func_map[lamb]
        except KeyError:
            raise ValueError(f"No N→I function defined for λ = {lamb}")
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
        dNcavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            dNcava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if lambdaArr[i] < 1000 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
            
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        dws = 2 * np.pi * c_const * dlambdas / lambdas**2
        
        # plt.plot(lambdas)
        
    
        if doubletBool[i]:
            wa = 2 * np.pi * c_const / lambdaa
            dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
            g = ws - wa
    
            kips = ws / Qips
            kipa = wa / Qipa
            kexs = ws / Qexs
            kexa = wa / Qexa
    
            dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)
    
            ks = kips + kexs
            ka = kipa + kexa
            
            p0 = p0Arr[i]
            
            norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                TOut = np.array(entry['T'])[0:-1:skip]
                
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j],  Delta - g[j]/2, power = powerW[j])
                
                Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                dNcavs[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_s)

                
                [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                              cT[j],  Delta + g[j]/2, power = powerW[j])
                
                Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                dNcava[j] = calcdNcav(etaFibre[i], etaUncertainty[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                            
            Ncavs = np.real(Ncavs)
            kips = np.real(kips)
            dNcavs = np.real(dNcavs)
            dkips = np.real(dkips)
            Ncava = np.real(Ncava)
            kipa = np.real(kipa)
            dNcava = np.real(dNcava)
            dkipa = np.real(dkipa)
            Imax = NtoI(np.max(Ncavs))
            Imin = NtoI(np.min(Ncavs))
            
            Ncav = np.abs(Ncavs + Ncava)/2
            kip = np.abs(kipa + kips)/2
            dkip = np.abs(dkipa + dkips)/np.sqrt(2)
            # initial guesses for parameters [p1, p2, p3]
            try:
                
                N = np.logspace(0, 6, 1000)
                if lambdaArr[i] < 1300:
                    
                    # initial guesses for parameters [p1, p2, p3]
                    p0 = [7.5, 4, 8.7]    
                    
                    logN = np.log10(Ncav)
                    
                    popt, pcov = curve_fit(saturable_model_logI, logN, (kip*GHzprad), p0=p0)
                    p1_fits, p2_fits, p3_fits = popt
                    perr = np.sqrt(np.diag(pcov))
               
                    
                    p0 = [p1_fits, 10**p2_fits, p3_fits]    
                    
                    y = kip* GHzprad
                    sy = dkip * GHzprad
                        
                    model = odr.Model(saturable_model_logI_odr)
                    data = odr.RealData(Ncav, y)
                    
                    
                    odr_fit = odr.ODR(data, model, beta0=p0, maxit=1000)
                    out = odr_fit.run()
                    # out.pprint()
                    
                    poptODR = out.beta
                    
                    
                    perrODR = out.sd_beta
                    k0, Nsat, k_other = poptODR
                    k0Err, NsatErr, k_otherErr = out.sd_beta
                    
                    msigma = k0/(c_const/1e9/nd)
                    msigmaerr = k0Err/(c_const/1e9/nd)
                    Isat = NtoI(Nsat)
                    Isaterr = NtoI(NsatErr)
                    
                    print('\n')
                    print("Wavelength: ", labels[i])
                    print("Fitted parameters:", poptODR)
                    print("Uncertainties:", perrODR)
                    print('Msigma: ', msigma, ' pm ', msigmaerr)
                    print('Isats:', Isat, ' pm ', Isaterr)
                    # p1s = popt[0]
                    N = np.logspace(0, 6, 1000)

                    
                    axArr.plot(N, saturable_model_logI(np.log10(N), *popt), lw = lw, color = colorArray_r_false[i] )
                    axArr.plot(N, saturable_model_logI_odr(poptODR,N), lw = lw, color = colorArray_r_false[i], zorder = 10 )
                    ax2[0].errorbar(lambdaArr[i], msigma/100, yerr=msigmaerr/100,
                                marker=markers[i], linestyle = 'none', label=labels[i], 
                                color=colorArray_r_false[i], ms = msFits, lw = lw, markeredgewidth=0.5, capsize = 3)
                    ax2[1].errorbar(lambdaArr[i], Isat, yerr=Isaterr,
                                marker=markers[i], linestyle = 'none', label=labels[i], 
                                color=colorArray_r_false[i], ms = msFits, lw = lw, markeredgewidth=0.5, capsize = 3)
                # elif lambdaArr[i] < lambdah:
                # #     ax2.fill_between([Imax, 1000*Imax], [100,100], [-100, -100], color = colorArray_r_false[i], alpha = 1, lw = 0)
                # #     ax2.fill_between([Imin/1e3, Imin], [100,100], [-100, -100], color = colorArray_r_false[i], alpha = 1, lw = 0)
                #     ax2sec.plot([lambdaArr[i], lambdaArr[i]], [Imin, 0], linestyle = '-', label=labels[i], 
                #                 color=rightaxiscolor, lw = 2,zorder = -1, alpha = 0.5)
                    
            except:
                # ax2.fill_between([Imax, 1000*Imax], [100,100], [-100, -100], color = colorArray_r_false[i], alpha = 1, lw = 0)
                print('could not fit')
            axArr.errorbar(Ncav, kip*GHzprad, yerr=dkip*GHzprad,
                        marker=markers[i], linestyle = 'none', label=labels[i], 
                        xerr=(dNcavs + dNcava)/np.sqrt(2),
                        color=colorArray_r_false[i], ms = ms, lw = lw, markeredgewidth=0.5)

            
            kappaabs = (k0+k0)/2 
            # dkappaabs =  np.abs(kappaabs - p1s)
            NtoI = func_map[lamb]
            # print('Wavelengt: ', lamb)
            # print('Max Intensity: ', NtoI(np.max(Ncav)))
            # print('Min Intensity: ', NtoI(np.min(Ncav)))
            # print('Kappa: ', kappaabs)
            # print('dKappa: ', dkappaabs)
            print('\n')
            
        else:
            
            kips = ws / Qips
            kexs = ws / Qexs
            
            dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            
            ks = kips + kexs
            
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9  
                TOut = np.array(entry['T'])[0:-1:skip] 
                filt = np.logical_and((w-np.mean([ws[j], ws[j]]))*GHzprad > xmin,(w-np.mean([ws[j], ws[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j]/2,  Delta, power = np.sqrt(etaFibre[i])*powerW[j])
                
                Ncavs[j] = np.max(np.sqrt(etaFibre[i])*(powerW[j])/(hbar*((ws[j]) + Delta))*np.abs(a_s)**2)
                dNcavs[j] = np.max(np.sqrt( ((etaFibre[i])**-0.5*etaUncertainty[i]*(powerW[j]))**2 + (np.sqrt(etaFibre[i])*(powerW[j]*0.05))**2)/(hbar*(np.mean([ws, ws]) + Delta))*np.abs(a_s)**2)

        
            Ncavs = np.real(Ncavs)
            kips = np.real(kips)
            dNcavs = np.real(dNcavs)
            dkips = np.real(dkips)
            Imax = NtoI(np.max(Ncavs))
            Imin = NtoI(np.min(Ncavs))
            
            Ncav = np.abs(Ncavs)
            kip = np.abs(kips)
            dkip = np.abs(dkips)
            # initial guesses for parameters [p1, p2, p3]
            try:
                
                N = np.logspace(0, 6, 1000)
                if lambdaArr[i] < 1200:
                    
                    # initial guesses for parameters [p1, p2, p3]
                    p0 = [7.5, 4, 8.7]    
                    
                    logN = np.log10(Ncav)
                    
                    popt, pcov = curve_fit(saturable_model_logI, logN, (kip*GHzprad), p0=p0)
                    p1_fits, p2_fits, p3_fits = popt
                    perr = np.sqrt(np.diag(pcov))
               
                    
                    p0 = [p1_fits, 10**p2_fits, p3_fits]    
                    
                    y = kip* GHzprad
                    sy = dkip * GHzprad
                        
                    model = odr.Model(saturable_model_logI_odr)
                    data = odr.RealData(Ncav, y)
                    
                    
                    odr_fit = odr.ODR(data, model, beta0=p0, maxit=1000)
                    out = odr_fit.run()
                    # out.pprint()
                    
                    poptODR = out.beta
                    
                    
                    perrODR = out.sd_beta
                    k0, Nsat, k_other = poptODR
                    k0Err, NsatErr, k_otherErr = out.sd_beta
                    
                    msigma = k0/(c_const/1e9/nd)
                    msigmaerr = k0Err/(c_const/1e9/nd)
                    Isat = NtoI(Nsat)
                    Isaterr = NtoI(NsatErr)
                    
                    print('\n')
                    print("Wavelength: ", labels[i])
                    print("Fitted parameters:", poptODR)
                    print("Uncertainties:", perrODR)
                    print('Msigma: ', msigma, ' pm ', msigmaerr)
                    print('Isats:', Isat, ' pm ', Isaterr)
                    # p1s = popt[0]
                    N = np.logspace(0, 6, 1000)

                    
                    if lambdaArr[i] == 1524:
                        filt = (N < 5e4)
                    else:
                        filt = (N < 1e10)
                    axArr.plot(N[filt], saturable_model_logI(np.log10(N[filt]), *popt), lw = lw, color = colorArray_r_false[i] )
                    axArr.plot(N[filt], saturable_model_logI_odr(poptODR,N[filt]), lw = lw, color = colorArray_r_false[i], zorder = 10 )
                    ax2[1].errorbar(lambdaArr[i], Isat, yerr=Isaterr,
                                marker=markers[i], linestyle = 'none', label=labels[i], 
                                color=colorArray_r_false[i], ms = msFits, lw = lw, markeredgewidth=0.5, capsize = 3,zorder = -1)
                    ax2[0].errorbar(lambdaArr[i], msigma/100, yerr=msigmaerr/100,
                                marker=markers[i], linestyle = 'none', label=labels[i], 
                                color=colorArray_r_false[i], ms = msFits, lw = 0.5, markeredgewidth=0.5, capsize = 3,zorder = 10)
                
                # elif lambdaArr[i] < lambdah:
                #     ax2sec.plot([lambdaArr[i], lambdaArr[i]], [Imin, 0], linestyle = '-', label=labels[i], 
                #                 color=rightaxiscolor, lw = 2,zorder = -1, alpha = 0.5)
                    # ax2sec.fill_between([Imax, 1000*Imax], [100,100], [-100, -100], color = colorArray_r_false[i], alpha = 1, lw = 0)
                    
                
                if lambdaArr[i] == 1524:
                    filt = (Ncav < 5e4)
                else:
                    filt = (Ncav < 1e10)
                axArr.errorbar(Ncav[filt], kip[filt]*GHzprad, yerr=dkip[filt]*GHzprad,
                            marker=markers[i], linestyle = 'none', label=labels[i], 
                            xerr=dNcavs[filt],
                            color=colorArray_r_false[i], ms = ms, lw = lw, markeredgewidth=0.5)
                # kappaabs = (p1s)
                # # dkappaabs =  np.abs(kappaabs - p1s)
                # print('Kappa: ', k0)
                # # print('dKappa: ', dkappaabs)
                # print('\n')
            except:
                # ax2.fill_between([Imax, 1000*Imax], [100,100], [-100, -100], color = colorArray_r_false[i], alpha = 1, lw = 0)

                print('could not fit')
          
        NtoI = func_map[lamb]
        # print('Wavelengt: ', lamb)
        # print('Max Intensity: ', NtoI(np.max(Ncav)))
        # print('Min Intensity: ', NtoI(np.min(Ncav)))
            
    # ax2sec.fill_between([lambdah, 1700], [1,1], [0, 0], color = 'lightgray', alpha = 1, lw = 0, zorder = -1)
    

    plt.tight_layout()
    # i = 0;
    ax = axArr
    
    # i += 1
    # ax.spines['bottom'].set_bounds(1e-0, 1e6)
    # y_low, y_high = ax.get_ylim()
    # x_low, x_high = ax.get_xlim()
    # if i == 1:
    #     y_low = p3_fita
    #     y_high = p3_fita+kappaabs+dkappaabs
    
    # ax.set_yticks([5, 10,15,20,25,30,35,40,45,50])
    # ax.spines['left'].set_bounds(5,50)
    ax.set_ylim([5,50])
    ax.set_xlim([0.5e0,2e6])
    # ax.set_yticks([y_low, y_high])
    # ax.set_yticklabels([rf"${val:.1f}$" for val in [y_low, y_high]])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    # ax.spines["bottom"].set_visible(False)
    ax.spines['bottom'].set_bounds(1e0,1e6)
    
    ax.set_xscale('log')
    NArr = np.array([1, 100, 1e4, 1e6])
    ax.set_xticks(NArr)
    
    NArr = [979, 1047, 1267]
    ax2[0].set_xticks(NArr)
    ax2[0].set_xticks([986, 1355], minor = True)
    
    # ax2.set_xlim([0.1,1])
    ax2[0].set_ylim([0,0.6])
    ax2[1].set_ylim([0,4])
    ax2[0].set_yticks([0,0.6])
    ax2[0].spines['left'].set_bounds(0,0.6)
    # ax2.set_xscale('log')
    # ax2.set_xscale('log')
    
    
    # axArr[-1].set_ylim([y_low-(y_high-y_low)/6,y_high+(y_high-y_low)/6])
    # axArr[-1].spines["bottom"].set_visible(True)
        
    # Create the secondary x-axis
    # ax1sec = axArr[0].secondary_xaxis('top', functions=(NtoI1047, ItoN1047))
    # ax1sec.set_xlabel(r"$I$ (MW/cm$^2$)",fontsize = labelfontsize)
    
    
    # Tick marks + tick labels
    # ax2.tick_params(axis='y', colors=leftaxiscolor)
    
    # Axis spine
    # ax2.spines['left'].set_color(leftaxiscolor)
    # ax2sec.spines['left'].set_color(leftaxiscolor)
    # ax2sec.spines['right'].set_color(rightaxiscolor)
    
    # Optional: offset text (e.g. ×10^n)
    # ax2.yaxis.get_offset_text().set_color(leftaxiscolor)
    # ax2sec.yaxis.get_offset_text().set_color(rightaxiscolor)
    
    
    # ax = ax2sec
    ax2[1].set_xlim([960, 1370])
    # ax2[1].set_ylim([0, 60])
    # ax2sec.set_ylim([0, 1])
    xrng = xmax-xmin
    # ax.spines['top'].set_bounds(xmin,xmax)
    # ax.yaxis.set_ticks_position('right')
    # ax2sec.tick_params(axis='y', colors=rightaxiscolor)
    # xticks = np.array([xmin, xmax])
    # ax.set_xticks(xticks)
    # ax.set_yticklabels([rf"${val:.1f}$" for val in xticks])
    # ax.set_xlim([xmin-xrng/10,xmax+xrng/10])

    alpha0 = lambda alpha0, Del, T2, omega: alpha0*(1 + Del/omega)/(1+ Del**2*T2**2)
    I0 = lambda I, Del, T2: I*(1+ Del**2*T2**2)
    
    lamb = np.linspace(600, 1700, 1000)
    omega = wavelength_to_freq(lamb)  
    
    omega0 = wavelength_to_freq(986)
    T2 = 0.26e-14
    alphaDel = alpha0(55, (omega - omega0)*1e12*2*np.pi, T2, omega0*1e12*2*np.pi )
    I0Del = I0(0.75, (omega - omega0)*1e12*2*np.pi, T2)
    
    # ax2[0].plot(lamb, alphaDel, 'k', lw = 0.5, zorder = -1)
    # ax2[1].plot(lamb, I0Del, 'k', lw = 0.5, zorder = -1)
    
    omega0 = wavelength_to_freq(1359)
    T2 = 10e-14
    alphaDel = alpha0(20, (omega - omega0)*1e12*2*np.pi, T2, omega0*1e12*2*np.pi )
    I0Del = I0(0.25, (omega - omega0)*1e12*2*np.pi, T2)
    
    # ax2[0].plot(lamb, alphaDel, 'k', lw = 0.5, zorder = -1)
    # ax2[1].plot(lamb, I0Del, 'k', lw = 0.5, zorder = -1)
    
    
    
    # Delta979 = wavelength_to_freq(979) - wavelength_to_freq(986)
    # Delta1047 = wavelength_to_freq(1047) - wavelength_to_freq(986)
    # n0 = 2.4
    # n2 = -0.76e-20
    # Is979 =0.81e10
    # Is1047=0.81e10
    # alpha0 = 56
    
    # T2979 = 8*n0*n2/c*Is979/alpha0*wavelength_to_freq(986)/Delta979
    # T21047 = 8*n0*n2/c*Is1047/alpha0*wavelength_to_freq(986)/Delta1047
    
    # ax.tick_params(
    #     axis="y",
    #     which="major",
    #     width=0.8,
    #     length=2,
    #     labelsize=numfontsize,
    #     zorder=1,
    #     direction="in",
    #     top="off",
    #     colors=rightaxiscolor
    # )
    
    # ax.set_yticks([y_low, y_high])
    # ax.set_yticklabels([rf"${val:.1f}$" for val in [y_low, y_high]])
    for ax in ax2:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=True)
        ax.spines['bottom'].set_bounds(979, 1355)
        ax.set_xlim([940, None])
        ax.set_ylim([-0.08, None])
        
    ax2[1].set_yticks([0, 4])
    ax2[1].set_ylim([-0.8, None])
    ax2[1].spines['left'].set_bounds(0, 4)
    ax2[0].tick_params(axis='x', which='both', top=False, bottom=False)
    ax2[0].spines["bottom"].set_visible(False)
    
    fig2.subplots_adjust(hspace=0.2)
    
    if saveBool:
        fig.savefig(
            "./figures/" + "allKappaFits.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        fig2.savefig(
            "./figures/" + "fittedValues.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()

def plotSI_1048Transmission(saveBool = False):
    # === File and plotting settings ===
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1604nm_Thermo_IRPowerSweep.mat',
    ]
    doubletBool = [True, False]
    lambdaArr = np.array([1048, 1604])
    
    etaFibre = np.array([0.5, 1604])
    
    xmin = -50
    xmax = 50
    
    # === Colormaps ===
    numFiles = len(filenames)
    # markers = ['.'] * numFiles
    cpalette = 'hot'
    
    cmap_full = cm.get_cmap(cpalette)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full(np.linspace(0, 0.7, 256))
    )
    
    norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    
    # === Set up plots ===
    fig, axArr = plot_custom(
       OpticaWidthFull*0.25, 0.30*OpticaWidthFull, 
       r"detuning, $\Delta\omega/2\pi$ (GHz)", 
       [r'',r''],#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = numFiles, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True, widthRatios = [1], heightRatios = [1-0.35, 0.5]
    )
    
    
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        # dws = 2 * np.pi * c_const * dlambdas / lambdas**2
        
        # plt.plot(lambdas)
        
    
        if doubletBool[i]:
            wa = 2 * np.pi * c_const / lambdaa
            # dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
            g = ws - wa
    
            kips = ws / Qips
            kipa = wa / Qipa
            kexs = ws / Qexs
            kexa = wa / Qexa
    
            # dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            # dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)
    
            ks = kips + kexs
            ka = kipa + kexa
            
            norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            # cpaletteT = 'viridis'
            # cmapT = cm.get_cmap(cpaletteT)  # or 'plasma', 'inferno', etc.
            colorArrayT = [cmap(norm(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], wa[j]]))*GHzprad > xmin,(w-np.mean([ws[j], wa[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j],  Delta - g[j]/2, power = powerW[j])
                
                Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                
                
                [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                              cT[j],  Delta + g[j]/2, power = powerW[j])
                
                Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                
                T = calcTransmission(kexs[j], a_s, kexa[j], a_a, tau = tau[j]*np.exp(1j*phi[j]))
                
                skip = 5
                
                if np.mod(j, 2) == 0:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw/5)
                    axArr[i].scatter((wMeas[1:-1:skip]-np.mean([ws[j], wa[j]])/1e9)/2/np.pi,TOut[1:-1:skip]/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2, zorder = 10)
                
                
        
            # axArr[i].errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
            #             marker=markers[i], linestyle = 'none', label=labels[i], 
            #             color=colorArray_r[i], ms = ms, markeredgewidth=0.5)
            # axArr[i].errorbar(Ncava, kipa*GHzprad, yerr=dkipa*GHzprad,
            #             marker=markers[i], linestyle = 'none', color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)

            
        else:
    
            kips = ws / Qips
            kexs = ws / Qexs
            
            # dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            
            ks = kips + kexs
            
            normT = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            colorArrayT = [cmap(normT(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9  
                TOut = np.array(entry['T'])[0:-1:skip] 
                filt = np.logical_and((w-np.mean([ws[j], ws[j]]))*GHzprad > xmin,(w-np.mean([ws[j], ws[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                wMeas = (w)/1e9 
                
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j]*0.00000001,  Delta, power = np.sqrt(etaFibre[i])*powerW[j])
                
                Ncavs[j] = np.max(np.sqrt(etaFibre[i])*(powerW[j])/(hbar*((ws[j]) + Delta))*np.abs(a_s)**2)
                
              
                
                T = calcTransmission(kexs[j], a_s, kexs[j], a_s, tau = tau[j]*np.exp(1j*phi[j]))
                
                 
                skip = 1
                
                if np.mod(j, 5) == 0:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    # if j < 60:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,5) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw/2 )
                    axArr[i].scatter((wMeas[0:-1:skip]-ws[j]/1e9)/2/np.pi,TOut[0:-1:skip]/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2)
               # colorArrayT[j]
               
              
    
            # axArr[i].errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
            #             marker=markers[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
           
          
            
    
    # === Add colorbar for wavelength ===
    
    norm = plt.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axArr, orientation='vertical', pad=0.15)
    cbar.set_label(r"input power, $P$ (dBm)", labelpad=-15, fontsize = labelfontsize)
    # Move it above the top axes manually
    pos = axArr[-1].get_position()
    cbar.ax.set_position([pos.x0+0.85, pos.y1-0.27, 0.1, 0.74])  # [x, y, width, height]
    # Move colorbar above (by adjusting position)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_ticks(np.log10([powerMin, powerMax]))
    cbar.set_ticklabels([rf'${l:.1f}$' for l in 10*np.log10(np.array([powerMin, powerMax])*1e3)], fontsize = numfontsize)
    # cbar.set_ticklabels([r'$10^{-7.0}$', r'$10^{-1.2}$'], fontsize = numfontsize)
    plt.tight_layout()
    
    
    axArr[0].set_ylim([0.38,1])
    axArr[0].spines['left'].set_bounds(0.38,1)
    axArr[1].set_ylim([0.5,1])
    axArr[1].spines['left'].set_bounds(0.5,1)
    
    for i, ax in enumerate(axArr):
        # ax.set_ylim([None,1])
        ax.set_xlim([-50, 50])
        y_low, y_high = ax.get_ylim()
        
        ax.spines['left'].set_bounds(y_low,1)
        if i == 0:
            ax.set_yticks([y_low,0.52, 1])
            ax.set_yticklabels([rf"${val:.2f}$" for val in [y_low, 0.52, 1]])
        ax.set_ylim([y_low-(1-y_low)/6,1+(1-y_low)/20])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.spines['bottom'].set_bounds(-50,50)
        
        ax.set_xlim([-55, 50])
        ax.set_xticks([-50,0,50])
    
    axArr[-1].tick_params(axis='x', which='both', top=False, bottom=True)
    axArr[-1].spines["bottom"].set_visible(True)
    
    
    fig.text(-0.02, 0.53, r'transmission, $T$ (\%)', va='center', rotation='vertical', fontsize = labelfontsize)
    
    
    fig.subplots_adjust(hspace=0.05)
    
    if saveBool:
        fig.savefig(
            "./figures/" + "SI_transmission1048.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotSI_1048TransmissionQ2(saveBool = False):
    # === File and plotting settings ===
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1604nm_Thermo_IRPowerSweep.mat',
    ]
    doubletBool = [True, False]
    lambdaArr = np.array([1048, 1604])
    
    etaFibre = np.array([0.5, 1604])
    
    xmin = -50
    xmax = 50
    
    # === Colormaps ===
    numFiles = len(filenames)
    # markers = ['.'] * numFiles
    cpalette = 'viridis'
    
    cmap_full = cm.get_cmap(cpalette)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full(np.linspace(0, 1, 256))
    )
    
    norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    
    # === Set up plots ===
    fig, axArr = plot_custom(
       OpticaWidthFull*0.25, 0.27*OpticaWidthFull, 
       r"detuning, $\Delta\omega/2\pi$ (GHz)", 
       [r'',r''],#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = numFiles, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True, widthRatios = [1], heightRatios = [1-0.35, 0.5]
    )
    
    
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        # dws = 2 * np.pi * c_const * dlambdas / lambdas**2
        
        # plt.plot(lambdas)
        
    
        if doubletBool[i]:
            wa = 2 * np.pi * c_const / lambdaa
            # dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
            g = ws - wa
    
            kips = ws / Qips
            kipa = wa / Qipa
            kexs = ws / Qexs
            kexa = wa / Qexa
    
            # dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            # dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)
    
            ks = kips + kexs
            ka = kipa + kexa
            
            norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            # cpaletteT = 'viridis'
            # cmapT = cm.get_cmap(cpaletteT)  # or 'plasma', 'inferno', etc.
            colorArrayT = [cmap(norm(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], wa[j]]))*GHzprad > xmin,(w-np.mean([ws[j], wa[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j],  Delta - g[j]/2, power = powerW[j])
                
                Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                
                
                [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                              cT[j],  Delta + g[j]/2, power = powerW[j])
                
                Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                
                T = calcTransmission(kexs[j], a_s, kexa[j], a_a, tau = tau[j]*np.exp(1j*phi[j]))
                
                skip = 5
                
                if np.mod(j, 2) == 0:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw/5)
                    axArr[i].scatter((wMeas[1:-1:skip]-np.mean([ws[j], wa[j]])/1e9)/2/np.pi,TOut[1:-1:skip]/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2, zorder = 10)
                
                
        
            # axArr[i].errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
            #             marker=markers[i], linestyle = 'none', label=labels[i], 
            #             color=colorArray_r[i], ms = ms, markeredgewidth=0.5)
            # axArr[i].errorbar(Ncava, kipa*GHzprad, yerr=dkipa*GHzprad,
            #             marker=markers[i], linestyle = 'none', color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)

            
        else:
    
            kips = ws / Qips
            kexs = ws / Qexs
            
            # dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            
            ks = kips + kexs
            
            normT = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            colorArrayT = [cmap(normT(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9  
                TOut = np.array(entry['T'])[0:-1:skip] 
                filt = np.logical_and((w-np.mean([ws[j], ws[j]]))*GHzprad > xmin,(w-np.mean([ws[j], ws[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                wMeas = (w)/1e9 
                
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j]*0.00000001,  Delta, power = np.sqrt(etaFibre[i])*powerW[j])
                
                Ncavs[j] = np.max(np.sqrt(etaFibre[i])*(powerW[j])/(hbar*((ws[j]) + Delta))*np.abs(a_s)**2)
                
              
                
                T = calcTransmission(kexs[j], a_s, kexs[j], a_s, tau = tau[j]*np.exp(1j*phi[j]))
                
                 
                skip = 1
                
                if np.mod(j, 5) == 0:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    # if j < 60:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,5) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw/2 )
                    axArr[i].scatter((wMeas[0:-1:skip]-ws[j]/1e9)/2/np.pi,TOut[0:-1:skip]/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2)
               # colorArrayT[j]
               
              
    
            # axArr[i].errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
            #             marker=markers[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
           
          
            
    
    # === Add colorbar for wavelength ===
    
    norm = plt.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axArr, orientation='vertical', pad=0.15)
    cbar.set_label(r"input power, $P$ (dBm)", labelpad=-15, fontsize = labelfontsize)
    # Move it above the top axes manually
    pos = axArr[-1].get_position()
    cbar.ax.set_position([pos.x0+0.85, pos.y1-0.27, 0.1, 0.74])  # [x, y, width, height]
    # Move colorbar above (by adjusting position)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_ticks(np.log10([powerMin, powerMax]))
    cbar.set_ticklabels([rf'${l:.1f}$' for l in 10*np.log10(np.array([powerMin, powerMax])*1e3)], fontsize = numfontsize)
    # cbar.set_ticklabels([r'$10^{-7.0}$', r'$10^{-1.2}$'], fontsize = numfontsize)
    plt.tight_layout()
    
    
    axArr[0].set_ylim([0.38,1])
    axArr[0].spines['left'].set_bounds(0.38,1)
    axArr[1].set_ylim([0.5,1])
    axArr[1].spines['left'].set_bounds(0.5,1)
    
    for i, ax in enumerate(axArr):
        # ax.set_ylim([None,1])
        ax.set_xlim([-50, 50])
        y_low, y_high = ax.get_ylim()
        
        ax.spines['left'].set_bounds(y_low,1)
        if i == 0:
            ax.set_yticks([y_low,0.52, 1])
            ax.set_yticklabels([rf"${val:.2f}$" for val in [y_low, 0.52, 1]])
        ax.set_ylim([y_low-(1-y_low)/6,1+(1-y_low)/20])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='x', which='both', top=False, bottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.spines['bottom'].set_bounds(-50,50)
        
        ax.set_xlim([-55, 50])
        ax.set_xticks([-50,0,50])
    
    axArr[-1].tick_params(axis='x', which='both', top=False, bottom=True)
    axArr[-1].spines["bottom"].set_visible(True)
    
    
    fig.text(-0.02, 0.53, r'transmission, $T$ (\%)', va='center', rotation='vertical', fontsize = labelfontsize)
    
    
    fig.subplots_adjust(hspace=0.05)
    
    if saveBool:
        fig.savefig(
            "./figures/" + "SI_transmission1048Q2.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
def plotSI_1061Transmission(saveBool = False):
    # === File and plotting settings ===
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    filenames = [
        '4.2um_0_1061nm_Thermo_IRPowerSweep.mat',
    ]
    doubletBool = [True, False]
    # labels = ['1061']
    lambdaArr = np.array([1061])
    etaFibre = [0.3]
    xmin = -50
    xmax = 50
    
    # === Colormaps ===
    # numFiles = len(filenames)
    # markers = ['.'] * numFiles
    cpalette = 'hot'
    
    cmap_full = cm.get_cmap(cpalette)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated", cmap_full(np.linspace(0, 0.7, 256))
    )
    
    norm = mcolors.Normalize(vmin=lambdaArr.min(), vmax=lambdaArr.max())
    
    # === Set up plots ===
    fig, axArr = plot_custom(
       OpticaWidthFull*0.3, 0.3*OpticaWidthFull, 
       r"detuning, $\Delta\omega/2\pi$ (GHz)", 
       [r'',r''],#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 5, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    
    
    powerMin = 1e3
    powerMax = 0
    # === Loop through each file ===
    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
            else:
                powerW[j] = params[6] * power_factor * 1e-3
                if np.abs(powerW[j]) < powerMin:
                    powerMin = np.abs(powerW[j])
                if np.abs(powerW[j]) > powerMax:
                    powerMax = np.abs(powerW[j])
                        
                    
    for i, fname in enumerate(filenames):
        data = loadmat(f'{folderPath}/{fname}', simplify_cells=True)
        deviceSweepData = data['deviceSweepData']
    
        numFits = len(deviceSweepData)
        powerW = np.zeros(numFits, dtype=complex)
        cT = np.zeros(numFits, dtype=complex)
        Qips = np.zeros(numFits, dtype=complex)
        Qexs = np.zeros(numFits, dtype=complex)
        lambdas = np.zeros(numFits, dtype=complex)
        dQips = np.zeros(numFits, dtype=complex)
        dQexs = np.zeros(numFits, dtype=complex)
        dlambdas = np.zeros(numFits, dtype=complex)
        tau = np.zeros(numFits, dtype=complex)
        phi = np.zeros(numFits, dtype=complex)
        Ncavs = np.zeros(numFits, dtype=complex)
    
        if doubletBool[i]:
            Qipa = np.zeros(numFits, dtype=complex)
            Qexa = np.zeros(numFits, dtype=complex)
            lambdaa = np.zeros(numFits, dtype=complex)
            dQipa = np.zeros(numFits, dtype=complex)
            dQexa = np.zeros(numFits, dtype=complex)
            dlambdaa = np.zeros(numFits, dtype=complex)
            Ncava = np.zeros(numFits, dtype=complex)
            
        # === Load parameters ===
        for j, entry in enumerate(deviceSweepData):
            params = np.array(entry['params'])
            resNorm = np.array(entry['resNorm'])
            power_factor = 1.86 if i < 1 else 1.0
    
            if doubletBool[i]:
                powerW[j] = params[6,0] * power_factor * 1e-3
                cT[j] = params[5,0]
                tau[j] = params[3,0]
                phi[j] = params[4,0]
                Qips[j], Qexs[j], lambdas[j] = params[2,0], params[1,0], params[0,0]*1e-9
                Qipa[j], Qexa[j], lambdaa[j] = params[2,1], params[1,1], params[0,1]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2,0], resNorm[1,0], resNorm[0,0]*1e-9
                dQipa[j], dQexa[j], dlambdaa[j] = resNorm[2,1], resNorm[1,1], resNorm[0,1]*1e-9
                
                
            else:
                powerW[j] = params[6] * power_factor* 1e-3
                cT[j] = params[5]
                tau[j] = params[3]
                phi[j] = params[4]
                Qips[j], Qexs[j], lambdas[j] = params[2], params[1], params[0]*1e-9
                dQips[j], dQexs[j], dlambdas[j] = resNorm[2], resNorm[1], resNorm[0]*1e-9
                
                
    
        # === Sort all by power ===
        sort_idx = np.argsort(powerW)
        powerW = powerW[sort_idx]
        Qips, Qexs, lambdas, dQips, dQexs, dlambdas = [arr[sort_idx] for arr in (Qips, Qexs, lambdas, dQips, dQexs, dlambdas)]
    
        if doubletBool[i]:
            Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa = [arr[sort_idx] for arr in (Qipa, Qexa, lambdaa, dQipa, dQexa, dlambdaa)]
    
        # === Derived quantities ===
        ws = 2 * np.pi * c_const / lambdas
        # dws = 2 * np.pi * c_const * dlambdas / lambdas**2
        
        # plt.plot(lambdas)
        
    
        if doubletBool[i]:
            wa = 2 * np.pi * c_const / lambdaa
            # dwa = 2 * np.pi * c_const * dlambdaa / lambdaa**2
            g = ws - wa
    
            kips = ws / Qips
            kipa = wa / Qipa
            kexs = ws / Qexs
            kexa = wa / Qexa
    
            # dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            # dkipa = np.sqrt((dwa/Qipa)**2 + (dQipa*wa/Qipa**2)**2)
    
            ks = kips + kexs
            ka = kipa + kexa
            
            norm = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            # cpaletteT = 'viridis'
            # cmapT = cm.get_cmap(cpaletteT)  # or 'plasma', 'inferno', etc.
            colorArrayT = [cmap(norm(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9   
                TOut = np.array(entry['T'])[0:-1:skip]
                filt = np.logical_and((w-np.mean([ws[j], wa[j]]))*GHzprad > xmin,(w-np.mean([ws[j], wa[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                
                wMeas = (w)/1e9 
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j],  Delta - g[j]/2, power = powerW[j])
                
                Ncavs[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_s)
                
                
                [a_a, N] = calcField(wa[j], kexa[j], ka[j], 
                              cT[j],  Delta + g[j]/2, power = powerW[j])
                
                Ncava[j] = calcNcav(etaFibre[i], powerW[j], np.mean([ws, wa]), Delta, a_a)
                
                T = calcTransmission(kexs[j], a_s, kexa[j], a_a, tau = tau[j]*np.exp(1j*phi[j]))
                
                skip = 5
                
                if np.mod(j, 2) == 0:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    axArr.plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw/5)
                    axArr.scatter((wMeas[1:-1:skip]-np.mean([ws[j], wa[j]])/1e9)/2/np.pi,TOut[1:-1:skip]/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2, zorder = 10)
                
                
        
            # axArr[i].errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
            #             marker=markers[i], linestyle = 'none', label=labels[i], 
            #             color=colorArray_r[i], ms = ms, markeredgewidth=0.5)
            # axArr[i].errorbar(Ncava, kipa*GHzprad, yerr=dkipa*GHzprad,
            #             marker=markers[i], linestyle = 'none', color=colorArray_r[i], ms = msa, lw = lw, markeredgewidth=0.5)

            
        else:
    
            kips = ws / Qips
            kexs = ws / Qexs
            
            # dkips = np.sqrt((dws/Qips)**2 + (dQips*ws/Qips**2)**2)
            
            ks = kips + kexs
            
            normT = mcolors.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
            colorArrayT = [cmap(normT(power)) for power in np.log10(np.abs(powerW))]
            for j, entry in enumerate(deviceSweepData):
                skip = int(np.floor(len(np.array(entry['W']))/200))
                w = 2*np.pi*c_const/np.array(entry['W'])[0:-1:skip]*1e9  
                TOut = np.array(entry['T'])[0:-1:skip] 
                filt = np.logical_and((w-np.mean([ws[j], ws[j]]))*GHzprad > xmin,(w-np.mean([ws[j], ws[j]]))*GHzprad < xmax)
                w = w[filt]
                TOut = TOut[filt]
                wMeas = (w)/1e9 
                
                Delta = np.linspace(xmin/GHzprad,xmax/GHzprad, 200)
                [a_s, N] = calcField(ws[j], kexs[j], ks[j], 
                              cT[j]*0.00000001,  Delta, power = np.sqrt(etaFibre[i])*powerW[j])
                
                Ncavs[j] = np.max(np.sqrt(etaFibre[i])*(powerW[j])/(hbar*((ws[j]) + Delta))*np.abs(a_s)**2)
                
              
                
                T = calcTransmission(kexs[j], a_s, kexs[j], a_s, tau = tau[j]*np.exp(1j*phi[j]))
                
                 
                skip = 1
                
                if np.mod(j, 5) == 0:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,2) == 0:
                    # if j < 60:#j == 0 or j == len(deviceSweepData)-1 or j == int(np.floor(len(deviceSweepData)/2)): #np.mod(j,5) == 0:
                    axArr[i].plot(Delta*GHzprad, T/np.max(T), color = colorArrayT[j], lw = lw/2 )
                    axArr[i].scatter((wMeas[0:-1:skip]-ws[j]/1e9)/2/np.pi,TOut[0:-1:skip]/np.max(T),color = colorArrayT[j], edgecolors='none', marker = '.', s = ms/2)
               # colorArrayT[j]
               
              
    
            # axArr[i].errorbar(Ncavs, kips*GHzprad, yerr=dkips*GHzprad,
            #             marker=markers[i], linestyle = 'none', label=labels[i], color=colorArray_r[i], ms = ms, lw = lw, markeredgewidth=0.5)
           
          
            
    
    # === Add colorbar for wavelength ===
    
    # norm = plt.Normalize(vmin=np.log10(powerMin), vmax=np.log10(powerMax))
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # cbar = fig.colorbar(sm, ax=axArr, orientation='horizontal', pad=0.15)
    # cbar.set_label(r"input power, $P$ (dBm)", labelpad=3, fontsize = labelfontsize)
    # # Move it above the top axes manually
    # pos = axArr[-1].get_position()
    # cbar.ax.set_position([pos.x0+0.07, pos.y1 - 0.8, pos.width, 0.06])  # [x, y, width, height]
    # # Move colorbar above (by adjusting position)
    # cbar.ax.xaxis.set_ticks_position('bottom')
    # cbar.ax.xaxis.set_label_position('bottom')
    # cbar.set_ticks(np.log10([powerMin, powerMax]))
    # cbar.set_ticklabels([rf'${l:.1f}$' for l in 10*np.log10(np.array([powerMin, powerMax])*1e3)], fontsize = numfontsize)
    # # cbar.set_ticklabels([r'$10^{-7.0}$', r'$10^{-1.2}$'], fontsize = numfontsize)
    # plt.tight_layout()
    
    # for ax in axArr:
    #     # ax.set_ylim([None,1])
    #     ax.set_xlim([-50, 50])
    #     y_low, y_high = ax.get_ylim()
        
    #     ax.set_yticks([y_low, 1])
    #     ax.spines['left'].set_bounds(y_low,1)
    #     ax.set_ylim([y_low-(1-y_low)/6,1+(1-y_low)/20])
    #     ax.set_yticklabels([rf"${val:.2f}$" for val in [y_low, 1]])
        
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     ax.yaxis.set_ticks_position('left')
    #     ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    #     ax.tick_params(axis='x', which='both', top=False, bottom=False)
    #     ax.spines["bottom"].set_visible(False)
    #     ax.spines['bottom'].set_bounds(-50,50)
        
    #     ax.set_xlim([-60, 50])
    
    # axArr[-1].tick_params(axis='x', which='both', top=False, bottom=True)
    # axArr[-1].spines["bottom"].set_visible(True)
    
    
    # fig.text(-0.02, 0.53, r'transmission, $T$ (\%)', va='center', rotation='vertical', fontsize = labelfontsize)
    
    
    # fig.subplots_adjust(hspace=0.05)
    
    # if saveBool:
    #     fig.savefig(
    #         "./figures/" + "SI_transmission1048.svg", dpi=300, bbox_inches="tight",transparent=True
    #     )
    
    plt.show()
    
    
def plotWideSweepData(saveBool = False):
    colorTM = 'tomato'
    colorTE = 'darkblue'
    
    fig, ax = plot_custom(
       OpticaWidthFull*0.61, 0.32*OpticaWidthFull, 
       r"laser frequency, $\omega/2\pi$ (THz)", 
       r'transmission, $T$ (au)',#[r'$\kappa_{\text{c}}/2\pi$ (GHz)']*numFiles,
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/data_for_Nick/'
    
    fileList = [
        'full_sweep_data/4.2_0_1260_1360_120uW_5nmps_GreenOff_noFilter_gain2_res10kOhm_1.mat',
        'full_sweep_data/4.2_0_1360_1500_1200uW_5nmps_GreenOff_noFilter_gain5_1.mat',
        'full_sweep_data/4.2_0_1480_1640_greenOff_1mw_1.mat'
        ];

    backgroundList = [
        'backgrounds/4.2_0_1260_1360_210uW_5nmps_background_1.mat',
        'backgrounds/4.2_0_1355_1505_3100uW_5nmps_background.mat',
        'backgrounds/4.2_0_1480_1640_2400uW_10nmps_background_1.mat'
        ]
    
    folderPathSmallRange = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData'
    
    filenamesSmallRange = [
        '4.2um_0_979nm_Thermo_IRPowerSweep.mat',
        '4.2um_0_1048nm_Thermo_IRPowerSweep.mat'
    ]

    
    wArrTM = np.array([186.8,196.7,206.6,216.5,226.5,236.5,246.3,256.3,266.3,276.2,286.2,296.2,306.2])+0.1
    wArrTE = np.array([190.2,200.57,210.91,221.1,231.3,241.5,251.8,262,272.2,282.3,292.5,302.7,312.9])+0.25
    xMaxArr = [1355, 1490, 2000]
    xMinArr = [0, 1355, 1490]
    
    f_lowpasses = [100, 20, 20, 20]
    
    NoiseFloor = 0
    NoiseFloorBack = NoiseFloor
    minLast = 1 / 1.24
    
    N = len(backgroundList)
    
    
    for i in range(N):
    
        # --- Load data ---
        power_mW, W, T = extract_averaged_transmission_data_power(
            fileList[i], folderPath
        )
        power_mW_back, WBack, TBack = extract_averaged_transmission_data_power(
            backgroundList[i], folderPath
        )
    
        # --- Normalize and filter ---
        TOut = norm_data(T, W, WBack, TBack, NoiseFloor, NoiseFloorBack)
        TOut = filter_data(W, TOut, 0, 0, f_lowpasses[i])
    
        # Apply window (xMin/xMax)
        mask = (W <= xMaxArr[i]) & (W >= xMinArr[i])
        W = W[mask]
        TOut = TOut[mask]
    
        # Normalize
        TOut = TOut / np.max(TOut)
        TOut = TOut * (minLast / TOut[0])
        minLast = TOut[-1]
    
        # Convert to angular freq (assuming W = wavelength (nm)?)
        f = c_const / W * 1e9 / 1e12   # exactly matching MATLAB expression
    
        # --- Plot ---
        ax.plot(f, TOut, "lightgray", lw = lw)
        
        fBW = 0.1 #THz
        for fFiltAround in wArrTM:
            filt = np.logical_and(f<fFiltAround + fBW,f>fFiltAround - fBW )
            fFilt = f[filt]
            TOutFilt = TOut[filt]
            if len(fFilt) > 0:
                ax.plot(fFilt, TOutFilt, colorTM, lw = lw)
                # ax.plot([fFiltAround,fFiltAround], [-1, np.min(TOutFilt)-0.1], colorTM, lw=lw)
                
        for fFiltAround in wArrTE:
            filt = np.logical_and(f<fFiltAround + fBW,f>fFiltAround - fBW )
            fFilt = f[filt]
            TOutFilt = TOut[filt]
            if len(fFilt) > 0:
                ax.plot(fFilt, TOutFilt, colorTE, lw = lw)
                # ax.plot([fFiltAround,fFiltAround], [-1, np.min(TOutFilt)-0.1], colorTE, lw=lw)

    skip = 1
    # for i, fname in enumerate(filenamesSmallRange):
    #     data = loadmat(f'{folderPathSmallRange}/{fname}', simplify_cells=True)
    #     deviceSweepData = data['deviceSweepData']
    #     f = c_const/np.array(deviceSweepData[0]['W'])[0:-1:skip]*1e9*1e-12
    #     # w = 2*np.pi*f
    #     TOut = np.array(deviceSweepData[0]['T'])[0:-1:skip]
        
        
    #     ax.plot(f, TOut, colorTM, lw = lw)
    
    
    # for fFiltAround in wArrTE:
    #     ax.plot([fFiltAround,fFiltAround], [1.03,2], colorTE, lw=lw)
    # for fFiltAround in wArrTM:
    #     ax.plot([fFiltAround,fFiltAround], [1.03, 2], colorTM, lw=lw)
    
    plt.tight_layout()
    
    # lambmin = 1640
    # lambmax = 950
    xmin = 183
    xmax = 238
    lambmin = freq_to_wavelength(xmin)
    lambmax = freq_to_wavelength(xmax)
    xticks = ax.get_xticks()
    ax.set_xticks([183, 200,220,238,260,280,300,307])
    ax.spines['bottom'].set_bounds(xmin, xmax)
    ax.set_xlim([xmin-2,xmax])
    
    ax.set_ylim([-0.05,1.05])
    ax.spines['left'].set_bounds(0,1)
    # ax.set_yticks([y_low, y_high])
    # ax.set_yticklabels([rf"${val:.1f}$" for val in [y_low, y_high]])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
            
    # Create the secondary x-axis
    axsec = ax.secondary_xaxis('top', functions=(freq_to_wavelength, wavelength_to_freq))
    axsec.set_xlabel(r"TM mode wavelength, $\lambda$ (nm)",fontsize = labelfontsize)
    
    
    ax = axsec
    # xmin = np.min(NtoI1047( NArr))
    # xmax = np.max(NtoI1047( NArr))
    # xrng = xmax-xmin
    ax.spines['top'].set_bounds(lambmin,lambmax)
    # ax.xaxis.set_ticks_position('top')
    xticks = np.array([1267,1322,1385,1450,1524,1604])
    ax.set_xticks(xticks)
    ax.set_yticklabels([rf"${val:.1f}$" for val in xticks])
    # ax.set_xlim([xmin-xrng/10,xmax+xrng/10])

    ax.tick_params(
        axis="x",
        which="major",
        width=0.8,
        length=2,
        labelsize=numfontsize,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )
    
    
    # fig.text(-0.05, 0.6, r'internal loss, $\kappa_{\text{c}}/2\pi$ (GHz)', va='center', rotation='vertical', fontsize = labelfontsize)
    
    
    # fig.subplots_adjust(hspace=0.2)
    
    if saveBool:
        fig.savefig(
            "./figures/" + "wavelengthSweepWide.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

def plot1042Transmission(saveBool = False):
    fig, ax = plot_custom(
       OpticaWidthFull*0.25, 0.2*OpticaWidthFull, 
       r"$\lambda$ (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fitAndDataArrays.mat'
    
    
    data = loadmat(filePath, squeeze_me=True)
    # dataParams = loadmat(filePathParams, squeeze_me=True)
    
    fit = data['fit']
    scalar = np.max(fit)
    fit /= scalar
    W = data['W']
    TOut = data['TOut']/ scalar
    Wfit = data['Wfit']
    
    # params = dataParams['params']
    filt = np.logical_and(Wfit< 1042.4, Wfit>1042.25)
    skip = 50
    Wfit= Wfit[filt]
    fit = fit[filt]
    ax.plot(Wfit, fit, lw = lw, color = 'gray')
    ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = ms/2, zorder = 10, color = 'k')
    
    plt.tight_layout()
    
    lambmin = np.min(Wfit)
    lambmax = np.max(Wfit)
    ax.set_xlim([lambmin-0.01,lambmax])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    ax.set_ylim([0.35,1.01])
    ax.spines['left'].set_bounds(0.4,1)
    ax.set_xticks([lambmin, lambmax])
    ax.set_yticks([0.4,0.7,1])
    ax.set_xticklabels([rf"${val:.1f}$" for val in [lambmin, lambmax]])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
        
    if saveBool:
        fig.savefig(
            "./figures/" + "1042Transmission.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def plot4p4umTransmission(saveBool = False):
    fig, ax = plot_custom(
       OpticaWidthFull*0.25, 0.2*OpticaWidthFull, 
       r"$\lambda$ (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fitAndDataArrays.mat'
    
    
    data = loadmat(filePath, squeeze_me=True)
    # dataParams = loadmat(filePathParams, squeeze_me=True)
    
    fit = data['fit']
    scalar = np.max(fit)
    fit /= scalar
    W = data['W']
    TOut = data['TOut']/ scalar
    Wfit = data['Wfit']
    
    # params = dataParams['params']
    skip = 50
    
    ax.plot(Wfit, fit, lw = lw, color = 'gray')
    ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = ms/2, zorder = 10, color = 'k')
    
    plt.tight_layout()
    
    lambmin = np.min(Wfit)
    lambmax = np.max(Wfit)
    ax.set_xlim([lambmin-0.015,lambmax])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    ax.set_ylim([0.15,1.01])
    ax.spines['left'].set_bounds(0.2,1)
    ax.set_xticks([lambmin, lambmax])
    ax.set_yticks([0.2,0.6,1])
    ax.set_xticklabels([rf"${val:.1f}$" for val in [lambmin, lambmax]])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
        
    if saveBool:
        fig.savefig(
            "./figures/" + "1042Transmission.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
        
    
def extractEta(saveBool = False):

    fig, axArr = plot_custom(
       OpticaWidthFull*0.5, 0.25*OpticaWidthFull, 
       [r"",r''], 
       r'$\eta_{\text{fiber}}$ (au)',
       labelPadX = 3, labelPadY = 8, commonY = True, commonX = False, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 2, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize-0.8,
       widthBool =True, widthRatios = [40, 380], heightRatios = [1]
    )
    
    
    lambdaArr = np.array( [979, 1047, 1267, 1322, 1383, 1450, 1524, 1604])
    depth   =    np.array([0,   0.4,  0.45,  0.45,  0.41,  0.48,  0.45,  0.35])
    lw = 1
    
    folderPath1040 = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsIRChris/20251106/'
    folderPath = 'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsIRChris/20251107/'
    folderTaper = 'TaperCalibration_Taper/'
    folderNoTaper = 'TaperCalibration/'
    
    file1040minTaper = '1042_Pmin/TaperCalibration_Standard_1042nm_0.18227mW_outTaperCalibration_1030_1070Newport_1042_Pmin_1.mat'
    file1040maxTaper = '1042_Pmax/TaperCalibration_Standard_1042nm_0.19753mW_outTaperCalibration_1030_1070Newport_1042_Pmax_1.mat'
    file1040NoTaper = '1042/TaperCalibration_Standard_1042nm_0.54361mW_outTaperCalibration_1040_1070Newport_1042_1.mat'
    
    file1300minTaper = '1267_Pmin/TaperCalibration_Standard_1267nm_0.094648mW_outTaperCalibration_1260_1360Santec_1267_Pmin_1.mat'
    file1300maxTaper = '1267_Pmax/TaperCalibration_Standard_1267nm_0.096764mW_outTaperCalibration_1260_1360Santec_1267_Pmax_2.mat'
    file1300NoTaper = '1267/TaperCalibration_Standard_1267nm_0.22539mW_outTaperCalibration_1260_1360Santec_1267_Fiber_3.mat'

    file1400minTaper = '1383_Pmin/TaperCalibration_Standard_1383nm_0.10794mW_outTaperCalibration_1358_1502Santec_1383_Pmin_2.mat'
    file1400maxTaper = '1383_Pmax/TaperCalibration_Standard_1383nm_0.10923mW_outTaperCalibration_1358_1502Santec_1383_Pmax_2.mat'
    file1400NoTaper = '1383/TaperCalibration_Standard_1383nm_0.30603mW_outTaperCalibration_1358_1502Santec_1383_Taper_2.mat'

    file1500minTaper = '1523_Pmin/TaperCalibration_Standard_1523nm_0.1896mW_outTaperCalibration_1480_1640Santec_1523_Pmin_3.mat'
    file1500maxTaper = '1523_Pmax/TaperCalibration_Standard_1523nm_0.19548mW_outTaperCalibration_1480_1640Santec_1523_Pmax_3.mat'
    file1500NoTaper = '1523/TaperCalibration_Standard_1523nm_0.45837mW_outTaperCalibration_1480_1640Santec_1523_Taper_2.mat'

    def calcAndPlotEta(folderPath1040, folderTaper, folderNoTaper, file1040minTaper, file1040maxTaper, file1040NoTaper, ax,skip = 5, window = 5000, color = 'k'):
        # filepath = os.path.join(folderPath1040, folderTaper, file1040minTaper)
        # data1040minTaper = loadmat(filepath, squeeze_me=True, struct_as_record=False)['scanData']
        filepath = os.path.join(folderPath1040, folderTaper, file1040maxTaper)
        data1040maxTaper = loadmat(filepath, squeeze_me=True, struct_as_record=False)['scanData']
        filepath = os.path.join(folderPath1040, folderNoTaper, file1040NoTaper)
        data1040NoTaper = loadmat(filepath, squeeze_me=True, struct_as_record=False)['scanData']
        
        interpolator = interp1d(data1040NoTaper.W, data1040NoTaper.T, kind='linear', fill_value='extrapolate')
        # ax.scatter(data1040maxTaper.W[1::skip], data1040maxTaper.T[1::skip]/interpolator(data1040maxTaper.W[1::skip]), edgecolors='none', marker = '.', s = ms, zorder = 10, color = 'r')
        # ax.scatter(data1040minTaper.W[1::skip], data1040minTaper.T[1::skip]/interpolator(data1040minTaper.W[1::skip]), edgecolors='none', marker = '.', s = ms, zorder = 10, color = 'darkblue')
        
        def running_mean_centered(x, w):
            kernel = np.ones(w) / w
            return np.convolve(x, kernel, mode='same')
        def running_range(x, window):
            x = np.asarray(x)
            if window > len(x):
                raise ValueError("window must be smaller than array length")
        
            # shape (N - window + 1, window)
            shape = (x.size - window + 1, window)
            strides = (x.strides[0], x.strides[0])
            X = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        
            return X.max(axis=1) - X.min(axis=1)
        def running_range_centered(x, window):
            x = np.asarray(x)
            pad = window // 2
            xpad = np.pad(x, pad, mode='edge')
            return running_range(xpad, window)[0:-1]
        window = int(np.floor(window/skip))
        dataMean = running_mean_centered(data1040maxTaper.T[1::skip]/interpolator(data1040maxTaper.W[1::skip]), window)
        dataRange = running_range_centered(data1040maxTaper.T[1::skip]/interpolator(data1040maxTaper.W[1::skip]), window)
        
        for lamb in lambdaArr:
            arr = np.abs(data1040maxTaper.W[1::skip] - lamb)
            
            if np.min(np.abs(arr)) < 1:
                idx = np.argmin(arr)
                print(['Lamdba: ', lamb, '  Lamdba: ', data1040maxTaper.W[1::skip][idx], '  mean: ', dataMean[idx], '  unc: ', dataRange[idx] ])
        
        valid = slice(window, -window)
        ax.plot(data1040maxTaper.W[1::skip][valid],dataMean[valid], color = color, lw = lw/2)
        
        ax.fill_between(
                data1040maxTaper.W[1::skip][valid],
                dataMean[valid] - dataRange[valid]/2,
                dataMean[valid] + dataRange[valid]/2,
                alpha=0.2,
                color = color,
                lw = 0
            )

    calcAndPlotEta(folderPath1040, folderTaper, folderNoTaper, file1040minTaper, file1040maxTaper, file1040NoTaper, axArr[0], skip =8, window = 100, color = 'coral')
    calcAndPlotEta(folderPath, folderTaper, folderNoTaper, file1300minTaper, file1300maxTaper, file1300NoTaper, axArr[1], skip = 100, color = 'red')        
    calcAndPlotEta(folderPath, folderTaper, folderNoTaper, file1400minTaper, file1400maxTaper, file1400NoTaper, axArr[1], skip = 100, window = 2000, color = 'firebrick')        
    calcAndPlotEta(folderPath, folderTaper, folderNoTaper, file1500minTaper, file1500maxTaper, file1500NoTaper, axArr[1], skip = 100, color = 'k')        

    for i, lamb in enumerate(lambdaArr):
        for ax in axArr:
            ax.plot([lamb, lamb], [0.6, depth[i]], lw = 0.8, color ='k')

    plt.tight_layout()
    
    lambmin = 1030
    lambmax = 1070
    axArr[0].set_xlim([lambmin-8,lambmax])
    axArr[0].spines['bottom'].set_bounds(lambmin, lambmax)
    
    axArr[0].set_ylim([0.08,0.52])
    axArr[0].spines['left'].set_bounds(0.1,0.5)
    
    axArr[0].set_xticks([lambmin,lambmax])
    axArr[0].set_yticks([0.1,0.2,0.3,0.4, 0.5])
    axArr[0].set_yticklabels([rf"${val:.1f}$" for val in axArr[0].get_yticks()])
    axArr[0].set_xticklabels([rf"${val:.0f}$" for val in axArr[0].get_xticks()])
    
    axArr[0].spines["right"].set_visible(False)
    axArr[0].spines["top"].set_visible(False)
    axArr[0].yaxis.set_ticks_position('left')
    axArr[0].tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    axArr[0].tick_params(axis='x', which='both', top=False, bottom=True)
    axArr[0].spines["bottom"].set_visible(True)
    # Create the secondary x-axis
    axsec1 = axArr[0].secondary_xaxis('top')
    axsec2 = axArr[1].secondary_xaxis('top')
    axsec1.set_xlabel(r"",fontsize = labelfontsize)
    
    
    ax = axsec1
    ax.spines['top'].set_bounds(lambmin,lambmax)
    ax.tick_params(
        axis="x",
        which="major",
        width=0.5,
        length=2,
        labelsize=numfontsize-1,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )
    
    ax.set_xticks(lambdaArr[0:3])
    ax.set_xticklabels([rf"${val:.0f}$" for val in ax.get_xticks()])
    ax.spines['top'].set_bounds(lambdaArr[1], None)
    ax.spines['top'].set_color('none')

    
    lambmin = 1260
    lambmax = 1640
    axArr[1].set_xlim([lambmin-40,lambmax])
    axArr[1].spines['bottom'].set_bounds(lambmin-50, lambmax)
    
    
    axArr[1].set_xticks([lambmin,1360, 1480, lambmax])
    axArr[1].set_xticklabels([rf"${val:.0f}$" for val in axArr[1].get_xticks()])
    
    axArr[1].spines["right"].set_visible(False)
    axArr[1].spines["left"].set_visible(False)
    axArr[1].spines["top"].set_visible(False)
    axArr[1].yaxis.set_ticks_position('left')
    axArr[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    axArr[1].tick_params(axis='x', which='both', top=False, bottom=True)
    axArr[1].spines["bottom"].set_visible(True)
    

     
    ax = axsec2
    ax.spines['top'].set_bounds(lambmin-50,lambmax)
    ax.tick_params(
        axis="x",
        which="major",
        width=0.8,
        length=2,
        labelsize=numfontsize-1,
        zorder=1,
        direction="in",
        top="off",
        colors='k'
    )
    
    
    ax.set_xticks(lambdaArr[2::])
    ax.set_xticklabels([rf"${val:.0f}$" for val in ax.get_xticks()])
    ax.spines['top'].set_bounds(None, lambdaArr[-1])
    ax.spines['top'].set_color('none')
    
    
    fig.text(0.38, 0.05, r'laser wavelength, $\lambda$ (nm)', va='center', fontsize = labelfontsize)
    # fig.text(0.35, 1.08, r'TM mode wavelengths, $\lambda$ (nm)', va='center', fontsize = labelfontsize)
    
    
    fig.subplots_adjust(wspace=0.005)    
    
    if saveBool:
        fig.savefig(
            "./figures/" + "fiberEfficiency.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()
    
def extract_averaged_transmission_data_power(filename, folder_path):
    """
    Extract averaged transmission data from a .mat file and return:
    - power_mW
    - W (1D array)
    - T (1D array)
    """
    
    # -------- Parse power from filename --------
    m = re.search(r"([0-9.]+)mW", filename)
    if m:
        power_mW = float(m.group(1))
    else:
        print(f"Warning: Could not parse power from filename '{filename}'. Using 0 mW.")
        power_mW = 0.0
    
    # -------- Load .mat file --------
    file_path = f"{folder_path}{filename}"
    try:
        data = loadmat(file_path, squeeze_me=True)
    except:
        raise FileNotFoundError(f"Could not load .mat file: {file_path}")

    # -------- Resolve scanData / scanNoiseFloor / fallback --------
    scan_data = None

    # Prefer scanData if present
    if "scanData" in data:
        scan_data = data["scanData"]

    # If missing, try scanNoiseFloor
    elif "scanNoiseFloor" in data:
        scan_data = data["scanNoiseFloor"]

    # Otherwise fallback to raw W and T
    elif "W" in data and "T" in data:
        scan_data = {"W": data["W"], "T": data["T"]}

    else:
        raise ValueError(f"File {file_path} missing expected W/T data")

    # -------- Extract W and T safely --------
    try:
        W = np.array(scan_data["W"]).flatten()
        T = np.array(scan_data["T"]).flatten()
    except:
        raise ValueError(f"Could not extract W/T arrays from file {file_path}")

    # -------- Remove duplicates --------
    W_unique, unique_idx = np.unique(W, return_index=True)
    T_unique = T[unique_idx]

    # -------- Sort --------
    sort_idx = np.argsort(W_unique)
    W_sorted = W_unique[sort_idx]
    T_sorted = T_unique[sort_idx]

    # -------- Apply buffer trim --------
    buffer = 300
    if len(W_sorted) > 2 * buffer:
        W_final = W_sorted[buffer:-buffer]
        T_final = T_sorted[buffer:-buffer]
    else:
        print("Warning: Not enough points for buffer trimming. Returning full dataset.")
        W_final = W_sorted
        T_final = T_sorted

    return power_mW, W_final, T_final

    
def norm_data(T, W, WNorm, TNorm, NormOffset, NormOffsetBack):
    """
    Normalize T(W) by an interpolated normalization curve TNorm(WNorm).

    Parameters
    ----------
    T : array-like
        Input data to be normalized.
    W : array-like
        Wavelength/frequency axis for T.
    WNorm : array-like
        Axis for the normalization data.
    TNorm : array-like
        Normalization data corresponding to WNorm.
    NormOffset : float
        Offset applied to the numerator.
    NormOffsetBack : float
        Offset applied to the denominator.

    Returns
    -------
    TOut : ndarray
        Normalized output array.
    """

    # Convert inputs to numpy arrays
    T = np.asarray(T)
    W = np.asarray(W)
    WNorm = np.asarray(WNorm)
    TNorm = np.asarray(TNorm)

    # ---- Step 1: Ensure WNorm is strictly unique for interpolation ----
    unique_vals, unique_idx = np.unique(WNorm, return_index=True)
    WNorm = unique_vals
    TNorm = TNorm[unique_idx]

    # ---- Step 2: Interpolate TNorm onto W (linear, allow extrapolation) ----
    interpolator = interp1d(WNorm, TNorm, kind='linear', fill_value='extrapolate')
    TNormInterp = interpolator(W)

    # ---- Step 3: Normalize ----
    denom = TNormInterp - NormOffsetBack
    if np.any(denom == 0):
        raise ValueError("Normalization denominator contains zeros after offset.")

    TOut = (T - NormOffset) / denom
    return TOut


def filter_data(W, TOut, f0Arr, QArr, f_lowpass):
    """
    Python equivalent of MATLAB filterData(W, TOut, f0Arr, QArr, f_lowpass)
    """

    # Compute FFT and sampling frequency
    f, TOutFFT, fs = cust_fft(W, TOut)

    # Start with input signal
    TOutFilt = np.array(TOut, dtype=float)
    
    # LOW-PASS FILTER (Butterworth 4th order)
    # MATLAB: designfilt('lowpassiir', FilterOrder=4)
    # -------------------------------
    # Normalize frequency for SciPy
    Wn = f_lowpass / (fs / 2)

    # 4th-order Butterworth
    b, a = butter(4, Wn, btype='low')

    # Zero-phase filtering
    TOutFilt = filtfilt(b, a, TOutFilt)

    return TOutFilt



def cust_fft(x, y):
    """
    Python equivalent of MATLAB custFFT(x, y)
    Returns:
        X  - frequency axis (half spectrum)
        Y  - FFT of y (single-sided, amplitudes doubled except DC/Nyquist)
        Xs - sampling frequency
    """
    x = np.asarray(x)
    y = np.asarray(y)

    N = len(x)

    # MATLAB: xSamp = (x(end) - x(1)) / N
    x_samp = (x[-1] - x[0]) / N

    # Sampling frequency
    Xs = 1 / x_samp

    # Frequency vector: Xs/N * (0:N-1)
    X_full = (Xs / N) * np.arange(N)

    # Single-sided spectrum indices
    half_N = N // 2 + 1  # same as floor(N/2+1)

    X = X_full[:half_N]

    # FFT
    Y_full = np.fft.fft(y)
    Y = Y_full[:half_N].copy()

    # Double amplitudes except DC and Nyquist (MATLAB: Y(2:end-1) *= 2)
    if half_N > 2:
        Y[1:-1] *= 2

    return X, Y, Xs


def print_params(params, param_names, res_norm):
    """
    Python equivalent of MATLAB printParams.

    params:     np.ndarray (n_params, n_fits)
    param_names: list of strings OR 2D array of characters
    res_norm:   np.ndarray (n_params, n_fits)
    """

    lower_limit = 1e-3
    upper_limit = 1e5

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Convert MATLAB-style char array to Python list of strings, if needed
    if isinstance(param_names, np.ndarray) and param_names.dtype.kind in ['U', 'S']:
        # param_names is an array like shape (n, m) containing characters
        param_names = ["".join(row).strip() for row in param_names]

    # Loop over parameters
    n_params, n_cols = params.shape
    for i in range(n_params):
        name = param_names[i] if isinstance(param_names, (list, tuple)) else str(param_names[i])
        print(f"{name:<25}", end="")

        for j in range(n_cols):
            val = params[i, j]
            unc = res_norm[i, j]

            if abs(val) < lower_limit or abs(val) >= upper_limit:
                # scientific notation
                print(f" {val:12.2e} +/- {unc:12.2e}", end="")
            else:
                # fixed-point
                print(f" {val:12.2f} +/- {unc:12.2f}", end="")

        print("")  # newline

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    
#%% Run code
saveBool = False

# plotSI_AllTransmissionAndKappa(saveBool)
# plotSI_AllTransmissionAndKappaNew(saveBool)
plotSI_1048KappasNew(saveBool)
plotAllFitsKappaIn(saveBool)
# plotSI_1048Transmission(saveBool)
# plotSI_1048TransmissionQ2(saveBool)
# plotWideSweepData(saveBool)
# plot1042Transmission(saveBool)
# extractEta(saveBool = saveBool)


# plotSI_1061Transmission(saveBool)
# plotSI_1061Transmission(saveBool)
# plotSI_1061Kappas(saveBool)