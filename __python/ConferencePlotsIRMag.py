# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 22:10:04 2026

@author: nicho
"""

import pandas as pd
import matplotlib.pyplot as plt
# import qutip
import numpy as np
import seaborn as sns
from datetime import datetime
from commonlabcode.plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible
from scipy.optimize import curve_fit
import scipy.io
from scipy.io import loadmat
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter
from scipy.signal import butter, filtfilt, lfilter
from scipy.ndimage import uniform_filter1d
# from allantools import oadev 
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors


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
nd = 2.4
GHzprad = 1e-9/2/np.pi
lambdah = 1355

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
    a1 = np.sqrt(kex1)/(ki1 + kex1 - 1j*(Delta - g/2))
    a2 = np.sqrt(kex2)/(ki2 + kex2 - 1j*(Delta + g/2))
    return np.abs(tau - np.sqrt(1/4*kex1)*a1 - np.sqrt(1/4*kex2)*a2)**2


def plotNarrowTransmission(saveBool = False):
    fig, ax = plot_custom(
       OpticaWidthFull*0.2, 0.15*OpticaWidthFull, 
       r"$\lambda$ (nm)", 
       r'$T$ (au)',
       labelPadX = 3, labelPadY = 3, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize-1, numsize=numfontsize-1,
       widthBool =False
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
    filt = np.logical_and(W< lambmin, W>lambmax)
    skip = 1
    W= W[filt]
    TOut= TOut[filt]
    ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = 3, zorder = 10, color = 'r')
    
    plt.tight_layout()
    (Tmin, Tmax) = ax.get_ylim()
    
    lamb = np.linspace(lambmin, lambmax, 1000)*1e-9
    nu = c_const/lamb
    lamb0 = 1040.77e-9
    nu0 = c_const/lamb0
    
    g = 0
    kex1 = 10.5e9
    kex2 = kex1
    ki1 = 15e9
    ki2 = ki1
    Delta = nu - nu0 
    print(nu0/ki1)
    fit= calcTransmission(kex1, kex2, ki1, ki2, Delta, g = 0, tau = 1*np.exp(-1j*0.1))
    
    
    
    ax.plot(lamb*1e9, fit, lw = 0.5, color = 'red')
    Tmin = 0.3
    ax.set_xlim([lambmin-0.2,lambmax])
    ax.spines['bottom'].set_bounds(lambmin, lambmax)
    
    ax.set_ylim([Tmin-0.1,1])
    ax.spines['left'].set_bounds(Tmin,1)
    ax.set_xticks([lambmin, lambmax])
    ax.set_yticks([Tmin, 1])
    ax.set_xticklabels([rf"${val:.1f}$" for val in [lambmin, lambmax]])
    ax.set_yticklabels([rf"${val:.2f}$" for val in ax.get_yticks()])
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    ax.spines["bottom"].set_visible(True)
        
    if saveBool:
        fig.savefig(
            "./figures/" + "NarrowTransmission_4p2um.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

def plotWideTransmission(saveBool = False):
    fig, axArr = plot_custom(
       OpticaWidthFull*0.45, 0.4*OpticaWidthFull, 
       r"laser wavelength, $\lambda$ (nm)", 
       [r'$T$ (au)',r'$\Delta T$ ($\times 10^{-4}$)'],
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 2, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =True, widthRatios = [1], heightRatios = [1,2] 
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
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
    axArr[0].plot(W[1::skip], TOut[1::skip], lw = 0.5, color = 'r')
    # ax.scatter(W[1::skip], TOut[1::skip], edgecolors='none', marker = '.', s = ms/2, zorder = 10, color = 'k')
    
    dataAbs = dataAbs['data']
    WAbs = dataAbs['laserWLsRead'].item()*1e9
    signalAbs = dataAbs['signal'].item()
    contrast = dataAbs['contrast'].item()
    
    filt = np.logical_and(WAbs< lambmax, WAbs>lambmin)
    skip = 1
    WAbs= WAbs[filt]
    signalAbs= signalAbs[filt]
    
    axArr[1].plot(WAbs[1::skip], signalAbs[1::skip]/np.max(contrast)*1e4, lw = 0.5, color = 'r')
    
    
    plt.tight_layout()
    
    for i,ax in enumerate(axArr):
        ax.set_xlim([lambmin-1.2,lambmax])
        ax.spines['bottom'].set_bounds(lambmin, lambmax)
        
        if i == 0:
            ax.set_ylim([0.25,1.01])
            ax.spines['left'].set_bounds(0.3,1)
            ax.set_xticks([lambmin, 1042, lambmax])
            ax.set_yticks([0.3,1])
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis='x', which='both', top=False, bottom=False)
        else:
            ax.set_ylim([-2.4, 3])
            ax.spines['left'].set_bounds(-2,3)
            ax.set_yticks([-2, -1, 0, 1, 2, 3])
            ax.tick_params(axis='x', which='both', top=False, bottom=True)
            
        ax.set_xticklabels([rf"${val:.0f}$" for val in [lambmin,1042, lambmax]])
        
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        
    if saveBool:
        fig.savefig(
            "./figures/" + "WideTransmissionAndAbsSignal_4p2um.svg", dpi=300, bbox_inches="tight",transparent=True
        )

    plt.show()

def plot2DData(saveBool = False):
    fig, ax = plot_custom(
       OpticaWidthFull*0.3, 0.25*OpticaWidthFull, 
       r"wavelength, $\lambda$ (nm)", 
       r'MW detuning, $\Delta f$ (MHz)',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    fig1, ax1 = plot_custom(
       OpticaWidthFull*0.2, 0.25*OpticaWidthFull, 
       r"$\Delta T$ ($\times 10^{-4}$)", 
       r'',
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    fig2, ax2 = plot_custom(
       OpticaWidthFull*0.3, 0.2*OpticaWidthFull, 
       r'',
       r"", 
       labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
       spineColor = 'black', tickColor = 'black', textColor = 'black',
       nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
       widthBool =False
    )
    lw = 1
    
    # filePathParams = 'C:/Users/nicho/OneDrive - University of Calgary/BarclayLab File Depot/Project Files/DNV Near IR Analysis/fittedData/4.4um_0_1044nm_fit_highQ.mat'
    filePath =  'C:/Users/nicho/OneDrive - University of Calgary/Documents/gitProjects/dataBarclay/measurementsODMR/CW/20260128/ODMR_20260128_CW__16_14dBAtt_4.2um_0_1_2.7765GHz_10MHz_003.mat'
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
    
    ax1.plot(signalX, f, 'r-', lw = 0.5)
    ax2.plot(WAbs, signalY, 'r-', lw = 0.5)
    
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
            "./figures/" + "2DSignal_4p2um.png", dpi=300, bbox_inches="tight",transparent=True
        )
        fig1.savefig(
            "./figures/" + "yLine.svg", dpi=300, bbox_inches="tight",transparent=True
        )
        fig2.savefig(
            "./figures/" + "xLine.svg", dpi=300, bbox_inches="tight",transparent=True
        )
    
    plt.show()
    
#%% call functions
saveBool = False

plotNarrowTransmission(saveBool)
# plotWideTransmission(saveBool)
plot2DData(saveBool)