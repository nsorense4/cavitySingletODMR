# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:10:56 2025

@author: nicho
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
from numba import njit
from commonlabcode.plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible


labelfontsize = 8
numfontsize = 8
textwidth = 180/25.4#8.5 - 2 # in inches
PRQuantumWidthFull = textwidth
opticaFullWidth = textwidth

def calcTransmission(kex1, k1, kex2, k2, g, Delta, tau = 1):
    return np.abs(tau - 1/2*(kex1)/(-1j*(Delta+g/2) + k1/2) - 1/2*(kex2)/(-1j*(Delta-g/2) + k2/2))**2


min_x_value = -4
max_x_value = 4
min_y_value = 0
max_y_value = 1
limScaleMin = 0.2
Delta = np.linspace(min_x_value, max_x_value, 1000)

fig, axArr = plot_custom(
   opticaFullWidth*0.8, 0.15*PRQuantumWidthFull, 
   [r"$\Delta\omega/\kappa$ (au)", r"$\Delta\omega/ \kappa$ (au)", r"$\Delta\omega/\kappa$ (au)"], 
   r"$T$ (au)" #"coincidence rate (mHz/mW)"
   , labelPadX = 5, labelPadY = 6, commonY = True, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   ncol = 3, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)


kex1 = 0.5
kex2 = 0.5
kin1 = kex1
kin2 = kex2
k1 = kin1 + kex1
k2 = kin2 + kex2

axArr[0].plot(Delta, calcTransmission(kex1, k1, kex2, k2, 0, Delta), color = 'k', lw= 1)
axArr[1].plot(Delta[1:500], calcTransmission(kex1, k1, kex2, k2, 1, Delta[1:500]), color = 'r', lw= 1)
axArr[1].plot(Delta[501:-1], calcTransmission(kex1, k1, kex2, k2, 1, Delta[501:-1]), color = 'darkblue', lw= 1)
axArr[2].plot(Delta[1:500], calcTransmission(kex1, k1, kex2, k2, 4, Delta[1:500]), color = 'r', lw= 1)
axArr[2].plot(Delta[501:-1], calcTransmission(kex1, k1, kex2, k2, 4, Delta[501:-1]), color = 'darkblue', lw= 1)

for i, ax in enumerate(axArr):
    if i > 0:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        # ax.set_yticks([])
    else:
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
        ax.yaxis.set_ticks_position('left')
        # ax.yaxis.set_ticks([0, 1-1/np.sqrt(2), 1])
        # ax.yaxis.set_ticks_labels([])
        
        
    
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    
    ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks([-4, -2, 0, 2, 4])
    
    ax.spines['left'].set_bounds(min_y_value,max_y_value)
    
    ax.set_ylim([min_y_value - limScaleMin, max_y_value])
    ax.set_xlim([min_x_value - 0.5, max_x_value])

# plt.savefig(
#     "__figures/" + "doubletSplitting" + ".svg", dpi=300, bbox_inches="tight",transparent=True
# )

plt.show()