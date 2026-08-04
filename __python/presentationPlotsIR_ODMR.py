# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 18:04:29 2025

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


min_x_value = -1
max_x_value = 1
min_y_value = 0
max_y_value = 1
limScaleMin = 0.1
Delta = np.linspace(min_x_value, max_x_value, 1000)

fig, ax = plot_custom(
   opticaFullWidth*0.4, 0.25*PRQuantumWidthFull, 
   r"$\Delta\omega$", 
   r"$T$ (au)" #"coincidence rate (mHz/mW)"
   , labelPadX = 5, labelPadY = 6, commonY = True, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)


kex1 = 0.1
kex2 = 0.1
kin1 = kex1/4
kin2 = kex2
k1 = kin1 + kex1
k2 = kin2 + kex2
T1 = calcTransmission(kex1, k1, kex1, k1, 0, Delta)
T2 = calcTransmission(kex2, k2, kex2, k2, 0, Delta)
ax.plot(Delta, T1, color = 'r', lw= 1)
ax.plot(Delta, T2, color = 'b', lw= 1)
ax.plot(Delta, T1-T2, color = 'k', lw= 1)

# ax.spines["left"].set_visible(False)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
      
    
    
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis='x', which='both', top=False, bottom=True)

ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
ax.xaxis.set_label_position("bottom")
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_ticks([-1,0, 1])

ax.spines['left'].set_bounds(0,1)

ax.set_ylim([min_y_value - limScaleMin, max_y_value])
ax.set_xlim([min_x_value - 0.1, max_x_value])

plt.savefig(
    "C:/Users/nicho/OneDrive - University of Calgary/Documents/LabBarclay/presentations/figuresPresentations/update20251113/" + "IRAbsorptionContrast" + ".svg", dpi=300, bbox_inches="tight",transparent=True
)

min_x_value = -1
max_x_value = 1
min_y_value = 0
max_y_value = 1
limScaleMin = 0.1
Delta = np.linspace(min_x_value, max_x_value, 1000)

fig, ax = plot_custom(
   opticaFullWidth*0.4, 0.25*PRQuantumWidthFull, 
   r"$\Delta\omega$", 
   r"$T$ (au)" #"coincidence rate (mHz/mW)"
   , labelPadX = 5, labelPadY = 6, commonY = True, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)


kex1 = 0.1
kex2 = 0.1
kin1 = kex1*0.999
kin2 = kex2
k1 = kin1 + kex1
k2 = kin2 + kex2
T1 = calcTransmission(kex1, k1, kex1, k1, 0, Delta)
T2 = calcTransmission(kex2, k2, kex2, k2, 0, Delta)
ax.plot(Delta, T1, color = 'r', lw= 1)
ax.plot(Delta, T2, color = 'b', lw= 1)
ax.plot(Delta, T1-T2, color = 'k', lw= 1)

# ax.spines["left"].set_visible(False)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
      
    
    
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis='x', which='both', top=False, bottom=True)

ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
ax.xaxis.set_label_position("bottom")
ax.xaxis.set_ticks_position('bottom')
ax.xaxis.set_ticks([-1,0, 1])

ax.spines['left'].set_bounds(0,1)

ax.set_ylim([min_y_value - limScaleMin, max_y_value])
ax.set_xlim([min_x_value - 0.1, max_x_value])

plt.savefig(
    "C:/Users/nicho/OneDrive - University of Calgary/Documents/LabBarclay/presentations/figuresPresentations/update20251113/" + "IRAbsorptionContrastSmall" + ".svg", dpi=300, bbox_inches="tight",transparent=True
)


fig, ax = plot_custom(
   opticaFullWidth*0.4, 0.25*PRQuantumWidthFull, 
   r"$\kappa_{\text{i+p}} /\kappa_{\text{abs}}  = \kappa_{\text{ex}}/\kappa_{\text{abs}}$", 
   r"contrast, $C$ (au)" #"coincidence rate (mHz/mW)"
   , labelPadX = 5, labelPadY = 6, commonY = True, commonX = False, wSpaceSetting=0, hSpaceSetting = 0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)
min_x_value = 0
max_x_value = 10
min_y_value = 0
max_y_value =1
limScaleMin = 0.1

k = np.linspace(0, max_x_value, 1000)
kabs = 1
k1 = kin1 + kex1
k2 = kin2 + kex2
T1 = calcTransmission(k, 2*k+kabs, k, 2*k+kabs, 0, 0)
T2 = calcTransmission(k, 2*k, k, 2*k, 0, 0)
C = -T2 +    T1
ax.plot(k, C, color = 'r', lw= 1)

# ax.spines["left"].set_visible(False)
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
      
    
    
ax.set_yscale('log')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis='x', which='both', top=False, bottom=True)

ax.spines['bottom'].set_bounds(min_x_value,max_x_value)
ax.xaxis.set_label_position("bottom")
ax.xaxis.set_ticks_position('bottom')
# ax.xaxis.set_ticks([-1,0, 1])

# ax.spines['left'].set_bounds(0,1)

# ax.set_ylim([min_y_value - limScaleMin, max_y_value])
# ax.set_xlim([min_x_value - 0.1, max_x_value])

# plt.savefig(
#     "C:/Users/nicho/OneDrive - University of Calgary/Documents/LabBarclay/presentations/figuresPresentations/update20251113/" + "contrastwithk" + ".svg", dpi=300, bbox_inches="tight",transparent=True
# )

plt.show()