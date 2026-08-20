# -*- coding: utf-8 -*-
"""
Shared physical constants, NV rate parameters, fitted operating points, and
plot styling constants used across IRMagnetometryPaperPlots.py,
NVpopulationDynamics.py, and linbladMESolverFunctionChargeConversionLiouvillian.py.

These were previously copy-pasted independently into each file and had
already drifted (detuning/zeeman, nd) between IRMagnetometryPaperPlots.py and
NVpopulationDynamics.py without anyone intending it. Resolved values below
were confirmed against NVpopulationDynamics.py's copies.
"""
import numpy as np

# === Fundamental constants ===
h = 6.62607015e-34      # Planck constant [J*s]
hbar = h/(2*np.pi)
c_const = 299792458.0   # speed of light [m/s]

# === NV intrinsic transition rates (power-dependent rates are magic numbers) ===
gamma_eg = 66e6
gamma_isc_0 = 7.9e6
gamma_isc_1 = 53e6
gamma_isc_g0 = 1e6
gamma_isc_g1 = 0.7e6
gamma_s = 1e9
gamma_NV0_eg = 53e6      # k^0_f
gamma_NV0_em = 22540.55  # k_75
gamma_NV0_mg = 357687.5  # k_56

# === Intrinsic absorption cross sections ===
sigma_p = 3e-21  # m^-2
sigma_s = 2e-22  # m^-2

# === Diamond/NV material constants ===
gamma = 1.761e11/2/np.pi  # electron gyromagnetic ratio, Hz/T
T2s = 0.43e-6
gamma_phi = 1/T2s
nd = 2.3904                     # refractive index
hyperfine = 2.16e6
carbon_density = 1.76e29        # atoms/m^3
rho_NV = carbon_density*4.5e-6  # atoms/m^3

# === Laser/optical setup ===
lamb_s = 1042e-9   # IR probe
lamb_p = 532e-9    # green pump
nus = c_const/lamb_s
omegas = nus*2*np.pi
detuning = 0.2e9*2*np.pi
zeeman = 0.2e9*2*np.pi
etaFibreIR = 0.25
etaPostFibre = 0.3

# === Fitted optimal operating points ===
# Earlier fit iterations kept below for reference/history.
# I_pOpt     = 3.135e+08
# I_sOpt     = 1.162e+10
# OmegaROpt  = 3.824e+06
# I_pOpt     = 1.785e+08
# I_sOpt     = 1.005e+10
# OmegaROptRadps  = 9.973e+05
I_pOpt = 3.043e+08
I_sOpt = 1.014e+10
OmegaROpt = 1.984e+06
OmegaROptRadps = 1.984e+06

I_pOpt_vis = 3.126e+08
OmegaROptRadps_vis = 2.317e+06

# Fitted NV0/NV- rate parameters. Earlier fit iterations kept below for
# reference/history.
# paramsRatesReal = np.array([
#     1e7, #kem W
#     1.7e7, # Ke0
#     66150.4, #K25 g [2]
#     7.11e5, #K51 g [4]
#     25690.73, #K71 [15]
#     25474.34, #K73 [17]
#     1012.032, #K74 [6]
#     4853.41, #K46 [8]
#     0.371923, # k25 double [3]
#     439, # k25 single [1]
#     0.006162, # k51 [16]
#     0.1,        # k73 [18]
#     1.71e3,   # k74 [5]
#     0.1,        # k46 [9]
#     208.5184, # k71 [14]
#     43546.84, # k36 [10]
#     20000.02, # k36 [11]
#     943, # k43 [7]
#     22540.55, #k_75
#     357687.5,   #k_56
#     0,
#     0
#     ])
paramsRatesReal = np.array([2.97702015e+07, 1.80180000e+07, 5.65801443e+04, 3.79558334e+06,
 3.69347363e+03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 3.68409690e-01, 1.90125394e+02, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 3.02741247e+02, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 2.12417732e+03, 1.35680211e+05,
 6.85335419e+02, 3.96197208e+01])
# paramsRatesReal = np.array([
#     1e7, #kem W
#     1.8e7, # Ke0
#     17877.72, #K25 g [2]
#     381868.3, #K51 g [4]
#     9430.601, #K71 g [15]
#     0, #K73 [17]
#     0, #K74 [6]
#     0, #K46 [8]
#     0.106659, # k25 double [3]
#     271.542, # k25 single [1]
#     0, # k51 [16]
#     0,        # k73 [18]
#     0,   # k74 single [5]
#     0,        # k46 [9]
#     460.2011, # k71 [14]
#     0, # k36 [10]
#     0, # k36 [11]
#     0, # k43 [7]
#     1783.011, #k_75
#     130000.6,   #k_56
#     213.4342, #k26 g
#     3.962, #k26 single
#     ])

paramsRatesSimple = np.array([2.97702015e+07, 1.80180000e+07, 5.65801443e+04 ,3.79558334e+06,
 3.69347363e+03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0, 0, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 2.12417732e+03, 1.35680211e+05,
 6.85335419e+02 ,0])

paramsRatesOnes = np.ones(22)

# === Plot styling ===
ms = 2
msa = 3
lw = 0.2/0.3528
GHzprad = 1e-9/2/np.pi

labelfontsize = 8
numfontsize = 8
textwidth = 174/25.4  # 8.5 - 2 # in inches
PRQuantumWidthFull = textwidth
PRQuantumWidthHalf = 85/25.4

# === Color palette ===
colorBlue = '#190cffff'
colorDarkBlue = '#0900a2ff'
colorLightBlue = '#3e9effff'

colorRed = '#fe0000ff'
colorDarkRed = '#800000ff'
colorLightRed = '#ff8484ff'

colorGreen = '#07b700ff'
colorDarkGreen = '#058000ff'
colorLightGreen = '#3dfe00ff'

nuOpt = 1840020
A = 0.1e-3**2
