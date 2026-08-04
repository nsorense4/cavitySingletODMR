# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:00:27 2025

@author: nicho
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from commonlabcode.plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible

from functools import lru_cache
import cProfile
from numba import njit

DIM = 11
BASIS = [qt.basis(DIM, i) for i in range(DIM)]

# State indices
g0, gp, gm = 0, 1, 2
e0, ep, em = 3, 4, 5
s1, s2 = 6, 7
gNV0, mNV0, eNV0 = 8, 9, 10

# MW couplings
H_MW_gp  = BASIS[g0] * BASIS[gp].dag() + BASIS[gp] * BASIS[g0].dag()
H_MW_gm  = BASIS[g0] * BASIS[gm].dag() + BASIS[gm] * BASIS[g0].dag()

# Zeeman / detuning projectors
H_gp = BASIS[gp] * BASIS[gp].dag()
H_gm = BASIS[gm] * BASIS[gm].dag()
H_g0 = BASIS[g0] * BASIS[g0].dag()

C = {}

# Optical pumping g -> e
C["g0e0"] = BASIS[e0] * BASIS[g0].dag()
C["gpep"] = BASIS[ep] * BASIS[gp].dag()
C["gmem"] = BASIS[em] * BASIS[gm].dag()

# e -> mNV0
C["e0m"] = BASIS[mNV0] * BASIS[e0].dag()
C["epm"] = BASIS[mNV0] * BASIS[ep].dag()
C["emm"] = BASIS[mNV0] * BASIS[em].dag()

# e -> gNV0
C["e0g"] = BASIS[gNV0] * BASIS[e0].dag()
C["epg"] = BASIS[gNV0] * BASIS[ep].dag()
C["emg"] = BASIS[gNV0] * BASIS[em].dag()

# NV0 transitions
C["gNV0eNV0"] = BASIS[eNV0] * BASIS[gNV0].dag()
C["mNV0g0"] = BASIS[g0] * BASIS[mNV0].dag()
C["mNV0gp"] = BASIS[gp] * BASIS[mNV0].dag()
C["mNV0gm"] = BASIS[gm] * BASIS[mNV0].dag()
C["eNV0s2"] = BASIS[s2] * BASIS[eNV0].dag()
C["eNV0s1"] = BASIS[s1] * BASIS[eNV0].dag()
C["eNV0g0"] = BASIS[g0] * BASIS[eNV0].dag()
C["eNV0gp"] = BASIS[gp] * BASIS[eNV0].dag()
C["eNV0gm"] = BASIS[gm] * BASIS[eNV0].dag()

C["s2gNV0"] = BASIS[gNV0] * BASIS[s2].dag()

# Singlet
C["s1s2"] = BASIS[s2] * BASIS[s1].dag()
C["s1gNV0"] = BASIS[gNV0] * BASIS[s1].dag()
C["s2s1"] = BASIS[s1] * BASIS[s2].dag()

# Radiative decay
C["e0g0"] = BASIS[g0] * BASIS[e0].dag()
C["epgp"] = BASIS[gp] * BASIS[ep].dag()
C["emgm"] = BASIS[gm] * BASIS[em].dag()

# ISC
C["e0s1"] = BASIS[s1] * BASIS[e0].dag()
C["eps1"] = BASIS[s1] * BASIS[ep].dag()
C["ems1"] = BASIS[s1] * BASIS[em].dag()

# Singlet to ground
C["s2g0"] = BASIS[g0] * BASIS[s2].dag()
C["s2gp"] = BASIS[gp] * BASIS[s2].dag()
C["s2gm"] = BASIS[gm] * BASIS[s2].dag()

# NV0 radiative decay 
C["eNV0gNV0"] = BASIS[gNV0] * BASIS[eNV0].dag()
C["mNV0gNV0"] = BASIS[gNV0] * BASIS[mNV0].dag()
C["eNV0mNV0"] = BASIS[mNV0] * BASIS[eNV0].dag()

# =========================
# ISC e -> s1
# =========================
C["e0_s1"] = BASIS[s1] * BASIS[e0].dag()
C["ep_s1"] = BASIS[s1] * BASIS[ep].dag()
C["em_s1"] = BASIS[s1] * BASIS[em].dag()


# =========================
# s2 -> ground triplet
# =========================
C["s2_g0"] = BASIS[g0] * BASIS[s2].dag()
C["s2_gp"] = BASIS[gp] * BASIS[s2].dag()
C["s2_gm"] = BASIS[gm] * BASIS[s2].dag()

# Pure dephasing
for i in [g0, gp, gm, e0, ep, em]:
    C[f"deph_{i}"] = BASIS[i] * BASIS[i].dag()

@lru_cache(maxsize=1)
def liouvillian_templates():
    """
    Unit-rate Liouvillian blocks.
    Cached so they are built only once.
    """
    L = {}

    # -------------------------
    # Hamiltonian components
    # -------------------------
    L["H_MW_gp"] = qt.liouvillian(H_MW_gp, [])
    L["H_MW_gm"] = qt.liouvillian(H_MW_gm, [])
    L["H_gp"]    = qt.liouvillian(H_gp, [])
    L["H_gm"]    = qt.liouvillian(H_gm, [])
    L["H_g0"]    = qt.liouvillian(H_g0, [])

    # -------------------------
    # Collapse processes
    # -------------------------
    for key, cop in C.items():
        L[key] = qt.liouvillian(None, [cop])

    return L



def nv_steady_state(
    Omega, #rad/s
    I_s,   #rad/s
    I_p, #W/m^2
    gamma_phi,   #Hz
    gamma_eg,
    gamma_isc_0,
    gamma_isc_1,
    gamma_isc_g0,
    gamma_isc_g1,
    gamma_s,
    gamma_NV0_eg,  #k^0_f
    gamma_NV0_em, #k_75
    gamma_NV0_mg,   #k_56
    sigma_p, # m^%-2
    sigma_s, # m^%-2
    lamb_s,   # IR probe
    lamb_p,    # green pump
    c, #m/s
    h,      # Planck constant [J·s]
    paramsRates,
    paramsRatesReal,
    detuning=0, #rad/s
    zeeman = 0, #rad/s
    steadystatebool = True,
    rhoPrev = None,
    tEnd = 2e-6,
    numT = 500,
    verboseResults = False,
    bool1042 = False
):
    """
    Compute steady-state populations of an 8-level NV center model.

    Returns:
        pop_ss: array of populations [g0,g+,g-,e0,e+,e-,s1,s2]
    """
    # --------------------------------------------------
    # Fundamental optical rates
    # --------------------------------------------------
    Gamma_pump = sigma_p * I_p * lamb_p / (h * c)
    Gamma_IR_singlet = sigma_s * I_s * lamb_s / (h * c)
    scalar = 1
    Int2mWp = 1 * np.pi*300e-9**2 * 1e3
    Int2Ns = 1 / (5e2 * 1e4)
    Pp = I_p * Int2mWp
    Ns = I_s * Int2Ns

    paramsRatesFull = paramsRates * paramsRatesReal
    
    # sigurd's model
    Gamma_pump           = Pp * paramsRatesFull[0] # Kem
    Gamma_pump_NV0ge     = Pp * paramsRatesFull[1]     # Ke0
    
    Gamma_pump_NVe_NV0m  = Pp * paramsRatesFull[2]    # K25
    Gamma_pump_NVe_NV0g  = Pp * paramsRatesFull[20]  # K26
    Gamma_pump_NV0m_NVg  = Pp * paramsRatesFull[3]      # K51
    Gamma_pump_NV0e_NVg  = Pp * paramsRatesFull[4]    # K71
    Gamma_pump_NV0e_NVs1 = Pp * paramsRatesFull[5]    # K73
    Gamma_pump_NV0e_NVs2 = Pp * paramsRatesFull[6]    # K74
    Gamma_pump_NVs2_NV0g = Pp * paramsRatesFull[7]   # K46
    
    # --------------------------------------------------
    # IR-assisted processes
    # --------------------------------------------------
    # sigurd's model
    Gamma_IR_NVe_NV0m = Ns**2 * paramsRatesFull[8]  # K25
    Gamma_IR_NVe_NV0g = Ns * paramsRatesFull[21]  # K26
    Gamma_IR_NVe_NV0m_single = Ns * paramsRatesFull[9] # K25
    Gamma_IR_NV0m_NVg = Ns**2 * paramsRatesFull[10] # K51
    Gamma_IR_NV0e_NVs1 = Ns**2 * paramsRatesFull[11] # K73
    Gamma_IR_NV0e_NVs2 = Ns * paramsRatesFull[12] # K74
    Gamma_IR_NVs2_NV0g = Ns**2 * paramsRatesFull[13] # K46
    Gamma_IR_NV0e_NVg  = Ns *paramsRatesFull[14] # K71
    
    Gamma_pump_s1_NV0g       = Pp    * paramsRatesFull[15] # k36
    Gamma_IR_s1_NV0g          = Ns    * paramsRatesFull[16] # k36
    # if not(bool1042):
    # Gamma_IR_singlet          = Ns    * paramsRatesFull[17] # k36 
       
    gamma_NV0_em = paramsRatesFull[18] #k_75
    gamma_NV0_mg = paramsRatesFull[19]   #k_56
    
    
    def fmt(x):
        return f"{x: .1e}"   # 2 significant digits, scientific notation, aligned

    if verboseResults:
        print("\nk109:", fmt(gamma_NV0_em))
        print("k98 :", fmt(gamma_NV0_mg))
    
        print("\nPump")
        scale_p = (I_p * lamb_p / (h * c))
        print("s03, s14, s25 :", fmt(Gamma_pump / scale_p))
        print("s810          :", fmt(Gamma_pump_NV0ge / scale_p))
        print("s100–102      :", fmt(Gamma_pump_NV0e_NVg / 3 / scale_p))
        print("s90–92        :", fmt(Gamma_pump_NV0m_NVg / scale_p))
        print("s38–58        :", fmt(Gamma_pump_NVe_NV0g / scale_p))
        print("s39–59        :", fmt(Gamma_pump_NVe_NV0m / scale_p))
        print("s68           :", fmt(Gamma_pump_s1_NV0g / scale_p))
        print("s107          :", fmt(Gamma_pump_NV0e_NVs2 / scale_p))
    
        print("\nIR")
        scale_s = (I_s * lamb_s / (h * c))
        print("s100–102      :", fmt(Gamma_IR_NV0e_NVg / 3 / scale_s))
        print("s90–92 (TPA)  :", fmt(Gamma_IR_NV0m_NVg / 3 / scale_s**2))
        print("s38–58        :", fmt(Gamma_IR_NVe_NV0g / scale_s))
        print("s39–59        :", fmt(Gamma_IR_NVe_NV0m_single / scale_s))
        print("s39–59 (TPA)  :", fmt(Gamma_IR_NVe_NV0m / scale_s**2))
        print("s68           :", fmt(Gamma_IR_s1_NV0g / scale_s))
        print("s107          :", fmt(Gamma_IR_NV0e_NVs2 / scale_s))
        print("s106 (TPA)    :", fmt(Gamma_IR_NV0e_NVs1 / scale_s**2))
        print("s76           :", fmt(Gamma_IR_singlet / scale_s))
        print("s78           :", fmt(Gamma_IR_NVs2_NV0g / scale_s**2))

    # --------------------------------------------------
    # Combined effective rates (exactly as in QuTiP)
    # --------------------------------------------------
    Gamma_e_to_m = (
        Gamma_pump_NVe_NV0m
        + Gamma_IR_NVe_NV0m
        + Gamma_IR_NVe_NV0m_single
    )
    Gamma_e_to_g = (
        Gamma_pump_NVe_NV0g
        + Gamma_IR_NVe_NV0g
    )

    Gamma_s1_NV0g    = Gamma_pump_s1_NV0g + Gamma_IR_s1_NV0g
    Gamma_eNV0_to_g  = Gamma_pump_NV0e_NVg  + Gamma_IR_NV0e_NVg
    Gamma_eNV0_to_s1 = Gamma_pump_NV0e_NVs1 + Gamma_IR_NV0e_NVs1
    Gamma_eNV0_to_s2 = Gamma_pump_NV0e_NVs2 + Gamma_IR_NV0e_NVs2
    Gamma_s2_to_gNV0 = Gamma_pump_NVs2_NV0g + Gamma_IR_NVs2_NV0g
    

    gammaIR_0 = 0
    gammaIR_1 = 0
    gammaIR_2 = 0
    gammaIR_3 = Gamma_IR_NVe_NV0m_single + Gamma_IR_NVe_NV0m
    gammaIR_4 = Gamma_IR_NVe_NV0m_single + Gamma_IR_NVe_NV0m
    gammaIR_5 = Gamma_IR_NVe_NV0m_single + Gamma_IR_NVe_NV0m
    gammaIR_6 = Gamma_IR_s1_NV0g
    gammaIR_7 = Gamma_IR_NVs2_NV0g + Gamma_IR_singlet
    gammaIR_8 = 0
    gammaIR_9 = Gamma_IR_NV0m_NVg
    gammaIR_10 = Gamma_IR_NV0e_NVs2 + Gamma_IR_NV0e_NVs1 + Gamma_IR_NV0e_NVg
    
    gammaIRtot = (gammaIR_0 + gammaIR_1 + gammaIR_2 + gammaIR_3 + gammaIR_4 
                  + gammaIR_5 + gammaIR_6 + gammaIR_7 + gammaIR_8 + gammaIR_9 
                  + gammaIR_10)
    
    
    dim = DIM
    basis = BASIS
    
    LT = liouvillian_templates()
    L = 0

    # Hamiltonian    
    L += (Omega / 2) * (LT["H_MW_gp"] + LT["H_MW_gm"])
    L += (zeeman + detuning) * LT["H_gp"]
    L += (zeeman - detuning) * LT["H_gm"]


    # Optical pumping g -> e
    L += Gamma_pump * LT["g0e0"]
    L += Gamma_pump * LT["gpep"]
    L += Gamma_pump * LT["gmem"]
    
    # Optical pumping eNV -> mNV0
    L += Gamma_e_to_m * LT["e0m"]
    L += Gamma_e_to_m * LT["epm"]
    L += Gamma_e_to_m * LT["emm"]
    
    # Optical pumping eNV -> gNV0
    L += Gamma_e_to_g * LT["e0g"]
    L += Gamma_e_to_g * LT["epg"]
    L += Gamma_e_to_g * LT["emg"]
    
    # optical pumping eNV0 to g
    L += Gamma_eNV0_to_g/3 * LT["eNV0g0"]
    L += Gamma_eNV0_to_g/3 * LT["eNV0gp"]
    L += Gamma_eNV0_to_g/3 * LT["eNV0gm"]
    
    # optical pumping eNV0 to s1
    L += Gamma_eNV0_to_s1 * LT["eNV0s1"]
    
    # Optical pumping eNV0 -> s2
    L += Gamma_eNV0_to_s2 * LT["eNV0s2"]
    
    # optical pumping s2 to gNV0
    L += Gamma_s2_to_gNV0 * LT["s2gNV0"]
    
    # Optical pumping g -> e NV0
    L += Gamma_pump_NV0ge * LT["gNV0eNV0"]
    
    # Optical pumping mNV0 -> g
    L += (Gamma_pump_NV0m_NVg + Gamma_IR_NV0m_NVg)/3 * LT["mNV0g0"]
    L += (Gamma_pump_NV0m_NVg + Gamma_IR_NV0m_NVg)/3 * LT["mNV0gp"]
    L += (Gamma_pump_NV0m_NVg + Gamma_IR_NV0m_NVg)/3 * LT["mNV0gm"]
    
    
    # IR pumping of singlet
    L += Gamma_IR_singlet * LT["s2s1"]
    
    # Singlet decay s1 -> s2
    L += gamma_s * LT["s1s2"]
    L += Gamma_s1_NV0g * LT["s1gNV0"]

    # Radiative decay e -> g
    L += gamma_eg * LT["e0g0"]
    L += gamma_eg * LT["epgp"]
    L += gamma_eg * LT["emgm"]
    
    # radiative decay eNV0 -> gNV0
    L += gamma_NV0_eg * LT["eNV0gNV0"]
    
    # Nonradiative decay mNV0 -> gNV0
    L += gamma_NV0_mg * LT["mNV0gNV0"]
    
    # Nonradiative decay eNV0 -> mNV0
    L += gamma_NV0_em * LT["eNV0mNV0"]
    
    # ISC e -> s
    L += gamma_isc_0 * LT["e0_s1"]
    L += gamma_isc_1 * LT["ep_s1"]
    L += gamma_isc_1 * LT["em_s1"]
    
    
    # s2 -> ground triplet
    L += gamma_isc_g0 * LT["s2_g0"]
    L += gamma_isc_g1 * LT["s2_gp"]
    L += gamma_isc_g1 * LT["s2_gm"]
    
    # Pure dephasing
    for idx in [gp, gm, g0, ep, em, e0]:
        L += gamma_phi/2 * LT[f"deph_{idx}"]

    if steadystatebool:
        
        # rho1 = qt.steadystate(H, c_ops)
        rho_ss = qt.steadystate(
                                L,
                                method="direct",   # fastest for DIM ~ 11
                                use_rcm=True,
                                sparse = True,
                                )
    
        # Extract populations
        pop_ss = [np.real((basis[i].dag() * rho_ss * basis[i])) for i in range(dim)]
        
        
        
        return pop_ss
    else:
        # ---------------------------
        # Initial state
        # ---------------------------
        rho0 = BASIS[g0] * BASIS[g0].dag()  # start in ground ms=0
    
        # ---------------------------
        # Observables (populations)
        # ---------------------------
        obs_list = [BASIS[i] * BASIS[i].dag() for i in range(DIM)]
    
        # ---------------------------
        # Solver options
        # ---------------------------
        opts = qt.Options(
            nsteps=10000,
            rtol=1e-6,
            atol=1e-8,
            store_states=False
        )
    
        t_list = np.linspace(0, tEnd, numT)
    
        # ---------------------------
        # Time evolution with Liouvillian
        # ---------------------------
        result = qt.mesolve(
            L,
            rho0,
            t_list,
            c_ops=[],          # MUST be empty
            e_ops=obs_list,
            options=opts
        )
    
        return result, t_list

# def collapse_to_rate_indices(c_op):
#     """Return (i, j) if c_op = |i><j|, else None."""
#     mat = c_op.full()
#     inds = np.argwhere(np.abs(mat) > 0)
#     if inds.shape[0] != 1:
#         return None
#     i, j = inds[0]
#     return int(i), int(j)

# def nv_steady_state_IRRate(
#     Omega, #rad/s
#     I_s,   #rad/s
#     I_p, #W/m^2
#     gamma_phi,   #Hz
#     gamma_eg,
#     gamma_isc_0,
#     gamma_isc_1,
#     gamma_isc_g0,
#     gamma_isc_g1,
#     gamma_s,
#     gamma_NV0_eg,  #k^0_f
#     gamma_NV0_em, #k_75
#     gamma_NV0_mg,   #k_56
#     sigma_p, # m^%-2
#     sigma_s, # m^%-2
#     lamb_s,   # IR probe
#     lamb_p,    # green pump
#     c, #m/s
#     h,      # Planck constant [J·s]
#     paramsRates,
#     paramsRatesReal,
#     detuning=0, #rad/s
#     zeeman = 0, #rad/s
#     steadystatebool = True,
#     rhoPrev = None,
#     tEnd = 2e-6,
#     numT = 500,
#     verboseResults = False,
#     bool1042 = False
# ):
#     """
#     Compute steady-state populations of an 8-level NV center model.

#     Returns:
#         pop_ss: array of populations [g0,g+,g-,e0,e+,e-,s1,s2]
#     """
#     # --------------------------------------------------
#     # Fundamental optical rates
#     # --------------------------------------------------
#     Gamma_pump = sigma_p * I_p * lamb_p / (h * c)
#     Gamma_IR_singlet = sigma_s * I_s * lamb_s / (h * c)
#     scalar = 1
#     Int2mWp = 1 * np.pi*300e-9**2 * 1e3
#     Int2Ns = 1 / (5e2 * 1e4)
#     Pp = I_p * Int2mWp
#     Ns = I_s * Int2Ns

#     paramsRatesFull = paramsRates * paramsRatesReal
    
#     # sigurd's model
#     Gamma_pump           = Pp * paramsRatesFull[0] # Kem
#     Gamma_pump_NV0ge     = Pp * paramsRatesFull[1]     # Ke0
    
#     Gamma_pump_NVe_NV0m  = Pp * paramsRatesFull[2]    # K25
#     Gamma_pump_NV0m_NVg  = Pp * paramsRatesFull[3]      # K51
#     Gamma_pump_NV0e_NVg  = Pp * paramsRatesFull[4]    # K71
#     Gamma_pump_NV0e_NVs1 = Pp * paramsRatesFull[5]    # K73
#     Gamma_pump_NV0e_NVs2 = Pp * paramsRatesFull[6]    # K74
#     Gamma_pump_NVs2_NV0g = Pp * paramsRatesFull[7]   # K46
    
#     # --------------------------------------------------
#     # IR-assisted processes
#     # --------------------------------------------------
#     # sigurd's model
#     Gamma_IR_NVe_NV0m = Ns**2 * paramsRatesFull[8]  # K25
#     Gamma_IR_NVe_NV0m_single = Ns * paramsRatesFull[9] # K25
#     Gamma_IR_NV0m_NVg = Ns**2 * paramsRatesFull[10] # K51
#     Gamma_IR_NV0e_NVs1 = Ns**2 * paramsRatesFull[11] # K73
#     Gamma_IR_NV0e_NVs2 = Ns * paramsRatesFull[12] # K74
#     Gamma_IR_NVs2_NV0g = Ns**2 * paramsRatesFull[13] # K46
#     Gamma_IR_NV0e_NVg  = Ns *paramsRatesFull[14] # K71
    
#     Gamma_pump_s1_NV0g       = Pp    * paramsRatesFull[15] # k36
#     Gamma_IR_s1_NV0g          = Ns    * paramsRatesFull[16] # k36
#     if not(bool1042):
#         Gamma_IR_singlet          = Ns    * paramsRatesFull[17] # k36 
       
#     gamma_NV0_em = paramsRatesFull[18] #k_75
#     gamma_NV0_mg = paramsRatesFull[19]   #k_56
#     def fmt(x):
#         return f"{x: .1e}"   # 2 significant digits, scientific notation, aligned

#     if verboseResults:
#         print("\nk109:", fmt(gamma_NV0_em))
#         print("k98 :", fmt(gamma_NV0_mg))
    
#         print("\nPump")
#         scale_p = (I_p * lamb_p / (h * c))
#         print("s03, s14, s25 :", fmt(Gamma_pump / scale_p))
#         print("s810          :", fmt(Gamma_pump_NV0ge / scale_p))
#         print("s100–102      :", fmt(Gamma_pump_NV0e_NVg / 3 / scale_p))
#         print("s90–92        :", fmt(Gamma_pump_NV0m_NVg / scale_p))
#         print("s39–59        :", fmt(Gamma_pump_NVe_NV0m / scale_p))
#         print("s68           :", fmt(Gamma_pump_s1_NV0g / scale_p))
#         print("s107          :", fmt(Gamma_pump_NV0e_NVs2 / scale_p))
    
#         print("\nIR")
#         scale_s = (I_s * lamb_s / (h * c))
#         print("s100–102      :", fmt(Gamma_IR_NV0e_NVg / 3 / scale_s))
#         print("s90–92 (TPA)  :", fmt(Gamma_IR_NV0m_NVg / 3 / scale_s**2))
#         print("s39–59        :", fmt(Gamma_IR_NVe_NV0m_single / scale_s))
#         print("s39–59 (TPA)  :", fmt(Gamma_IR_NVe_NV0m / scale_s**2))
#         print("s68           :", fmt(Gamma_IR_s1_NV0g / scale_s))
#         print("s107          :", fmt(Gamma_IR_NV0e_NVs2 / scale_s))
#         print("s106 (TPA)    :", fmt(Gamma_IR_NV0e_NVs1 / scale_s**2))
#         print("s76           :", fmt(Gamma_IR_singlet / scale_s))
#         print("s78           :", fmt(Gamma_IR_NVs2_NV0g / scale_s**2))

#     # --------------------------------------------------
#     # Combined effective rates (exactly as in QuTiP)
#     # --------------------------------------------------
#     Gamma_e_to_m = (
#         Gamma_pump_NVe_NV0m
#         + Gamma_IR_NVe_NV0m
#         + Gamma_IR_NVe_NV0m_single
#     )

#     Gamma_s1_NV0g    = Gamma_pump_s1_NV0g + Gamma_IR_s1_NV0g
#     Gamma_eNV0_to_g  = Gamma_pump_NV0e_NVg  + Gamma_IR_NV0e_NVg
#     Gamma_eNV0_to_s1 = Gamma_pump_NV0e_NVs1 + Gamma_IR_NV0e_NVs1
#     Gamma_eNV0_to_s2 = Gamma_pump_NV0e_NVs2 + Gamma_IR_NV0e_NVs2
#     Gamma_s2_to_gNV0 = Gamma_pump_NVs2_NV0g + Gamma_IR_NVs2_NV0g
    

#     gammaIR_0 = 0
#     gammaIR_1 = 0
#     gammaIR_2 = 0
#     gammaIR_3 = Gamma_IR_NVe_NV0m_single + Gamma_IR_NVe_NV0m
#     gammaIR_4 = Gamma_IR_NVe_NV0m_single + Gamma_IR_NVe_NV0m
#     gammaIR_5 = Gamma_IR_NVe_NV0m_single + Gamma_IR_NVe_NV0m
#     gammaIR_6 = Gamma_IR_s1_NV0g
#     gammaIR_7 = Gamma_IR_NVs2_NV0g + Gamma_IR_singlet
#     gammaIR_8 = 0
#     gammaIR_9 = Gamma_IR_NV0m_NVg
#     gammaIR_10 = Gamma_IR_NV0e_NVs2 + Gamma_IR_NV0e_NVs1 + Gamma_IR_NV0e_NVg
    
#     gammaIRtot = np.array([gammaIR_0, gammaIR_1, gammaIR_2, gammaIR_3, gammaIR_4,
#                   gammaIR_5, gammaIR_6 , gammaIR_7, gammaIR_8, gammaIR_9 ,
#                   gammaIR_10])
    
#     return gammaIRtot

# from dataclasses import dataclass

# def build_rate_matrix_from_intensities(
#     I_p, I_s, Omega, gamma_phi, detuning, zeeman,
#     sigma_p, sigma_s, lamb_p, lamb_s, h, c,
#     gamma_eg, gamma_isc_0, gamma_isc_1,
#     gamma_isc_g0, gamma_isc_g1, gamma_s,
#     gamma_NV0_eg, gamma_NV0_em, gamma_NV0_mg, paramsRates, 
#     bool1042 = False
# ):
#     dim = DIM
#     R = np.zeros((dim, dim))

#     # --------------------------------------------------
#     # Fundamental optical rates
#     # --------------------------------------------------
#     Gamma_pump = sigma_p * I_p * lamb_p / (h * c)
#     Gamma_IR_singlet = sigma_s * I_s * lamb_s / (h * c)
#     scalar = 1
#     Int2mWp = 1 * np.pi*300e-9**2 * 1e3
#     Int2Ns = 1 / (5e2 * 1e4)
#     Pp = I_p * Int2mWp
#     Ns = I_s * Int2Ns

#     paramsRatesFull = paramsRates
    
#     # sigurd's model
#     Gamma_pump           = Pp * paramsRatesFull[0] # Kem
#     Gamma_pump_NV0ge     = Pp * paramsRatesFull[1]     # Ke0
    
#     Gamma_pump_NVe_NV0m  = Pp * paramsRatesFull[2]    # K25
#     Gamma_pump_NV0m_NVg  = Pp * paramsRatesFull[3]      # K51
#     Gamma_pump_NV0e_NVg  = Pp * paramsRatesFull[4]    # K71
#     Gamma_pump_NV0e_NVs1 = Pp * paramsRatesFull[5]    # K73
#     Gamma_pump_NV0e_NVs2 = Pp * paramsRatesFull[6]    # K74
#     Gamma_pump_NVs2_NV0g = Pp * paramsRatesFull[7]   # K46
    
#     # --------------------------------------------------
#     # IR-assisted processes
#     # --------------------------------------------------
#     # sigurd's model
#     Gamma_IR_NVe_NV0m = Ns**2 * paramsRatesFull[8]  # K25
#     Gamma_IR_NVe_NV0m_single = Ns * paramsRatesFull[9] # K25
#     Gamma_IR_NV0m_NVg = Ns**2 * paramsRatesFull[10] # K51
#     Gamma_IR_NV0e_NVs1 = Ns**2 * paramsRatesFull[11] # K73
#     Gamma_IR_NV0e_NVs2 = Ns * paramsRatesFull[12] # K74
#     Gamma_IR_NVs2_NV0g = Ns**2 * paramsRatesFull[13] # K46
#     Gamma_IR_NV0e_NVg  = Ns *paramsRatesFull[14] # K71
    
#     Gamma_pump_s1_NV0g       = Pp    * paramsRatesFull[15] # k36
#     Gamma_IR_s1_NV0g          = Ns    * paramsRatesFull[16] # k36
#     if not(bool1042):
#         Gamma_IR_singlet          = Ns    * paramsRatesFull[17] # k36 
       
#     gamma_NV0_em = paramsRatesFull[18] #k_75
#     gamma_NV0_mg = paramsRatesFull[19]   #k_56

#     # --------------------------------------------------
#     # Combined effective rates (exactly as in QuTiP)
#     # --------------------------------------------------
#     Gamma_e_to_m = (
#         Gamma_pump_NVe_NV0m
#         + Gamma_IR_NVe_NV0m
#         + Gamma_IR_NVe_NV0m_single
#     )

#     Gamma_s1_NV0g    = Gamma_pump_s1_NV0g + Gamma_IR_s1_NV0g
#     Gamma_eNV0_to_g  = Gamma_pump_NV0e_NVg  + Gamma_IR_NV0e_NVg
#     Gamma_eNV0_to_s1 = Gamma_pump_NV0e_NVs1 + Gamma_IR_NV0e_NVs1
#     Gamma_eNV0_to_s2 = Gamma_pump_NV0e_NVs2 + Gamma_IR_NV0e_NVs2
#     Gamma_s2_to_gNV0 = Gamma_pump_NVs2_NV0g + Gamma_IR_NVs2_NV0g

#     # --------------------------------------------------
#     # Rate map (1-to-1 with nv_steady_state)
#     # --------------------------------------------------
#     rate_map = {
#         # g → e
#         "g0e0": Gamma_pump,
#         "gpep": Gamma_pump,
#         "gmem": Gamma_pump,

#         # e → mNV0
#         "e0m": Gamma_e_to_m,
#         "epm": Gamma_e_to_m,
#         "emm": Gamma_e_to_m,

#         # NV0 optical channels
#         "eNV0g0": Gamma_eNV0_to_g/3,
#         "eNV0gp": Gamma_eNV0_to_g/3,
#         "eNV0gm": Gamma_eNV0_to_g/3,
#         "eNV0s1": Gamma_eNV0_to_s1,
#         "eNV0s2": Gamma_eNV0_to_s2,
#         "s2gNV0": Gamma_s2_to_gNV0,

#         "gNV0eNV0": Gamma_pump_NV0ge,
#         "mNV0g0": (Gamma_pump_NV0m_NVg + Gamma_IR_NV0m_NVg)/3,
#         "mNV0gp": (Gamma_pump_NV0m_NVg + Gamma_IR_NV0m_NVg)/3,
#         "mNV0gm": (Gamma_pump_NV0m_NVg + Gamma_IR_NV0m_NVg)/3,
#         's1gNV0': Gamma_s1_NV0g,

#         # IR singlet pump
#         "s2s1": Gamma_IR_singlet,

#         # Radiative decay
#         "e0g0": gamma_eg,
#         "epgp": gamma_eg,
#         "emgm": gamma_eg,

#         # NV0 decay
#         "eNV0gNV0": gamma_NV0_eg,
#         "mNV0gNV0": gamma_NV0_mg,
#         "eNV0mNV0": gamma_NV0_em,

#         # ISC
#         "e0_s1": gamma_isc_0,
#         "ep_s1": gamma_isc_1,
#         "em_s1": gamma_isc_1,

#         # Singlet relaxation
#         "s1s2": gamma_s,
        
#         "s2_g0": gamma_isc_g0,
#         "s2_gp": gamma_isc_g1,
#         "s2_gm": gamma_isc_g1,
#     }

#     for key, rate in rate_map.items():
#         if rate == 0.0:
#             continue
#         i, j = collapse_to_rate_indices(C[key])
#         R[i, j] += rate
#         R[j, j] -= rate

#     # --------------------------------------------------
#     # MW incoherent coupling (detuning + Zeeman resolved)
#     # --------------------------------------------------
#     T2 = 1.0 / gamma_phi

#     Delta_gp = detuning + zeeman
#     Delta_gm = detuning - zeeman

#     Wmw_gp = (Omega**2 * T2 / 2.0) / (1 + (Delta_gp * T2)**2)
#     Wmw_gm = (Omega**2 * T2 / 2.0) / (1 + (Delta_gm * T2)**2)

#     for j, Wmw in [(gp, Wmw_gp), (gm, Wmw_gm)]:
#         R[g0, j] += Wmw
#         R[j, g0] += Wmw
#         R[g0, g0] -= Wmw
#         R[j, j] -= Wmw

#     return R

def add_rate(R, i_from, i_to, rate):
    """
    Population flows from i_from -> i_to with rate.
    """
    if rate == 0:
        return
    R[i_to,   i_from] += rate
    R[i_from, i_from] -= rate


def build_rate_matrix_from_intensities_purematrix_noRabi(
    I_p, I_s,
    sigma_p, sigma_s, lamb_p, lamb_s, h, c,
    gamma_eg, gamma_isc_0, gamma_isc_1,
    gamma_isc_g0, gamma_isc_g1, gamma_s,
    gamma_NV0_eg, gamma_NV0_em, gamma_NV0_mg,
    paramsRates,
    bool1042 = False
):

    dim = 11
    R = np.zeros((dim, dim), dtype=np.float64)

    # --------------------------------------------------
    # Intensity scaling
    # --------------------------------------------------
    Int2mWp = np.pi * (300e-9)**2 * 1e3
    Int2Ns  = 1 / (5e2 * 1e4)

    Pp = I_p * Int2mWp
    # print('Pp: ', Pp)
    Ns = I_s * Int2Ns
    # print('Ns: ', Ns)
    p  = paramsRates

    # --------------------------------------------------
    # Pump-induced rates
    # --------------------------------------------------
    Gamma_pump               = Pp * p[0]   # Kem
    Gamma_pump_NV0ge         = Pp * p[1]   # Ke0
    Gamma_pump_NVe_NV0m      = Pp * p[2]   # K25
    Gamma_pump_NVe_NV0g      = Pp * p[20]  # K26
    Gamma_pump_NV0m_NVg      = Pp * p[3]   # K51
    Gamma_pump_NV0e_NVg      = Pp * p[4]   # K71
    Gamma_pump_NV0e_NVs1     = Pp * p[5]   # K73
    Gamma_pump_NV0e_NVs2     = Pp * p[6]   # K74
    Gamma_pump_NVs2_NV0g     = Pp * p[7]   # K46
    

    
    
    Gamma_IR_singlet = sigma_s * I_s * lamb_s / (h * c)

    # --------------------------------------------------
    # IR-assisted rates
    # --------------------------------------------------
    Gamma_IR_NVe_NV0m         = Ns**2 * p[8]
    Gamma_IR_NVe_NV0m_single  = Ns    * p[9]
    Gamma_IR_NVe_NV0g         = Ns    * p[21]  # K26
    Gamma_IR_NV0m_NVg         = Ns**2 * p[10]
    Gamma_IR_NV0e_NVs1        = Ns**2 * p[11]
    Gamma_IR_NV0e_NVs2        = Ns    * p[12]
    Gamma_IR_NVs2_NV0g        = Ns**2 * p[13]
    Gamma_IR_NV0e_NVg         = Ns    * p[14]
    
    Gamma_pump_s1_NV0g       = Pp    * p[15] # k36
    Gamma_IR_s1_NV0g          = Ns    * p[16] # k36
    if not(bool1042):
        Gamma_IR_singlet          = Ns    * p[17] # k36 
    gamma_NV0_em = p[18] #k_75
    gamma_NV0_mg = p[19]   #k_56

    # --------------------------------------------------
    # Bare decay rates
    # --------------------------------------------------
    k30 = gamma_eg
    k41 = gamma_eg
    k52 = gamma_eg

    k36 = gamma_isc_0
    k46 = gamma_isc_1
    k56 = gamma_isc_1

    k67 = gamma_s

    k70 = gamma_isc_g0
    k71 = gamma_isc_g1
    k72 = gamma_isc_g1

    k108 = gamma_NV0_eg
    k109 = gamma_NV0_em
    k98  = gamma_NV0_mg

    # --------------------------------------------------
    # Combined effective rates
    # --------------------------------------------------
    Gamma_e_to_m = (
        Gamma_pump_NVe_NV0m
        + Gamma_IR_NVe_NV0m
        + Gamma_IR_NVe_NV0m_single
    )
    
    Gamma_e_to_g = (
        Gamma_pump_NVe_NV0g
        + Gamma_IR_NVe_NV0g
    )
    Gamma_NV0m_to_NVg = (
        Gamma_pump_NV0m_NVg
        + Gamma_IR_NV0m_NVg
    )

    Gamma_s1_NV0g    = Gamma_pump_s1_NV0g + Gamma_IR_s1_NV0g
    Gamma_eNV0_to_g  = Gamma_pump_NV0e_NVg  + Gamma_IR_NV0e_NVg
    Gamma_eNV0_to_s1 = Gamma_pump_NV0e_NVs1 + Gamma_IR_NV0e_NVs1
    Gamma_eNV0_to_s2 = Gamma_pump_NV0e_NVs2 + Gamma_IR_NV0e_NVs2
    Gamma_s2_to_gNV0 = Gamma_pump_NVs2_NV0g + Gamma_IR_NVs2_NV0g

    # --------------------------------------------------
    # Populate matrix
    # --------------------------------------------------
    # Optical pump
    add_rate(R, 0, 3, Gamma_pump)
    add_rate(R, 1, 4, Gamma_pump)
    add_rate(R, 2, 5, Gamma_pump)

    # Radiative decay
    add_rate(R, 3, 0, k30)
    add_rate(R, 4, 1, k41)
    add_rate(R, 5, 2, k52)

    # ISC
    add_rate(R, 3, 6, k36)
    add_rate(R, 4, 6, k46)
    add_rate(R, 5, 6, k56)

    # Singlet decay
    add_rate(R, 6, 7, k67)

    # Return to ground
    add_rate(R, 7, 0, k70)
    add_rate(R, 7, 1, k71)
    add_rate(R, 7, 2, k72)

    # NV0 manifold
    add_rate(R, 10, 8, k108)
    add_rate(R, 10, 9, k109)
    add_rate(R, 9,  8, k98)

    # Cross-manifold transitions
    add_rate(R, 6, 8, Gamma_s1_NV0g)
    add_rate(R, 3, 9, Gamma_e_to_m)
    add_rate(R, 4, 9, Gamma_e_to_m)
    add_rate(R, 5, 9, Gamma_e_to_m)
    
    add_rate(R, 3, 8, Gamma_e_to_g)
    add_rate(R, 4, 8, Gamma_e_to_g)
    add_rate(R, 5, 8, Gamma_e_to_g)

    add_rate(R, 8, 10, Gamma_pump_NV0ge)
    add_rate(R, 7, 6, Gamma_IR_singlet)
    
    add_rate(R, 9, 0, Gamma_NV0m_to_NVg / 3)
    add_rate(R, 9, 1, Gamma_NV0m_to_NVg / 3)
    add_rate(R, 9, 2, Gamma_NV0m_to_NVg / 3)

    add_rate(R, 10, 0, Gamma_eNV0_to_g / 3)
    add_rate(R, 10, 1, Gamma_eNV0_to_g / 3)
    add_rate(R, 10, 2, Gamma_eNV0_to_g / 3)

    add_rate(R, 10, 7, Gamma_eNV0_to_s2)
    add_rate(R, 10, 6, Gamma_eNV0_to_s1)
    add_rate(R, 7,  8, Gamma_s2_to_gNV0)

    return R



def solve_steady_state_rate(R):
    dim = R.shape[0]

    A = np.vstack([R, np.ones(dim)])
    b = np.zeros(dim + 1)
    b[-1] = 1.0

    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(p, 0.0, 1.0)

# @njit(fastmath=True, cache=True)
# def solve_rate_jit(R):
#     dim = R.shape[0]
#     A = R.copy()
#     b = np.zeros(dim)
#     A[-1, :] = 1.0
#     b[-1] = 1.0
#     return np.linalg.solve(A, b)

def nv_steady_state_fast(
    Omega, #rad/s
    I_s,   #rad/s
    I_p, #W/m^2
    gamma_phi,   #Hz
    gamma_eg,
    gamma_isc_0,
    gamma_isc_1,
    gamma_isc_g0,
    gamma_isc_g1,
    gamma_s,
    gamma_NV0_eg,  #k^0_f
    gamma_NV0_em, #k_75
    gamma_NV0_mg,   #k_56
    sigma_p, # m^%-2
    sigma_s, # m^%-2
    lamb_s,   # IR probe
    lamb_p,    # green pump
    c, #m/s
    h,      # Planck constant [J·s]
    paramsRates,
    paramsRatesReal,
    detuning=0, #rad/s
    zeeman = 0, #rad/s\
    bool1042 = False,
    ):
    # R = build_rate_matrix_from_intensities(
    #     Omega=Omega, #rad/s
    #     I_s=I_s,   #rad/s
    #     I_p=I_p, #W/m^2
    #     gamma_phi=gamma_phi,   #Hz
    #     detuning=detuning, #rad/s
    #     zeeman = zeeman, #rad/s
    #     gamma_eg=gamma_eg,
    #     gamma_isc_0=gamma_isc_0,
    #     gamma_isc_1=gamma_isc_1,
    #     gamma_isc_g0=gamma_isc_g0,
    #     gamma_isc_g1=gamma_isc_g1,
    #     gamma_s=gamma_s,
    #     gamma_NV0_eg = gamma_NV0_eg,  #k^0_f
    #     gamma_NV0_em = gamma_NV0_em, #k_75
    #     gamma_NV0_mg = gamma_NV0_mg,   #k_56
    #     sigma_p = sigma_p, # m^%-2
    #     sigma_s = sigma_s, # m^%-2
    #     lamb_s = lamb_s,   # IR probe
    #     lamb_p = lamb_p,    # green pump
    #     c = c, #m/s
    #     h = h,     # Planck constant [J·s])
    #     paramsRates = paramsRatesReal * paramsRates
    #     )
    R = build_rate_matrix_from_intensities_purematrix_noRabi(
        I_s=I_s,   #rad/s
        I_p=I_p, #W/m^2
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
        h = h,     # Planck constant [J·s])
        paramsRates = paramsRatesReal * paramsRates,
        bool1042 = bool1042
        )
    pop = solve_steady_state_rate(R)
    return pop