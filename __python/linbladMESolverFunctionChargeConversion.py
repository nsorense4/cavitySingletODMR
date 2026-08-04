# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:00:27 2025

@author: nicho
"""

import numpy as np
import qutip as qt

def nv_steady_state(
    Omega, #rad/s
    I_s,   #rad/s
    I_p, #W/m^2
    gamma_phi,   #Hz
    detuning=0, #rad/s
    zeeman = 0, #rad/s
    steadystatebool = True,
    tEnd = 2e-6,
    numT = 500,
    gamma_ge=66e6,
    gamma_isc_0=7.9e6,
    gamma_isc_1=53e6,
    gamma_isc_g0=1e6,
    gamma_isc_g1=0.7e6,
    gamma_s=1e9,
    gamma_NV0_eg = 53e6,  #k^0_f
    gamma_NV0_em = 1.4e3, #k_75
    gamma_NV0_mg = 4e3,   #k_56
    sigma_p = 3e-21, # m^%-2
    sigma_s = 2e-22, # m^%-2
    lamb_s = 1042e-9,   # IR probe
    lamb_p = 532e-9,    # green pump
    c = 299792458.0, #m/s
    h = 6.62607015e-34      # Planck constant [J·s]
):
    """
    Compute steady-state populations of an 8-level NV center model.

    Returns:
        pop_ss: array of populations [g0,g+,g-,e0,e+,e-,s1,s2]
    """
    
    
    Gamma_IR_singlet = sigma_s * I_s * lamb_s / (h * c)
    Gamma_pump = sigma_p * I_p * lamb_p / (h * c)
    Gamma_pump_NVe_NV0m = Gamma_pump*10.3e3/10e6  #k^i_25
    Gamma_pump_NV0m_NVg = Gamma_pump*12.6e3/10e6  #k^r_51
    Gamma_pump_NV0e_NVs2 = Gamma_pump*3e3/10e6     #k^r_74
    Gamma_pump_NV0ge = Gamma_pump*18e6/10e6     #k^0_e
    IRratescaling = 1e-8
    Gamma_IR_NVe_NV0m = I_s**2 * IRratescaling*0.0000001*2    #k^i_25 2IR
    Gamma_IR_NV0e_NVs2 = I_s * IRratescaling*239/0.11*0.7 #k^i_74 1IR
    
    DIM = 11
    BASIS = [qt.basis(DIM, i) for i in range(DIM)]

    # State indices
    g0, gp, gm = 0, 1, 2
    e0, ep, em = 3, 4, 5
    s1, s2 = 6, 7
    gNV0, eNV0, mNV0 = 8, 9, 10

    # MW couplings
    H_MW_gp  = BASIS[g0] * BASIS[gp].dag() + BASIS[gp] * BASIS[g0].dag()
    H_MW_gm  = BASIS[g0] * BASIS[gm].dag() + BASIS[gm] * BASIS[g0].dag()

    # Zeeman / detuning projectors
    H_gp = BASIS[gp] * BASIS[gp].dag()
    H_gm = BASIS[gm] * BASIS[gm].dag()

    C = {}

    # Optical pumping g -> e
    C["g0e0"] = BASIS[e0] * BASIS[g0].dag()
    C["gpep"] = BASIS[ep] * BASIS[gp].dag()
    C["gmem"] = BASIS[em] * BASIS[gm].dag()

    # e -> mNV0
    C["e0m"] = BASIS[mNV0] * BASIS[e0].dag()
    C["epm"] = BASIS[mNV0] * BASIS[ep].dag()
    C["emm"] = BASIS[mNV0] * BASIS[em].dag()

    # NV0 transitions
    C["g0eNV0"] = BASIS[eNV0] * BASIS[gNV0].dag()
    C["mNV0g0"] = BASIS[g0] * BASIS[mNV0].dag()
    C["eNV0s2"] = BASIS[s2] * BASIS[eNV0].dag()

    # Singlet
    C["s1s2"] = BASIS[s1] * BASIS[s2].dag()
    C["s2s1"] = BASIS[s2] * BASIS[s1].dag()

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
    # Singlet decay s1 -> s2
    # =========================
    C["s1_s2"] = BASIS[s2] * BASIS[s1].dag()

    # =========================
    # s2 -> ground triplet
    # =========================
    C["s2_g0"] = BASIS[g0] * BASIS[s2].dag()
    C["s2_gp"] = BASIS[gp] * BASIS[s2].dag()
    C["s2_gm"] = BASIS[gm] * BASIS[s2].dag()

    # Pure dephasing
    for i in [g0, gp, gm, e0, ep, em]:
        C[f"deph_{i}"] = BASIS[i] * BASIS[i].dag()
    

    dim = DIM
    basis = BASIS

    # Hamiltonian    
    H = (
            (Omega / 2) * (H_MW_gp + H_MW_gm)
            + (zeeman + detuning) * H_gp
            + (zeeman - detuning) * H_gm
        )

    # Collapse operators
    c_ops = []

    # Optical pumping g -> e
    c_ops.append(np.sqrt(Gamma_pump) * C["g0e0"])
    c_ops.append(np.sqrt(Gamma_pump) * C["gpep"])
    c_ops.append(np.sqrt(Gamma_pump) * C["gmem"])
        
    # Optical pumping eNV -> mNV0
    c_ops.append(np.sqrt(Gamma_pump_NVe_NV0m + Gamma_IR_NVe_NV0m) * C["e0m"])
    c_ops.append(np.sqrt(Gamma_pump_NVe_NV0m + Gamma_IR_NVe_NV0m) * C["epm"])
    c_ops.append(np.sqrt(Gamma_pump_NVe_NV0m + Gamma_IR_NVe_NV0m) * C["emm"])
    
    # Optical pumping g -> e NV0
    c_ops.append(np.sqrt(Gamma_pump_NV0ge) * C["g0eNV0"])
        
    # Optical pumping mNV0 -> g
    c_ops.append(np.sqrt(Gamma_pump_NV0m_NVg) * C["mNV0g0"])
    
    # Optical pumping eNV0 -> s2
    c_ops.append(np.sqrt(Gamma_pump_NV0e_NVs2 + Gamma_IR_NV0e_NVs2) * C["eNV0s2"])
    
    # IR pumping of the singlet state
    c_ops.append(np.sqrt(Gamma_IR_singlet) * C["s1s2"])

    # Radiative decay e -> g
    c_ops.append(np.sqrt(gamma_ge) * C["e0g0"])
    c_ops.append(np.sqrt(gamma_ge) * C["epgp"])
    c_ops.append(np.sqrt(gamma_ge) * C["emgm"])
        
    # radiative decay eNV0 -> gNV0
    c_ops.append(np.sqrt(gamma_NV0_eg) * C["eNV0gNV0"])
    
    # Nonradiative decay mNV0 -> gNV0
    c_ops.append(np.sqrt(gamma_NV0_mg) * C["mNV0gNV0"])
    
    # Nonradiative decay eNV0 -> mNV0
    c_ops.append(np.sqrt(gamma_NV0_em) * C["eNV0mNV0"])
    
    # ISC e -> s
    c_ops.append(np.sqrt(gamma_isc_0) * C["e0_s1"])
    c_ops.append(np.sqrt(gamma_isc_1) * C["ep_s1"])
    c_ops.append(np.sqrt(gamma_isc_1) * C["em_s1"])

    # Singlet decay s1 -> s2
    c_ops.append(np.sqrt(gamma_s) * C["s1_s2"])

    # s2 -> ground triplet
    c_ops.append(np.sqrt(gamma_isc_g0) * C["s2_g0"])
    c_ops.append(np.sqrt(gamma_isc_g1) * C["s2_gp"])
    c_ops.append(np.sqrt(gamma_isc_g1) * C["s2_gm"])

    # Pure dephasing on triplet states
    for idx in [g0, gp, gm, e0, ep, em]:
        c_ops.append(np.sqrt(gamma_phi) * C[f"deph_{idx}"])


    if steadystatebool:
        # Compute steady state
        rho_ss = qt.steadystate(H, c_ops, method='direct')
    
        # Extract populations
        pop_ss = [np.real((basis[i].dag() * rho_ss * basis[i])) for i in range(dim)]
        return pop_ss
    else:
        # ---------------------------
        # Initial state
        # ---------------------------
        rho0 = basis[g0]*basis[g0].dag()  # start in ground ms=0

        # ---------------------------
        # Solve master equation
        # ---------------------------
        obs_list = [basis[i]*basis[i].dag() for i in range(dim)]  # populations
        
        opts = qt.Options(
            nsteps=10000,
            rtol=1e-6,
            atol=1e-8,
            store_states=False
        )
        t_list = np.linspace(0, tEnd, numT)

        result = qt.mesolve(H, rho0, t_list, c_ops, obs_list, options=opts)
        return result, t_list


