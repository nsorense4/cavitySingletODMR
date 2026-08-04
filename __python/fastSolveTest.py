# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:16:56 2025

@author: nicho
"""


from linbladMESolverFunctionChargeConversionLiouvillian import *
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve


params = dict(
    I_p=1e8,
    I_s=1e11,
    Omega=2*np.pi*1e6,
    gamma_phi=2*np.pi*1e6,
    detuning = 0.1e9*2*np.pi,
    zeeman = 0.1e9*2*np.pi,
    sigma_p=sigma_p,
    sigma_s=sigma_s,
    lamb_p=lamb_p,
    lamb_s=lamb_s,
    h=h,
    c=c,
    gamma_ge=66e6,
    gamma_isc_0=7.9e6,
    gamma_isc_1=53e6,
    gamma_isc_g0=1e6,
    gamma_isc_g1=0.7e6,
    gamma_s=1e9,
    gamma_NV0_eg=53e6,
    gamma_NV0_em=1.4e3,
    gamma_NV0_mg=4e3,
)

# --- QuTiP ---
pop_qutip = nv_steady_state(**params)

# --- Rate model ---
R = build_rate_matrix_from_intensities(**params)
pop_rate = solve_steady_state_rate(R)

# Compare
diff = np.abs(np.array(pop_qutip) - pop_rate)
print("Max population error:", diff.max())

# comapre timing
# warm-up
nv_steady_state(**params)

t0 = time.perf_counter()
for _ in range(10):
    pop_qutip = nv_steady_state(**params)
t_qutip = (time.perf_counter() - t0) / 10

# warm-up
R = build_rate_matrix_from_intensities(**params)
solve_steady_state_rate(R)

t0 = time.perf_counter()
for _ in range(1000):
    R = build_rate_matrix_from_intensities(**params)
    pop_rate = solve_steady_state_rate(R)
t_rate = (time.perf_counter() - t0) / 1000


# warm-up compile
solve_rate_jit(R)

t0 = time.perf_counter()
for _ in range(10000):
    pop_rate = solve_rate_jit(R)
t_jit = (time.perf_counter() - t0) / 10000

