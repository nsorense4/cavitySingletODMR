# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:00:27 2025

@author: nicho
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from plot_custom import plot_custom, thesisColorPalette, plotPalette, makeTopLeftSpinesInvisible

labelfontsize = 9
numfontsize = 9
textwidth = 174/25.4#8.5 - 2 # in inches
PRQuantumWidthFull = textwidth

c = 299792458.0 #m/s
h = 6.62607015e-34      # Planck constant [J·s]

lamb_s = 1042e-9   # IR probe
lamb_p = 532e-9    # green pump
w_s = c/lamb_s   # IR probe
w_p = c/lamb_p    # green pump

sigma_p = 3e-21 # m^%-2
sigma_s = 2e-22

def nv_steady_state(
    Omega=2*np.pi*1e6,
    I_s=2*np.pi*0e6,
    gamma_phi=0.9e6,
    detuning=0e9*2*np.pi,
    I_p=5e6,
    zeeman = 0.4e9*2*np.pi,
    steadystatebool = True,
    tEnd = 2e-6,
    numT = 500
):
    """
    Compute steady-state populations of an 8-level NV center model.

    Returns:
        pop_ss: array of populations [g0,g+,g-,e0,e+,e-,s1,s2]
    """
    OmegaIR = sigma_s * I_s * lamb_s / (h * c)
    Gamma_pump = sigma_p * I_p * lamb_p / (h * c)
    
    gamma_ge=66e6,
    gamma_isc_0=7.9e6,
    gamma_isc_1=53e6,
    gamma_isc_g0=1e6,
    gamma_isc_g1=0.7e6,
    gamma_s=1e9,
    
    dim = 8
    basis = [qt.basis(dim, i) for i in range(dim)]
    g0, gp, gm = 0, 1, 2
    e0, ep, em = 3, 4, 5
    s1, s2 = 6, 7

    # Hamiltonian
    H = qt.Qobj(np.zeros((dim, dim), dtype=complex))
    # MW coupling g0 <-> g±1
    H += (Omega)/2 * (basis[g0]*basis[gp].dag() + basis[gp]*basis[g0].dag())
    H += (Omega)/2 * (basis[g0]*basis[gm].dag() + basis[gm]*basis[g0].dag())
    # H += 1 * OmegaIR * (basis[s1]*basis[s2].dag() + basis[s2]*basis[s1].dag())
    
    H += (zeeman + detuning) * (basis[gp]*basis[gp].dag())
    H += (zeeman - detuning) * (basis[gm]*basis[gm].dag())

    # Collapse operators
    c_ops = []

    # Optical pumping g -> e
    for (g, e) in [(g0,e0),(gp,ep),(gm,em)]:
        c_ops.append(np.sqrt(Gamma_pump) * basis[e]*basis[g].dag())
        
    c_ops.append(np.sqrt(OmegaIR) * basis[s1]*basis[s2].dag())

    # Radiative decay e -> g
    for (eg, ee) in [(g0,e0),(gp,ep),(gm,em)]:
        c_ops.append(np.sqrt(gamma_ge) * basis[eg]*basis[ee].dag())

    # ISC e -> s
    c_ops.append(np.sqrt(gamma_isc_0) * basis[s1]*basis[e0].dag())
    for ee in [ep, em]:
        c_ops.append(np.sqrt(gamma_isc_1) * basis[s1]*basis[ee].dag())

    # Singlet decay s1 -> s2
    c_ops.append(np.sqrt(gamma_s) * basis[s2]*basis[s1].dag())

    # s2 -> ground triplet
    c_ops.append(np.sqrt(gamma_isc_g0) * basis[g0]*basis[s2].dag())
    for g in [gp, gm]:
        c_ops.append(np.sqrt(gamma_isc_g1) * basis[g]*basis[s2].dag())

    # Pure dephasing on triplet states
    for idx in [g0, gp, gm, e0, ep, em]:
        c_ops.append(np.sqrt(gamma_phi) * basis[idx]*basis[idx].dag())

    if steadystatebool:
        # Compute steady state
        rho_ss = qt.steadystate(H, c_ops)
    
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
        
        
        t_list = np.linspace(0, tEnd, numT)

        result = qt.mesolve(H, rho0, t_list, c_ops, obs_list)
        return result, t_list

def sweep_probe(Is_range, I_p, xopt = [], omega = 1e6):
    try:
        if xopt == 0:
            x = Is_range
    except: 
        x = xopt        

    m = len(Is_range)
    p_all = np.zeros((8, m))
    for i, Is in enumerate(Is_range):
        p_all[:, i] = nv_steady_state(detuning =  0.1e9, zeeman = 0.1e9, I_s = Is, I_p = I_p, Omega = omega)

    plt.figure(figsize=(3,3))
    colors = ['silver', 'grey', 'darkgrey', 'lightgreen', 'forestgreen', 'darkgreen', 'red', 'firebrick']
    for idx in range(8):
        plt.plot(x*1e-6*1e-4, p_all[idx, :], label=f'$p_{idx+1}$', color=colors[idx % len(colors)])
    try:
        if xopt == 0:
            plt.xscale('log'); 
    except:
        print()
    plt.yscale('log')
    plt.xlabel(r'IR intensity $I_s$ (MW/cm$^2$)')
    plt.ylabel('Population $p_i$')
    plt.title('IR (probe) intensity')
    plt.legend(loc = 'lower right')
    plt.ylim([None, 1])
    # plt.grid(True, which='both', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(
    #     "C:/Users/nicho/OneDrive - University of Calgary/Documents/LabBarclay/presentations/figuresPresentations/update20251113/" + "PopulationProbe" + ".svg", dpi=300, bbox_inches="tight",transparent=True
    # )
    plt.show()


def sweep_pump(Ip_range, I_s):
    m = len(Ip_range)
    p_all = np.zeros((8, m))
    for i, Ip in enumerate(Ip_range):
        p_all[:, i] = nv_steady_state(detuning =  0.1e9, zeeman = 0.1e9, I_s = I_s, I_p = Ip)

    plt.figure(figsize=(3,3))
    colors = ['silver', 'grey', 'darkgrey', 'lightgreen', 'forestgreen', 'darkgreen', 'red', 'firebrick']
    for idx in range(8):
        plt.plot(Ip_range*1e-10, p_all[idx, :], label=f'$p_{idx+1}$', color=colors[idx % len(colors)])
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(r'pump intensity $I_p$ (MW/cm$^2$)')
    plt.ylabel('Population $p_i$')
    plt.title('pump intensity')
    plt.legend(loc = 'lower right')
    plt.ylim([None, 1])
    # plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.savefig(
    #     "C:/Users/nicho/OneDrive - University of Calgary/Documents/LabBarclay/presentations/figuresPresentations/update20251113/" + "PopulationPump" + ".svg", dpi=300, bbox_inches="tight",transparent=True
    # )
    plt.show()

# %% system definitions
T2 = 0.5e-6
omega = 2*np.pi*2e6
I_p = 1e8
I_s = 1e8


#%% time dependent
result, tlist = nv_steady_state(detuning = 2*np.pi*0.2e9, 
                                zeeman = 2*np.pi*0.2e9, 
                                Omega = omega, 
                                I_p = I_p, 
                                I_s = I_s, 
                                gamma_phi = 1/np.pi/T2,
                                steadystatebool=False)

# ---------------------------
# Plot populations
# ---------------------------
fig, ax = plot_custom(
   PRQuantumWidthFull*0.4, 0.45*PRQuantumWidthFull, 
   r"$\lambda$ (nm)", 
   r'$T$ (au)',
   labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)

labels = [r'$|0\rangle$',r'$|1\rangle$',r'$|2\rangle$',r'$|3\rangle$',r'$|4\rangle$',r'$|5\rangle$',r'$|6\rangle$',r'$|7\rangle$']
colors = ['#b3b3b3ff','#666666ff','#000000ff','#ff5e09ff','#fe9800ff','#ffbd09ff','#800000ff','#190cffff'];
for i in range(8):
    ax.plot(tlist*1e6, result.expect[i], label=labels[i], color = colors[i], lw = 0.5)

ax.set_yscale('log'); 
# plt.yscale('log');
ax.set_xlabel(r"time ($\upmu$s)")
ax.set_ylabel(r"population, $p_i$ (au)")
# plt.legend(fontsize = labelfontsize, loc = 'lower right')
# plt.tight_layout()

tmin = 0
tmax = 2
ax.set_ylim([8e-7, 1.1e0])
ax.spines['bottom'].set_bounds(tmin, tmax)
xtick = ax.get_xticks()
ax.set_xticklabels([rf"${val:.1f}$" for val in xtick])

ax.spines['left'].set_bounds(1e-6,1)
ytick = ax.get_yticks(minor = True)
filt = np.logical_and(ytick > 1e-6, ytick < 1) 
ytick = ytick[filt]
ax.set_yticks(ytick,minor = True)
ax.spines['left'].set_bounds(1e-6,1)

ax.set_xticklabels([rf"${val:.1f}$" for val in xtick])

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
ax.tick_params(axis='x', which='both', top=False, bottom=True)
ax.spines["bottom"].set_visible(True)

# fig.savefig(
#     "./figures/" + "NVPowerSweep.svg", dpi=300, bbox_inches="tight",transparent=True
# )
# if saveBool:

plt.show()


#%% detuning
detunings = np.linspace(-0.2e9, 0.2e9, 200)
p7 = np.array([])
for det in detunings:
    populations = nv_steady_state(detuning = det*2*np.pi, 
                                    zeeman = 0.1e9*2*np.pi, 
                                    Omega = 2*np.pi*0.1e6, 
                                    I_p = 10e6, 
                                    I_s = 1e7, 
                                    gamma_phi = 1/np.pi/0.5e-6,
                                    steadystatebool=True)
    p7 = np.append(p7,populations[7] - populations[6])

plt.figure(figsize=(6,6))
plt.plot(detunings/1e9, p7/np.min(p7))
plt.show()

#%% detuning and power Sweep


N_IR = 5
N_green = 5
N_MW = 5
Ndets = 100;
I_IR = np.logspace(6, 16, N_IR)
I_green = np.logspace(6, 12, N_green)
Omega = np.logspace(2, 7, N_MW)
omega = 0.5e6
detunings = np.linspace(-0.2e9, 0.2e9, Ndets)

p7Green = np.zeros([N_green, Ndets])
p7IR = np.zeros([N_green, Ndets])
p7MW = np.zeros([N_green, Ndets])

for i, I_p in enumerate(I_green):
    for j, det in enumerate(detunings):
        populations =         nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = omega*2*np.pi, I_p = I_p)
        populationsDetuning = nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = 0, I_p = I_p)
        p7Green[i,j] = (populations[7] - populationsDetuning[7])
    p7Green[i,:] /= np.max(p7Green[i,:])


for i, I_s in enumerate(I_IR):
    for j, det in enumerate(detunings):
        populations =         nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = I_s, Omega = omega*2*np.pi, I_p = 1e8)
        populationsDetuning = nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = I_s, Omega = 0, I_p = 1e8)
        p7IR[i,j] = (populations[7] - populationsDetuning[7])
    p7IR[i,:] /= np.max(p7IR[i,:])


for i, Om in enumerate(Omega):
    for j, det in enumerate(detunings):
        populations =         nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = Om*2*np.pi, I_p = 1e8)
        populationsDetuning = nv_steady_state(detuning =  det*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = 1e10, Omega = 0, I_p = 1e8)
        p7MW[i,j] = (populations[7] - populationsDetuning[7])
    p7MW[i,:] /= np.max(p7MW[i,:])

#%%
fig, ax = plot_custom(
   PRQuantumWidthFull*0.6, 0.35*PRQuantumWidthFull, 
   [r"$I_{\text{P}}$ (W/m$^2$)", r"$I_{\text{P}}$ (W/m$^2$)", r"$I_{\text{P}}$ (W/m$^2$)"], 
   r'microwave detuning (MHz)',
   labelPadX = 3, labelPadY = 8, commonY = True, commonX = False, wSpaceSetting=0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   nrow = 1, ncol = 3, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)

X, Y = np.meshgrid(np.log10(I_green), detunings/1e6)
im = ax[0].pcolormesh(X, Y, np.transpose(np.log10(p7Green)), shading = 'auto')
X, Y = np.meshgrid(np.log10(I_IR), detunings/1e6)
im = ax[1].pcolormesh(X, Y, np.transpose(np.log10(p7IR)), shading = 'auto')
X, Y = np.meshgrid(np.log10(Omega), detunings/1e6)
im = ax[2].pcolormesh(X, Y, np.transpose(np.log10(p7MW)), shading = 'auto')
# im = ax.contourf(X, Y, p7, 100)
plt.tight_layout()

fig.subplots_adjust(wspace=0.1)

# cbar = plt.colorbar(im, orientation = 'horizontal', location = 'top')
# cbar.set_label(r'$m_7^{\text{(on)}} - m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)

plt.show()


#%% Power Sweep
N_IR = 100
N_green = 100
I_IR = np.logspace(6, 14, N_IR)
I_green = np.logspace(6, 12, N_green)
omega = 2*np.pi/np.pi/T2
p7 = np.zeros([N_IR, N_green])
p7On = np.zeros([N_IR, N_green])
p7Off = np.zeros([N_IR, N_green])
for i, I_s in enumerate(I_IR):
    for j, I_p in enumerate(I_green):
        populations =         nv_steady_state(detuning =  0.1e9*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = I_s, Omega = omega, I_p = I_p)
        populationsDetuning = nv_steady_state(detuning =  0.1e9*2*np.pi, zeeman = 0.1e9*2*np.pi, I_s = I_s, Omega = omega*0, I_p = I_p)
        p7[i,j] = (populations[7] - populationsDetuning[7])
        p7On[i,j] = (populations[7])
        p7Off[i,j] = (populationsDetuning[7])

fig, ax = plot_custom(
   PRQuantumWidthFull*0.4, 0.45*PRQuantumWidthFull, 
   r"$I_{\text{P}}$ (MW/cm$^2$)", 
   r'$I_{\text{S}}$ (MW/cm$^2$)',
   labelPadX = 3, labelPadY = 8, commonY = False, commonX = True, wSpaceSetting=0,
   spineColor = 'black', tickColor = 'black', textColor = 'black',
   nrow = 1, ncol = 1, fontType = 'BeraSans', axefont = labelfontsize, numsize=numfontsize,
   widthBool =False
)

X, Y = np.meshgrid(np.log10(I_green), np.log10(I_IR))
im = ax.pcolormesh(X-10, Y-10, (p7), cmap = 'hot')
# im = ax.contourf(X, Y, p7, 100)
plt.tight_layout()

cbar = plt.colorbar(im, orientation = 'horizontal', location = 'top')
cbar.set_label(r'$m_7^{\text{(on)}} - m_7^{\text{(off)}}$ (au)', labelpad=2, fontsize = labelfontsize)
yticks = ax.get_yticks()
xticks = ax.get_xticks()
ax.set_yticklabels([rf"$10^{{{val:.0f}}}$" for val in yticks])
ax.set_xticklabels([rf"$10^{{{val:.0f}}}$" for val in xticks])
ticks = [0, np.max(p7)]
cbar.set_ticks(ticks)
cbar.set_ticklabels([rf"${val:.3f}$" for val in ticks], fontsize = numfontsize)

# fig.savefig(
#     "./figures/" + "NVPopulationContrast.png", dpi=300, bbox_inches="tight",transparent=True
# )
plt.show()

#%%
Ip_range = np.logspace(5, 12, 100)
Is_range = np.logspace(5, 20, 100)
# sweep_pump(Ip_range, I_s = 0)
sweep_probe(Is_range, I_p = 1e7, xopt = 0)
sweep_probe(Is_range, I_p = 1e7, xopt = 0, omega = 0)