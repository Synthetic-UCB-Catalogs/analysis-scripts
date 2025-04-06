# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import rapid_code_load_T0 as load
from get_mass_norm import get_mass_norm
import formation_channels as fc
import numpy as np
import matplotlib.pyplot as plt
import tqdm

dat_path = 'data/IC_variations/'
variations = ['fiducial', 'ecc_thermal', 'ecc_uniform', 'm2_min_05', 'porb_log_uniform']
codes = ['ComBinE', 'COMPAS']#, 'COSMIC']


# +
def get_stats_and_dat(variation, code):
    d, h = load.load_T0_data(f'{dat_path}/{variation}/{code}_T0.hdf5')

    # get evol states    
    ZAMS, WDMS, DWD = fc.select_evolutionary_states(d)

    # get initial condition mass normalization
    m_simulated_initial = get_mass_norm(variation)

    # Calculate stats on WDMS/DWD formation per unit Msun
    n_WDMS = len(WDMS)/m_simulated_initial
    n_DWD = len(DWD)/m_simulated_initial

    return n_WDMS, n_DWD, WDMS, DWD

def chirp_mass(m1, m2):
    return (m1*m2)**(3/5) / (m1 + m2)**(1/5)


# -

stats_WDMS = []
stats_DWD = []
WDMS = []
DWD = []
v = variations[0]
for c in tqdm.tqdm(codes):
    s_WDMS = []
    s_DWD = []
    W = []
    D = []
    for v in variations:
        n_WDMS, n_DWD, wdms, dwd = get_stats_and_dat(v, c)
        s_WDMS.append(n_WDMS)
        s_DWD.append(n_DWD)
        W.append(wdms)
        D.append(dwd)
    stats_WDMS.append(s_WDMS)
    stats_DWD.append(s_DWD)
    WDMS.append(W)
    DWD.append(D)
    

for nD, nW, c in zip(stats_DWD, stats_WDMS, codes):
    n_fiducial = nD[0]
    print(nD)
    if c == 'ComBinE': 
        s=20
    else:
        s=10
    plt.scatter(range(len(nD)), nD/n_fiducial, label=c, s=s)
plt.xticks(ticks=range(len(nD)), labels=variations, rotation=45)
#plt.yscale('log')
plt.ylabel('# of DWDs/Msun relative to fiducial')
plt.legend(loc='upper left')
plt.show()

for nD, nW, c in zip(stats_DWD, stats_WDMS, codes):
    n_fiducial = nD[0]
    print(nD)
    if c == 'ComBinE': 
        s=20
    else:
        s=10
    plt.plot(range(len(nD)), nD, label=c, marker='o')
plt.xticks(ticks=range(len(nD)), labels=variations, rotation=45)
#plt.yscale('log')
plt.ylabel('# of DWDs/Msun')
plt.legend()
plt.show()

fig, axs = plt.subplots(1, len(codes), figsize=(4*len(codes),3.5))
bins = np.linspace(-2.2, 8.2, 20)
for ii, (D, W, c) in enumerate(zip(DWD, WDMS, codes)):
    for d, l in zip(D, variations):
        axs[ii].hist(np.log10(d.semiMajor), histtype='step', label=l, bins=bins)
    axs[ii].legend(loc='upper left', prop={'size': 8})
    axs[ii].set_xlim(-3, 9)
    axs[ii].set_xlabel(r'log$_{10}$(a/$R_{\odot}$)')
    axs[ii].set_title(c)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(codes), figsize=(4*len(codes),3.5))
bins = np.linspace(0, 1.5, 20)
for ii, (D, W, c) in enumerate(zip(DWD, WDMS, codes)):
    for d, l in zip(D, variations):
        axs[ii].hist(d.mass1, histtype='step', label=l)#, bins=bins)
    axs[ii].legend(loc='upper right', prop={'size': 8})
    axs[ii].set_xlim(0, 1.5)
    axs[ii].set_xlabel(r'M$_1$ [$M_{\odot}$]')
    axs[ii].set_title(c)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(codes), figsize=(4*len(codes),3.5))
bins = np.linspace(0, 1.5, 20)
for ii, (D, W, c) in enumerate(zip(DWD, WDMS, codes)):
    for d, l in zip(D, variations):
        axs[ii].hist(d.mass2, histtype='step', label=l)#, bins=bins)
    axs[ii].legend(loc='upper right', prop={'size': 8})
    axs[ii].set_xlim(0, 1.5)
    axs[ii].set_xlabel(r'M$_2$ [$M_{\odot}$]')
    axs[ii].set_title(c)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(codes), figsize=(4*len(codes),3.5))
bins = np.linspace(0, 1.5, 20)
for ii, (D, W, c) in enumerate(zip(DWD, WDMS, codes)):
    for d, l in zip(D, variations):
        axs[ii].hist(chirp_mass(d.mass1.values, d.mass2.values), histtype='step', label=l)#, bins=bins)
    axs[ii].legend(loc='upper right', prop={'size': 8})
    axs[ii].set_xlim(0, 1.5)
    axs[ii].set_xlabel(r'M$_c$ [$M_{\odot}$]')
    axs[ii].set_title(c)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(codes), figsize=(4*len(codes),3.5))
bins = np.linspace(0, 1.5, 20)
for ii, (D, W, c) in enumerate(zip(DWD, WDMS, codes)):
    for d, l in zip(D, variations):
        axs[ii].scatter(d.mass1.values[::10], d.mass2.values[::10], label=l, s=1)#, bins=bins)
    axs[ii].legend(loc='upper right', prop={'size': 8})
    axs[ii].set_xlim(0, 1.5)
    axs[ii].set_xlabel(r'M$_1$ [$M_{\odot}$]')
    axs[ii].set_title(c)
axs[0].set_ylabel(r'M$_2$ [$M_{\odot}$]')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, len(codes), figsize=(4*len(codes),3.5))
bins = np.linspace(0, 1.5, 20)
for ii, (D, W, c) in enumerate(zip(DWD, WDMS, codes)):
    for d, l in zip(D, variations):
        axs[ii].scatter(d.semiMajor[::10], chirp_mass(d.mass1.values[::10], d.mass2.values[::10]), label=l, s=1)#, bins=bins)
    axs[ii].legend(loc='upper right', prop={'size': 8})
    axs[ii].set_xlim(0, 1.5)
    axs[ii].set_xlabel(r'log$_{10}$(a/$R_{\odot}$)')
    axs[ii].set_title(c)
axs[0].set_ylabel(r'M$_c$ [$M_{\odot}$]')
plt.tight_layout()
plt.show()


