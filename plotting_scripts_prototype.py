# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Plot settings

# %% [markdown]
# #### Font settings

# %% executionInfo={"elapsed": 829, "status": "ok", "timestamp": 1702490200188, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="u-iEr9RRL9ys"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, Normalize

import os


MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rcdefaults()

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update()


# %matplotlib inline

# %% [markdown]
# #### Plot formatting template

# %% executionInfo={"elapsed": 131, "status": "ok", "timestamp": 1702490201364, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="WLSvHDxQNAHG"
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

#fig.tight_layout()
#fig.savefig('test.pdf')

def log_bins(arr, N=50):
    
    return np.logspace(np.log10(arr.min()), np.log10(arr.max()), N)


# %% [markdown] id="Ng_lEt9xOfMJ"
# ### Notebook: *merger_time_timeto_LISA* (Petra)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2912, "status": "ok", "timestamp": 1702490218666, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="45AaDE7ROxB9" outputId="0782d412-422b-48bd-a12b-96a6a0ad628e"
# Uncomment if launched in the Colab environment

# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)

# %% executionInfo={"elapsed": 8714, "status": "ok", "timestamp": 1702490228604, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="9XfU372AOyj_"
# Uncomment if launched in the Colab environment

# #!pip install -q legwork
# #!sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended

# %% executionInfo={"elapsed": 2103, "status": "ok", "timestamp": 1702490248801, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="fDOHi3V3PQRW"
from legwork import evol, source, utils, visualisation
import astropy.units as u

import astropy.constants as aconst

# G=c=1: everything is measured in seconds or inverse seconds; R -> R/c, M -> G*M/c^3
# reference values: Msun = 5e-6 seconds, AU = 500 seconds


Msun = aconst.G*aconst.M_sun/aconst.c**3
Msun = Msun.cgs.value  # s

AU = aconst.au/aconst.c   
AU = AU.cgs.value   # s

Rsun = (1*u.R_sun) / aconst.c
Rsun = Rsun.cgs.value

year = (1*u.year).cgs.value
day = 24*3600

# %% executionInfo={"elapsed": 9108, "status": "ok", "timestamp": 1702490260389, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="64k5uKxq25hS"


local_path = './data'
colab_path = '''
/content/drive/MyDrive/LISA/Synthetic UCB Catalog Project/code comparison/Code Comparison Steps/Binary Comparison
'''

COSMIC = pd.read_hdf(
    os.path.join(local_path, 'Initial Conditions Variations/COSMIC/fiducial.h5'), 
    key="bpp"
)


print(COSMIC.columns)
COSMIC.head()

# %% executionInfo={"elapsed": 1435, "status": "ok", "timestamp": 1702490265777, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="UPBVitAIaGo3"
COSMIC = COSMIC.rename(
    columns={"tphys": "time", "mass_1": "mass1", "mass_2": "mass2",
             "massc_1": "massHecore1", "massc_2": "massHecore2",
             "porb": "period", "sep": "semiMajor",
             "teff_1": "Teff1", "teff_2": "Teff2",
             "rad_1": "radius1", "rad_2": "radius2",
             "bin_num": "UID"})

# %% [markdown]
# #### Consistency check: Period are measured in days
# Compute the periods directly and compare to the `period` column

# %%
# delete values<=0 rows
COSMIC = COSMIC[(COSMIC['semiMajor'] > 0)]
COSMIC = COSMIC[(COSMIC['mass1'] != 0)]
COSMIC = COSMIC[(COSMIC['mass2'] != 0)]

ratio_periods = 2*np.pi*np.sqrt(
    (COSMIC.semiMajor.values*Rsun)**3 / (COSMIC.mass1.values + COSMIC.mass2.values) / Msun
) / day / COSMIC.period

ratio_periods.values.min(), ratio_periods.values.max()

# %% executionInfo={"elapsed": 3956, "status": "ok", "timestamp": 1702490270764, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="0rJTkahrabch"
fmin = 1e-4
fmax = 1.
Pmax,Pmin = 2/fmin, 2/fmax
COSMIC = COSMIC[(COSMIC['period'] >= Pmin/day) & (COSMIC['period'] <= Pmax/day)]

f = 2/COSMIC.period/day
len(f)

# %% executionInfo={"elapsed": 132, "status": "ok", "timestamp": 1702490274714, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="9ad2DVtBbHRD"
# calculate time to merger using legwork
m_1 = COSMIC.mass1.values*u.Msun
m_2 = COSMIC.mass2.values*u.Msun
a = COSMIC.semiMajor.values*u.Rsun
time = COSMIC.time.values

# %% [markdown]
# ####  `legwork` gives the answer in *Gyr*

# %% executionInfo={"elapsed": 267, "status": "ok", "timestamp": 1702490278677, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="pYZafTUuchkG"
# compute with individual masses and frequency
t_merge = evol.get_t_merge_circ( m_1=m_1, m_2=m_2, a_i=a)

t_merge[0]

# %% executionInfo={"elapsed": 120, "status": "ok", "timestamp": 1702490279617, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="2pn3wsXSMZoL"
#np.any(t_merge<=4)
time_merge = 1e+9 * t_merge.value   # yr
cond_merged = time_merge <= 12

T_Hubble = 13.7 * 1e+9    # Hubble time [yr]

# %% colab={"base_uri": "https://localhost:8080/", "height": 593} executionInfo={"elapsed": 3862, "status": "ok", "timestamp": 1702490416715, "user": {"displayName": "Vladimir Strokov", "userId": "04489751532780417747"}, "user_tz": 300} id="R5dIrltKOGre" outputId="6192b7f7-0ee2-4cf0-c35f-a33682d33160"
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(8,6))

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xscale('log')
ax.set_yscale('log')


h = ax.hist2d(
    f, time_merge, 
    bins=[log_bins(f), log_bins(time_merge)], 
    cmap='viridis', norm=LogNorm()
)


ax.scatter(
    f[cond_merged], time_merge[cond_merged], 
    color='red', marker='o', label='merge within 12 yrs'
)

xx = np.linspace(*ax.get_xlim(), 100)
yy = np.full_like(xx, T_Hubble)
ax.loglog(xx,yy, c='black', ls='dashed')

ax.set_xticks(np.logspace(-4,0,5))


# Add color bar and labels
ax.set_xlabel('frequency (Hz)', fontsize=BIGGER_SIZE)
ax.set_ylabel('time till merger (yr)', fontsize=BIGGER_SIZE)
ax.legend(prop={'size':BIGGER_SIZE})

fig.colorbar(h[3], ax=ax)


fig.tight_layout()

# %%
