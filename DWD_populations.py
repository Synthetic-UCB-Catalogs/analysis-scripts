# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# %matplotlib inline

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import SubplotSpec
import h5py as h5
import tarfile
import cmasher as cmr
from collections import Counter

from rapid_code_load_T0 import load_T0_data
import formation_channels as fc
# -



fname = 'data/Pilot_runs_T0_data/basic_BSE/COMPAS_T0.hdf5'
d, header = load_T0_data(fname)

# +

ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=d)

# +
m1 = DWD['mass1']
m2 = DWD['mass2']

def chirp_mass(m1, m2):
    return np.power(m1*m2, 3/5)/np.power(m1+m2, 1/5)

fig, ax = plt.subplots()

ax.hist(chirp_mass(m1, m2), bins=100)
ax.set_xlabel("Chirp Mass [$M_\odot$]")
print()
# -




