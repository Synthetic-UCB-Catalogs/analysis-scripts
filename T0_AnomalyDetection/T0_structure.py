# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# In this notebook we look at the statistics of different parameters in one of the T0 outputs. *Aim*: to get an idea of their ranges, so that we better normalize the data before training a NN.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner

import os
import shutil

MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rcdefaults()

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font',**{'family':'serif','serif':['Times']})
#plt.rc('text', usetex=True)


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %matplotlib inline

# %%
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

#fig.tight_layout()
#fig.savefig('test.pdf')

# %%
filename = './SEVN/MIST/setA/Z0.02/sevn_mist'

with open(filename, 'r') as f:
    for k,line in enumerate(f):
        if k>4:
            break
        print(line)

# %% [markdown]
# - load data;
# - replace empty entries (= 0 or more spaces) with $(-2)$ for missing values;
# - replace NaN entries with $(-1)$;
#
# The replacements are consistent with the BinCodex convention.

# %%
df = pd.read_csv(filename, skiprows=2, index_col=False)

# # (-1) is NaN and (-2) is missing which should be encoded as empty strings ''
# df.replace('^\s*$', -2., inplace=True, regex=True)
# df.fillna(-1., inplace=True)

df.head()

# %% [markdown]
# The number of IDs and UIDs must be the same in accordance with BinCodex.

# %%
unique_IDs = df.ID.unique()
unique_UIDs = df.UID.unique()

print(
    '# IDs: {:d}\n# unique IDs: {:d}'.format(
        len(unique_IDs), len(unique_UIDs)
    )
)

# %% [markdown]
# For this particular dataset there are systems for which one UID corresponds to more than one ID. It looks like the truly unique identifies for this dataset is ID rather than UID.

# %%
k = 0
systems = {}

for unique_id in unique_UIDs:
    if k > 9:
        break
    ids = df[df.UID == unique_id].ID.unique()
    if len(ids) == 1:
        continue
    systems[str(unique_id)] = ids
    k += 1

print('The first {:d} systems with 2+ IDs for one UID\n'.format(k))
print('UID:\t\t IDs:')
for uid,ids in systems.items():
    print('{}\t'.format(uid), *ids)


# %%
log_cols = [
    'mass1', 'mass2', 'radius1', 'radius2', 'semiMajor', 'Teff1', 'Teff2', 'massHecore1', 'massHecore2', 'eccentricity' 
]

linear_cols = [
    'massHecore1', 'massHecore2', 'eccentricity'
]

eps = 1e-16

for col in log_cols:

    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
    
    ax.hist(np.log10(df[col] + eps), bins=100, density=True)
    
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
    
    ax.set_xlabel(col)
    ax.set_ylabel('counts')
    
    fig.tight_layout()

for col in linear_cols:

    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

    ax.hist(df[col], bins=100, density=True)
    
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
    
    ax.set_xlabel(col)
    ax.set_ylabel('counts')
    
    fig.tight_layout()

# %%
