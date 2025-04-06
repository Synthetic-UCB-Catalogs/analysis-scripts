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

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        if k>10:
            break
        print(line)

# %%
df = pd.read_csv(filename, skiprows=2, index_col=False)

# reduce the ranges of "time" and "Teff1"/"Teff2": convert to Gyr and kiloK, respectively;
conversion_facs = {
    'time': 1000.,
    'Teff1': 1000.,
    'Teff2': 1000.
}

for col,fac in conversion_facs.items():
    df[col] /= fac

# (-1) is NaN and (-2) is missing which should be encoded as empty strings ''
df.replace('^\s*$', -2., inplace=True, regex=True)
df.fillna(-1., inplace=True)

df.head()

# %%
# 1. one-hot encoding for the categorical variables [event,type1,type2]
# 2. convert time to Gyr
columns = ['event', 'type1', 'type2']

for col in columns:
    df[col] = df[col].astype(str)

df_onehot = pd.get_dummies(df, columns=columns, dtype=float)

df_onehot.head()

# %%
unique_IDs = df_onehot.ID.unique()
unique_UIDs = df_onehot.UID.unique()

print(
    '# IDs: {:d}\n# unique IDs: {:d}'.format(
        len(unique_IDs), len(unique_UIDs)
    )
)

# %%
k = 0
systems = {}

for unique_id in unique_UIDs:
    if k > 9:
        break
    ids = df_onehot[df_onehot.UID == unique_id].ID.unique()
    if len(ids) == 1:
        continue
    systems[str(unique_id)] = ids
    k += 1

print('The first {:d} systems with 2+ IDs for one UID\n'.format(k))
print('UID:\t\t IDs')
for uid,ids in systems.items():
    print('{}\t'.format(uid), *ids)

# %%
seq_len = 5

rng = np.random.default_rng()

random_id = rng.choice(unique_IDs)

(df[df.ID == random_id]).sort_values('time')

# %%
