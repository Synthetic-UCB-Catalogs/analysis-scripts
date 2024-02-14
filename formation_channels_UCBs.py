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

from rapid_code_load_T0 import load_COSMIC_data, load_SeBa_data, load_COMPAS_data

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
pd.set_option('display.max_rows', 500)
pd.set_option('display.min_rows', 100)




# +
# Want to look for systems that contain
# 11 or 12
# with a stype of the component at 21, 22, or 23...
# can I trust that the stellar type is that of the end of the timestep?

# Need to decide if I want to write up some array arithmetic for it, or just iterate per seed
# The second is easier to write, but slower. I could speed it up potentially by removing many bad seeds, e.g those that don't form DWDs


# -


datapath = 'COMPAS_Output.h5'
df = load_COMPAS_data(datapath)
np.unique(df.loc[:,'event'])


df

df['DWD_Channel'] = np.zeros(df.shape[0])
df

# +
# First, want to mask for things that become DWDs

wd_vals = np.array([21, 22, 23])
st1 = df.loc[:, 'stellarType1']
st2 = df.loc[:, 'stellarType2']
id = df.loc[:, 'ID']

ids_dwd = np.unique(id[st1.isin(wd_vals) & st2.isin(wd_vals)])
mask_dwd = id.isin(ids_dwd)

# This reduces the size of the df to work with
df = df[mask_dwd]


# +
# Want to get the last timestep for a DWD, then grab the semimaj and ecc
ids = df.loc[:, 'ID']
time = df.loc[:, 'time']
dwds = df.loc[:, 'DWD_Channel'] 
a = df.loc[:, 'semiMajor']
e = df.loc[:, 'eccentricity']
#print(ids)

mask_new_binary = ids.diff() != 0
a_masked = a.mask(mask_new_binary)
e_masked = a.mask(mask_new_binary)
print(a_masked*e_masked)
#dwds = (dwds).where(mask_new_binary, a*(1-e)) # make it the peripasis for now, this is not going to be explicitly useful until later


# -

x = pd.Series([1, 1, 2, 2, 3, 4, 5])
print(x.size)
xd = x.diff() != 0
print(xd.size)
xd






