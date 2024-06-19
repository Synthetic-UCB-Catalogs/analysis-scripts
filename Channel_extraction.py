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

# + id="453IjyqHHSa3" executionInfo={"status": "ok", "timestamp": 1716398576819, "user_tz": 240, "elapsed": 1020, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import SubplotSpec
import h5py as h5
import tarfile

import formation_channels as fc
from rapid_code_load_T0 import load_BSE_data, load_COMPAS_data, load_COSMIC_data, load_SeBa_data
# -
COSMIC = 'data/basic.h5'
d, s_header = load_COSMIC_data(COSMIC, metallicity=0.02)

ZAMS, WDMS, DWD = fc.select_ZAMS_WDMS_DWD(d=d)
No_RLO, SMT1_SMT2, SMT1_CE2, CE1_SMT2, CE1_CE2, other = fc.select_channels_simple(d=d)

ZAMS['porb'] = ((ZAMS.semiMajor / 215.032)**3 / (ZAMS.mass1+ZAMS.mass2))**0.5 * 365.25

other_ID = []
for key in other.keys():
    print(key)
    other_ID.extend(other[key])

plt.hist(d.groupby('ID', as_index=False).nth(0).mass2/d.groupby('ID', as_index=False).nth(0).mass1)

init_01 = ZAMS.loc[np.isclose(ZAMS.mass2/ZAMS.mass1,0.1, 0.01)]
init_05 = ZAMS.loc[np.isclose(ZAMS.mass2/ZAMS.mass1,0.49, 0.01)]
init_05 = init_05.loc[init_05.mass1.isin(init_01.mass1)]
init_09 = ZAMS.loc[np.isclose(ZAMS.mass2/ZAMS.mass1,0.9, 0.01)]
init_09 = init_09.loc[init_09.mass1.isin(init_01.mass1)]




init_09.mass2/init_09.mass1

init_01

plt.scatter(init_01.porb, init_01.mass1)
plt.xscale('log')
plt.yscale('log')

plt.scatter(init_05.porb, init_05.mass1)
plt.xscale('log')
plt.yscale('log')

plt.scatter(init_09.porb, init_09.mass1)
plt.xscale('log')
plt.yscale('log')

np.sort(init_01.mass2/init_01.mass1)


ZAMS_ID_all = ZAMS.ID.unique()

id_not_hit = np.setxor1d(ZAMS_ID_all, id_select_all)

len(id_not_hit), len(id_select_all)

# +
import cmasher as cmr

# Take 6 colors from rainforest in [0.15, 0.85] range in HEX
colors = cmr.take_cmap_colors('cmr.rainforest', 6, cmap_range=(0.15, 0.85), return_fmt='hex')
# -


init_01.loc[init_01.ID.isin(other_ID)]

# + colab={"base_uri": "https://localhost:8080/"} id="q1cShMsXHZ9n" executionInfo={"status": "ok", "timestamp": 1716398731600, "user_tz": 240, "elapsed": 19009, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="9735f1f2-99ee-4bb2-b38b-a4341eb77bfc"
ID_list = [No_RLO, SMT1_SMT2, SMT1_CE2, CE1_SMT2, CE1_CE2, other_ID]
label_list = ['No RLO', 'SMT1, SMT2', 'SMT1, CE2', 'CE1, SMT2', 'CE1, CE2', 'other']

for id, c, l, ii in zip(ID_list, colors, label_list, range(len(colors))):
    ZAMS_select = init_09.loc[(init_09.ID.isin(id))]
    if len(ZAMS_select) > 0:
        print(len(ZAMS_select), l)
        plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=10, label=l, zorder=100 - (1+ii)*5)
        
        
    else:
        print(0, l)
plt.xscale('log')
plt.legend(loc=(0.0, 1.01), ncol=4)
plt.yscale('log')
plt.xlim(min(init_05.porb)-0.1, max(init_05.porb)+1e3)
plt.ylim(min(init_05.mass1)-0.1, max(init_05.mass1)+0.2)
plt.show()
# -

plt.scatter(ZAMS.porb, ZAMS.mass1)
plt.xscale('log')

# + id="BAXF2WpCM5PF"
seba_dwd = seba_full.loc[(seba_full.type1.isin([21,22,23]) & seba_full.type2.isin([21,22,23]) & seba_full.semiMajor > 0.0)]

seba_int = seba_full.loc[seba_full.event.isin([30,31,32,33,40,41,42,43,510,511,512,513,50,52,53,54])].ID.unique()

seba_dwd_nonint = seba_dwd.loc[~seba_dwd.ID.isin(seba_int)].groupby('ID', as_index=False).first()
seba_dwd_int = seba_dwd.loc[seba_dwd.ID.isin(seba_int)].groupby('ID', as_index=False).first()

# + colab={"base_uri": "https://localhost:8080/", "height": 73} id="-xQXYvn7NhRi" executionInfo={"status": "ok", "timestamp": 1710936847640, "user_tz": 240, "elapsed": 152, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="9521a301-711a-46f0-a6b9-cbf485a89e64"
seba_dwd_int.loc[seba_dwd_int.semiMajor == 0.0]

# + colab={"base_uri": "https://localhost:8080/", "height": 448} id="LXA6vvPxNCrB" executionInfo={"status": "ok", "timestamp": 1710936849099, "user_tz": 240, "elapsed": 436, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="a765edd6-9431-4643-8f73-a4974e2872e4"
plt.hist(np.log10(seba_dwd_nonint.semiMajor), histtype='step', label='non interacting', density=True, bins=np.linspace(-1.5, 4.5, 25))
plt.hist(np.log10(seba_dwd_int.semiMajor), histtype='step', label='interacting', density=True, bins=np.linspace(-1.5, 4.5, 25))
plt.legend()

# + id="VrhDe4UxOeah"
cosmic = "/content/drive/MyDrive/Synthetic UCB Catalog Project/code comparison/Code Comparison Steps/Binary Comparison/Pilot runs/COSMIC/basic.h5"


# + id="p6IR8LwNOgmk"
#read in files, keeping only dwd
#cosmic

cosmic_full, c_header = load_COSMIC_data(cosmic, metallicity=0.2)

cosmic_dwd = cosmic_full.loc[(cosmic_full.type1.isin([21,22,23]) & cosmic_full.type2.isin([21,22,23]) & cosmic_full.semiMajor > 0.0)]

cosmic_int = cosmic_full.loc[cosmic_full.event.isin([30,31,32,33,40,41,42,43,510,511,512,513,50,52,53,54])].ID.unique()

cosmic_dwd_nonint = cosmic_dwd.loc[~cosmic_dwd.ID.isin(cosmic_int)].groupby('ID', as_index=False).first()

cosmic_dwd_int = cosmic_dwd.loc[cosmic_dwd.ID.isin(cosmic_int)].groupby('ID', as_index=False).first()


# + id="HGs68JZVQzyO"
cosmic_full.to_hdf("/content/drive/MyDrive/Synthetic UCB Catalog Project/code comparison/Code Comparison Steps/Binary Comparison/Pilot runs/COSMIC/COSMIC_BinCodex.h5", key="T0")
c_header.to_hdf("/content/drive/MyDrive/Synthetic UCB Catalog Project/code comparison/Code Comparison Steps/Binary Comparison/Pilot runs/COSMIC/COSMIC_BinCodex.h5", key="header")

# + colab={"base_uri": "https://localhost:8080/", "height": 329} id="2P0OzVMMOpYq" executionInfo={"status": "ok", "timestamp": 1710273434053, "user_tz": 240, "elapsed": 814, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="746cb7b2-93ba-4277-8aed-4372a2c20c9a"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

ax1.hist(np.log10(seba_dwd_nonint.semiMajor), histtype='step', label='non interacting', density=True, bins=np.linspace(-1.5, 4.5, 25))
ax1.hist(np.log10(seba_dwd_int.semiMajor), histtype='step', label='interacting', density=True, bins=np.linspace(-1.5, 4.5, 25))
ax1.legend()

ax2.hist(np.log10(cosmic_dwd_nonint.semiMajor), histtype='step', label='non interacting', density=True, bins=np.linspace(-1.5, 4.5, 25))
ax2.hist(np.log10(cosmic_dwd_int.semiMajor), histtype='step', label='interacting', density=True, bins=np.linspace(-1.5, 4.5, 25))
ax2.legend()

ax1.set_xlabel('log(a/Rsun)')
ax2.set_xlabel('log(a/Rsun)')

# + id="jfMEaRnFPM37"
bpp = pd.read_hdf('data/basic.h5', key='bpp')
# -

bpp.loc[(bpp.kstar_1.isin([10,11,12])) & (bpp.kstar_2.isin([10,11,12]))][['mass_1','mass_2', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']]

CE = bpp.loc[bpp.evol_type == 7][['mass_1','mass_2', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']]

SMT = bpp.loc[(bpp.evol_type == 3)]

# +
#SMT = SMT.loc[~SMT.bin_num.isin(CE.bin_num)]
# -

for bn in SMT.bin_num.unique()[:5]:
    print(bpp.loc[bpp.bin_num == bn][['mass_1','mass_2', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']])


