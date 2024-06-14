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
# -




# + colab={"base_uri": "https://localhost:8080/"} id="q1cShMsXHZ9n" executionInfo={"status": "ok", "timestamp": 1716398731600, "user_tz": 240, "elapsed": 19009, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="9735f1f2-99ee-4bb2-b38b-a4341eb77bfc"
from rapid_code_load_T0 import load_BSE_data, load_COMPAS_data, load_COSMIC_data, load_SeBa_data

# + id="gCM5jUEPHdMc" executionInfo={"status": "ok", "timestamp": 1716398764160, "user_tz": 240, "elapsed": 198, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
#COSMIC = "/content/drive/MyDrive/Synthetic UCB Catalog Project/code comparison/Code Comparison Steps/Binary Comparison/Pilot runs/COSMIC/basic.h5"
COSMIC = 'data/basic.h5'


# + id="x8tGcq__Hurt" executionInfo={"status": "ok", "timestamp": 1716398821262, "user_tz": 240, "elapsed": 53936, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
d, s_header = load_COSMIC_data(COSMIC, metallicity=0.02)

# + id="H3IKg77vlbJk" executionInfo={"status": "ok", "timestamp": 1716398842489, "user_tz": 240, "elapsed": 7052, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
ZAMS, WDMS, DWD = fc.select_ZAMS_WDMS_DWD(d=d)

# + colab={"base_uri": "https://localhost:8080/"} id="fRC_sgSqfogc" executionInfo={"status": "ok", "timestamp": 1716398842489, "user_tz": 240, "elapsed": 15, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="515486e3-4d82-412e-b4b7-955db5eb1eca"
print(len(ZAMS), len(WDMS), len(DWD), len(WDMS) + len(DWD), len(d.ID.unique()))
# -

ZAMS = []
WDMS = []
DWD = []


# + id="6i_6qcGDpyq9"
#dat = pd.read_csv(seba, sep="\s+",
#        names=["UID", "SID", "mass_transfer_type", "time", "semiMajor", "eccentricity",
#               "stellar_indentity1", "star_type1", "mass1", "radius1", "Teff1", "massHeCore1",
#               "stellar_indentity2", "star_type2", "mass2", "radius2", "Teff2", "massHeCore2"])

bpp = pd.read_hdf(COSMIC, key='bpp')
# -

RLO_1 = d.loc[d.event == 31].ID.unique()
d.loc[d.ID.isin(RLO_1)]


bpp.loc[(bpp.evol_type == 7) & (bpp.RRLO_1 > 1) & (bpp.RRLO_2 < 1)][['tphys', 'kstar_1', 'kstar_1', 'evol_type']]

for r in RLO_1[:10]:
    print(bpp.loc[bpp.bin_num==r][['mass_1', 'mass_2', 'RRLO_1', 'RRLO_2', 'evol_type', 'sep']])
    print()
    print(d.loc[d.ID == r][['mass1', 'mass2', 'type1', 'type2', 'event', 'semiMajor']])


# +
def single_event_select_ID(d, event):
    single_event_select, = np.where(d.loc[d.event==event].ID.value_counts() == 1)
    
    IDs = d.loc[d.event==event].ID.value_counts().index

    return IDs[single_event_select].values


def multi_event_select_ID(d, events):
    multi_event_select, = np.where(d.loc[d.event.isin(events)].ID.value_counts() > 1)
    IDs = d.loc[d.event.isin(events)].ID.value_counts().index

    return IDs[multi_event_select].values

    
def select_channels_simple(d):

    RLO_all = d.loc[d.event.isin([31,32,511,512,513,53])]
    No_RLO = d.loc[~d.ID.isin(RLO_all.ID)].ID.unique()

    RLO_2 = RLO_all.loc[RLO_all.ID.value_counts() == 2]
    
    SMT1_CE1 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event.isin([511, 513, 53])))].ID.unique()
    SMT2_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 32) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event.isin([512, 513, 53])))].ID.unique()

    other = np.append(SMT1_CE1, SMT2_CE2)
    
    # filter out the 
    RLO_2 = RLO_2.loc[~(RLO_2.ID.isin(other))]
    
    SMT1_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    SMT1_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()
    CE1_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    CE1_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()
    
    return No_RLO, SMT1_SMT2, SMT1_CE2, CE1_SMT2, CE1_CE2, other


# -

No_RLO, SMT1_SMT2, SMT1_CE2, CE1_SMT2, CE1_CE2, other = select_channels_simple(d)


No_RLO

dat_out = [No_RLO, SMT1_SMT2, SMT1_CE2, CE1_SMT2, CE1_CE2, SMT1_CE1, SMT2_CE2]
IDs = []
for ID in dat_out:
    IDs.extend(ID)

d.loc[d.ID.isin(IDs)]

for r in SMT1_SMT2[:10]:
    print(d.loc[d.ID == r][['mass1', 'mass2', 'type1', 'type2', 'event', 'semiMajor']])
    print(bpp.loc[bpp.bin_num==r][['tphys', 'RRLO_1', 'RRLO_2','kstar_1', 'kstar_2', 'evol_type', 'sep']])
    print()

# + colab={"base_uri": "https://localhost:8080/", "height": 332} id="Tza_sJ08wpgN" executionInfo={"status": "ok", "timestamp": 1715795203673, "user_tz": 240, "elapsed": 20, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="6add6558-b471-4ea7-df0c-c9ed513db682"
bpp.loc[bpp.bin_num == 347673][['tphys', 'kstar_1', 'kstar_2', 'porb', 'evol_type']]

# + colab={"base_uri": "https://localhost:8080/"} id="vo-37HggfokW" executionInfo={"status": "ok", "timestamp": 1715795206781, "user_tz": 240, "elapsed": 3126, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="d875eed3-2557-418d-a5bf-df36ea544fd1"
RLO1_ID = d.loc[d.event == 31].ID.unique()
RLO2_ID = d.loc[d.event == 32].ID.unique()

CE1_ID = d.loc[d.event == 511].ID.unique()
CE2_ID = d.loc[d.event == 512].ID.unique()
CEboth_ID = d.loc[d.event == 513].ID.unique()

RLO1_CE2_ID = np.intersect1d(RLO1_ID, CE2_ID)
CE1_CE2_ID = np.intersect1d(CE1_ID, CE2_ID)
RLO1_RLO2_ID = np.intersect1d(RLO1_ID, RLO2_ID)

int_ID = d.loc[d.event.isin([31, 32, 511, 512, 513, 52, 53])].ID.unique()
nonint = d.loc[~d.ID.isin(int_ID)].ID.unique()
all_ID = np.concatenate([RLO1_CE2_ID, CE1_CE2_ID, RLO1_RLO2_ID, nonint])

print(len(np.setdiff1d(DWD.ID.unique(), all_ID)))


CE2_RLO1 = np.intersect1d(CE1_ID, RLO2_ID)
print(len(np.intersect1d(CE2_RLO1, DWD.ID.unique())))
print(len(np.intersect1d(CE2_RLO1, WDMS.ID.unique())))
print(len(np.intersect1d(CE2_RLO1, WDMS.ID.unique())))

# + id="22EaVN-GWMKj"
short_nonint = d.loc[(d.ID.isin(nonint)) & (d.time == 0) & (d.mass1 < 1.5) & (d.semiMajor < 200)].ID

# + colab={"base_uri": "https://localhost:8080/"} id="s7o5nOhjlXhY" executionInfo={"status": "ok", "timestamp": 1715795347042, "user_tz": 240, "elapsed": 727, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="3555cd79-4086-4774-fc3f-b7575a410ab7"
for ii in short_nonint[:10]:
  print(d.loc[d.ID == ii])

# + id="dBEyui9sznN3" executionInfo={"status": "ok", "timestamp": 1715796054097, "user_tz": 240, "elapsed": 3302, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} colab={"base_uri": "https://localhost:8080/", "height": 434} outputId="cddcc954-e719-43a9-b1c7-35fa029d2bc3"
plt.scatter(d.loc[(d.ID.isin(nonint)) & (d.ID.isin(DWD.ID.unique()))].groupby('ID').first().semiMajor, d.loc[(d.ID.isin(nonint)) & (d.ID.isin(DWD.ID.unique()))].groupby('ID').first().mass1,
            c=d.loc[(d.ID.isin(nonint)) & (d.ID.isin(DWD.ID.unique()))].groupby('ID').first().mass2, norm=mpl.colors.LogNorm(), s=2)
plt.colorbar()
plt.xscale('log')
plt.yscale('log')

# + id="Up4k_ExCzZoA"
d = d.loc[~((d.event.shift() == 41) & (d.event == 31) & (d.time - d.time.shift() < 0.2))]
d = d.loc[~((d.event.shift() == 42) & (d.event == 32) & (d.time - d.time.shift() < 0.2))]


# + colab={"base_uri": "https://localhost:8080/"} id="UUx1lhjL8HCb" executionInfo={"status": "ok", "timestamp": 1715190663087, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="d94791c9-aa28-49f1-d050-822c83a51c17"
d.loc[d.event == 32].groupby('ID').event.value_counts().sort_values()

# + id="0ryiobfs8x8f"
pd.set_option('display.max_rows', 500)
bpp = pd.read_hdf(COSMIC, key='bpp')

# + id="e-vDhYS042S6" executionInfo={"status": "ok", "timestamp": 1715192010554, "user_tz": 240, "elapsed": 340, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="80c49e10-be93-437a-e1e5-f8f6588203bb"
bpp.loc[bpp.bin_num == 407710][['tphys', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'porb', 'evol_type']]
#d.loc[d.ID == 407710]

# + colab={"base_uri": "https://localhost:8080/"} id="2rIILXCPfonv" executionInfo={"status": "ok", "timestamp": 1712262234191, "user_tz": 240, "elapsed": 264, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="46aa8a05-7d44-4fb1-d2de-6a35027c4e92"
for id in CE1_ID[:10]:
  print(d.loc[d.ID==id][['ID', 'time', 'event', 'semiMajor', 'radius1', 'RRLO_1', 'radius2', 'RRLO_2']])

# + id="c6bsRd5lforG"



# + id="Ioh3G3-ofouQ"



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

