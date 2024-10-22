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
import cmasher as cmr

import formation_channels as fc
from rapid_code_load_T0 import convert_BSE_data_to_T0, convert_COMPAS_data_to_T0, convert_COSMIC_data_to_T0, convert_SeBa_data_to_T0, load_T0_data
# -
COSMIC = 'data/basic.h5'
c = convert_COSMIC_data_to_T0(COSMIC, metallicity=0.02)

# +
#METISSE = 'data/basic_METISSE.h5'
#m = convert_COSMIC_data_to_T0(METISSE, metallicity=0.02)
# -

COMPAS = 'data/COMPAS_pilot.h5'
co = convert_COMPAS_data_to_T0(COMPAS)

co['porb'] = ((co.semiMajor / 215.032)**3 / (co.mass1+co.mass2))**0.5 * 365.25


# +
#SEVN_mist = 'data/T0_format_pilot/MIST/setA/Z0.02/sevn_mist'
#sm, sm_header = load_T0_data(SEVN_mist, code='SEVN', metallicity=0.02)
# -

def get_first_RLO_figure(d, q=0.49, savefig=None):
    ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=d)
    #channels = fc.select_channels(d=d)

    ZAMS['porb'] = ((ZAMS.semiMajor / 215.032)**3 / (ZAMS.mass1+ZAMS.mass2))**0.5 * 365.25
    WDMS['porb'] = ((WDMS.semiMajor / 215.032)**3 / (WDMS.mass1+WDMS.mass2))**0.5 * 365.25
    DWD['porb'] = ((DWD.semiMajor / 215.032)**3 / (DWD.mass1+DWD.mass2))**0.5 * 365.25

    init_q = ZAMS.loc[(np.round(ZAMS.mass2/ZAMS.mass1, 2) == q)]

    d = d.loc[d.ID.isin(init_q.ID)]
    first_RLO = fc.first_interaction_channels(d=d)

    #check that all IDs are accounted for:
    all_IDs = d.ID.unique()
    keys = ['SMT_1', 'SMT_2', 'CE_1', 'CE_2', 'DCCE', 'merger', 'nonRLO']
    id_check = []
    for k in keys:
        id_check.extend(first_RLO[k])
    if len(np.setxor1d(all_IDs, id_check)) > 0:
        print("waning, you missed ids:", setxor1d(all_IDS, id_check))

    SMT_colors = cmr.take_cmap_colors('cmr.sapphire', 2, cmap_range=(0.4, 0.85), return_fmt='hex')
    CE_colors = cmr.take_cmap_colors('cmr.sunburst', 3, cmap_range=(0.3, 0.9), return_fmt='hex')
    other_colors = cmr.take_cmap_colors('cmr.neutral', 2, cmap_range=(0.35, 0.85), return_fmt='hex')

    keys_list = [['SMT_1', 'SMT_2'], ['CE_1', 'CE_2', 'DCCE'], ['merger', 'nonRLO']]
    colors_list = [SMT_colors, CE_colors, other_colors]
    plt.figure(figsize=(6,4.8))
    for colors, keys in zip(colors_list, keys_list):
        for c, k, ii in zip(colors, keys, range(len(colors))):
            ZAMS_select = init_q.loc[(init_q.ID.isin(first_RLO[k]))]
            
            if len(ZAMS_select) > 0:
                #if k != 'failed_CE':
                print(len(ZAMS_select), k)
                plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
                
                
            else:
                print(0, k)
    print()
    print()
    plt.xscale('log')
    plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
    plt.yscale('log')
    plt.xlim(min(init_q.porb)-0.1, max(init_q.porb))
    plt.ylim(min(init_q.mass1)-0.05, max(init_q.mass1)+0.5)
    
    plt.xlabel('orbital period [day]')
    plt.ylabel('M$_1$ [Msun]')
    if savefig != None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=100, facecolor='white')
    plt.show()

    return first_RLO

first_RLO_c_05 = get_first_RLO_figure(d=co, q=0.49, savefig='first_RLO_COMPAS_pilot_qinit05.png')

CE1_ID = first_RLO_c_05['CE_1'].values

len(CE1_ID)

plt.hist(co.loc[co.ID.isin(CE1_ID)].groupby('ID', as_index=False).first().mass2 / co.loc[co.ID.isin(CE1_ID)].groupby('ID', as_index=False).first().mass1)

CE1_weird = co.loc[(co.mass1 > 2.5) & (co.porb < 4) & (co.ID.isin(CE1_ID))]

# +

for id in CE1_weird.ID.unique()[:50]:
    print(CE1_weird.loc[CE1_weird.ID == id][['time', 'ID', 'event', 'mass1', 'mass2', 'porb', 'type1', 'type2']])
# -

co.loc[co.groupby('ID', as_index=False).first().time > 0]

first_RLO_c09 = get_first_RLO_figure(d=co, q=0.88, savefig='first_RLO_COMPAS_pilot_qinit09.png')

first_RLO_co_05.keys()

first_RLO_co_05['SMT_1']

co.columns

first_RLO_co_05 = get_first_RLO_figure(d=c, q=0.49, savefig='first_RLO_COSMIC_pilot_qinit05.png')

co.loc[(co.ID.isin(first_RLO_co_05['nonRLO']))& (co.type1.isin([21.0, 22.0, 23.0])) & (co.semiMajor < 100)][['mass1', 'mass2', 'ID']]

co.loc[co.ID == 366369]

first_RLO_co_09 = get_first_RLO_figure(d=co, q=0.88, savefig='first_RLO_COMPAS_pilot_qinit09.png')

first_RLO_m_05 = get_first_RLO_figure(d=m, q=0.49, savefig='first_RLO_METISSE_pilot_qinit05.png')

first_RLO_m_09 = get_first_RLO_figure(d=m, q=0.88, savefig='first_RLO_METISSE_pilot_qinit09.png')







# +
fig = plt.figure(figsize=(5.8,4), dpi=150)
first_RLO_keys = first_RLO.keys()
keys_list = [['SMT_1', 'SMT_2'], ['CE_1', 'CE_2', 'DCCE'], ['merger', 'nonRLO']]
colors_list = [SMT_colors, CE_colors, other_colors]
for colors, keys in zip(colors_list, keys_list):
    for c, k, ii in zip(colors, keys, range(len(colors))):
        ZAMS_select = init_05.loc[(init_05.ID.isin(first_RLO[k]))]
        
        if len(ZAMS_select) > 0:
            #if k != 'failed_CE':
            print(len(ZAMS_select), k)
            plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
            
            
        else:
            print(0, k)

#DWD_select = DWD.loc[DWD.ID.isin(init_09.ID)]
#DWD_ZAMS = init_09.loc[init_09.ID.isin(DWD.ID)]
#print(DWD_select)
#plt.scatter(DWD_ZAMS.porb, DWD_ZAMS.mass1, c='black', s=50, zorder=0)
print()
print()
plt.xscale('log')
plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
plt.yscale('log')
plt.xlim(min(init_05.porb)-0.1, max(init_05.porb))
plt.ylim(min(init_05.mass1)-0.05, max(init_05.mass1)+0.5)

plt.xlabel('orbital period [day]')
plt.ylabel('M$_1$ [Msun]')
plt.show()
#plt.savefig('cosmic_q_i_09.png', facecolor='white', dpi=150)
# -

for ID in sm.ID.unique()[::1000]:
    print(sm.loc[sm.ID == ID][['time', 'event', 'semiMajor', 'type1', 'mass1', 'type2', 'mass2']])

DWD

ZAMS['porb'] = ((ZAMS.semiMajor / 215.032)**3 / (ZAMS.mass1+ZAMS.mass2))**0.5 * 365.25
WDMS['porb'] = ((WDMS.semiMajor / 215.032)**3 / (WDMS.mass1+WDMS.mass2))**0.5 * 365.25
DWD['porb'] = ((DWD.semiMajor / 215.032)**3 / (DWD.mass1+DWD.mass2))**0.5 * 365.25

channels.keys()

# +
IDs = []
for k in channels.keys():
    IDs.extend(channels[k])

import collections
duplicates = [item for item, count in collections.Counter(IDs).items() if count > 1]
# -


duplicates

# +
#for k in channels.keys():
#    IDs = channels[k]
#    i = np.intersect1d(IDs, duplicates)
#    if len(i) > 0:
#        print(k, len(i))

# +
init_05 = ZAMS.loc[(np.round(ZAMS.mass2/ZAMS.mass1, 2) == 0.49)]

init_09 = ZAMS.loc[(np.round(ZAMS.mass2/ZAMS.mass1, 2) == 0.88) & (ZAMS.mass1.isin(init_05.mass1))] 

# -

for c in channels.keys():
    print(c, len(init_05.loc[init_05.ID.isin(channels[c])]))

# +
#for ID in channels['RLO_3_other']:
#    print(d.loc[d.ID == ID][['time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
# -

for c in channels.keys():
    print(c, len(init_09.loc[init_09.ID.isin(channels[c])]))


import cmasher as cmr



SMT_keys = ['SMT1_SMT1', 'SMT2_SMT2', 'SMT1', 'SMT2', 'SMT1_SMT2']
CE_keys = ['CE1_CE1', 'CE2_CE2', 'CE1_merge', 'CE1_survive', 'CE2', 'CE3', 'CE1_CE2']
mixed_keys_2 = ['SMT1_CE1', 'SMT2_CE2', 'CE1_SMT1', 'CE2_SMT2', 'SMT1_CE2', 'CE1_SMT2']
mixed_keys_3 = ['evCE1_SMT2', 'evCE1_CE2', 'SMT1_evCE2', 'CE1_evCE2', 'SMT1_CE2_SMT2', 'SMT1_CE2_CE2', 'SMT1_SMT1_CE2']
other_keys = ['RLO_3_other', 'RLO_4_or_more_other', 'No_RLO']

SMT_colors = cmr.take_cmap_colors('cmr.sapphire', len(SMT_keys), cmap_range=(0.4, 0.85), return_fmt='hex')
CE_colors = cmr.take_cmap_colors('cmr.sunburst', len(CE_keys), cmap_range=(0.3, 0.9), return_fmt='hex')
mixed_colors_2 = cmr.take_cmap_colors('cmr.nuclear', len(mixed_keys_2), cmap_range=(0.15, 0.95), return_fmt='hex')
mixed_colors_3 = cmr.take_cmap_colors('cmr.horizon', len(mixed_keys_3), cmap_range=(0.15, 0.95), return_fmt='hex')
other_colors = cmr.take_cmap_colors('cmr.neutral', len(other_keys), cmap_range=(0.35, 0.85), return_fmt='hex')

keys_list = [SMT_keys, CE_keys, mixed_keys_2, mixed_keys_3, other_keys]
colors_list = [SMT_colors, CE_colors, mixed_colors_2, mixed_colors_3, other_colors]

# + colab={"base_uri": "https://localhost:8080/"} id="q1cShMsXHZ9n" executionInfo={"status": "ok", "timestamp": 1716398731600, "user_tz": 240, "elapsed": 19009, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="9735f1f2-99ee-4bb2-b38b-a4341eb77bfc"
from rapid_code_load_T0 import load_T0_data, convert_COSMIC_data_to_T0, convert_COMPAS_data_to_T0, convert_SeBa_data_to_T0, convert_BSE_data_to_T0

# + id="gCM5jUEPHdMc" executionInfo={"status": "ok", "timestamp": 1716398764160, "user_tz": 240, "elapsed": 198, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
# Convert the data to T0 format - should only have to do this once, then comment out these lines
#COSMIC_OG = 'data/large_OG_datafiles/pilot_runs/cosmic_pilot.h5'
#convert_COSMIC_data_to_T0(COSMIC_OG, metallicity=0.02)


# + id="gCM5jUEPHdMc" executionInfo={"status": "ok", "timestamp": 1716398764160, "user_tz": 240, "elapsed": 198, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
#COMPAS_OG  = 'data/large_OG_datafiles/pilot_runs/COMPAS_pilot.h5'
#convert_COMPAS_data_to_T0(COMPAS_OG)


# + id="gCM5jUEPHdMc" executionInfo={"status": "ok", "timestamp": 1716398764160, "user_tz": 240, "elapsed": 198, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
#SeBa_OG = 'data/large_OG_datafiles/pilot_runs/Seba_BinCodex.h5'
#convert_SeBa_data_to_T0(SeBa_OG, metallicity=0.02)


# + id="gCM5jUEPHdMc" executionInfo={"status": "ok", "timestamp": 1716398764160, "user_tz": 240, "elapsed": 198, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
#BSE_OG = 'data/large_OG_datafiles/pilot_runs/bse_pilot.dat'
#convert_BSE_data_to_T0(BSE_OG, metallicity=0.02)
# -




# + id="x8tGcq__Hurt" executionInfo={"status": "ok", "timestamp": 1716398821262, "user_tz": 240, "elapsed": 53936, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
# Use the loader to load the data files
d_cosmic = load_T0_data('COSMIC_T0.hdf5')
d_compas = load_T0_data('COMPAS_T0.hdf5')

# + id="WPauZbOHQQAH" executionInfo={"status": "ok", "timestamp": 1716398821262, "user_tz": 240, "elapsed": 13, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
d = COSMIC_full

# + id="KsQAhB7YMzNj" executionInfo={"status": "ok", "timestamp": 1716398821262, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
COSMIC_full = []


# + id="ZBBAbpvae-hK" executionInfo={"status": "ok", "timestamp": 1716398834787, "user_tz": 240, "elapsed": 151, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
def select_ZAMS_WDMS_DWD(d):
    '''Selects the WDMS and DWD populations at the formation of the first and second white dwarfs

    Params
    ------
    d : `pandas.DataFrame`
        contains T0 data for binaries as specified by BinCodex

    Returns
    -------
    ZAMS : `pandas.DataFrame`
        T0 columns for Zero Age Main Sequence binaries

    WDMS : `pandas.DataFrame`
        T0 columns for WDMS binaries at the formation of the 1st WD

    DWD : `pandas.DataFrame`
        T0 columns for DWD binaries at the formation of the 2nd WD
    '''

    ZAMS = d.groupby('ID', as_index=False).first()

    WDMS1 = d.loc[((d.type1.isin([21,22,23]) & (d.type2 == 121))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    WDMS2 = d.loc[((d.type2.isin([21,22,23]) & (d.type1 == 121))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    WDMS = pd.concat([WDMS1, WDMS2])
    DWD = d.loc[(d.type1.isin([21,22,23])) & (d.type2.isin([21,22,23])) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    return ZAMS, WDMS, DWD


# + id="H3IKg77vlbJk" executionInfo={"status": "ok", "timestamp": 1716398842489, "user_tz": 240, "elapsed": 7052, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}}
ZAMS, WDMS, DWD = select_ZAMS_WDMS_DWD(d=d)

# + colab={"base_uri": "https://localhost:8080/"} id="fRC_sgSqfogc" executionInfo={"status": "ok", "timestamp": 1716398842489, "user_tz": 240, "elapsed": 15, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="515486e3-4d82-412e-b4b7-955db5eb1eca"
print(len(ZAMS), len(WDMS), len(DWD), len(WDMS) + len(DWD), len(d.ID.unique()))

# + colab={"base_uri": "https://localhost:8080/"} id="r0QJUyrVpnr9" executionInfo={"status": "ok", "timestamp": 1716398844266, "user_tz": 240, "elapsed": 427, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="ffec38a0-ab91-452f-eebd-6b682a4d6511"
RLO1_ID = d.loc[d.event == 31]

RLO1_ID.ID.value_counts()

# + id="6i_6qcGDpyq9"
#dat = pd.read_csv(seba, sep="\s+",
#        names=["UID", "SID", "mass_transfer_type", "time", "semiMajor", "eccentricity",
#               "stellar_indentity1", "star_type1", "mass1", "radius1", "Teff1", "massHeCore1",
#               "stellar_indentity2", "star_type2", "mass2", "radius2", "Teff2", "massHeCore2"])

bpp = pd.read_hdf(COSMIC, key='bpp')

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
fig = plt.figure(figsize=(5.8,4), dpi=150)
channel_keys = channels.keys()
for colors, keys in zip(colors_list, keys_list):
    for c, k, ii in zip(colors, keys, range(len(colors))):
        ZAMS_select = init_09.loc[(init_09.ID.isin(channels[k]))]
        
        if len(ZAMS_select) > 0:
            #if k != 'failed_CE':
            print(len(ZAMS_select), k)
            plt.scatter(ZAMS_select.semiMajor, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
            
            
        else:
            print(0, k)

#DWD_select = DWD.loc[DWD.ID.isin(init_09.ID)]
#DWD_ZAMS = init_09.loc[init_09.ID.isin(DWD.ID)]
#print(DWD_select)
#plt.scatter(DWD_ZAMS.porb, DWD_ZAMS.mass1, c='black', s=50, zorder=0)
print()
print()
plt.xscale('log')
plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
plt.yscale('log')

plt.xlim(min(init_05.semiMajor)-0.1, max(init_05.semiMajor))
plt.ylim(min(init_05.mass1)-0.05, max(init_05.mass1)+0.5)

plt.xlabel('semimajor axis [Rsun]')
plt.ylabel('M$_1$ [Msun]')
plt.show()
#plt.savefig('cosmic_q_i_09.png', facecolor='white', dpi=150)

# +
fig = plt.figure(figsize=(5.8,4), dpi=150)
channel_keys = channels.keys()
for colors, keys in zip(colors_list, keys_list):
    for c, k, ii in zip(colors, keys, range(len(colors))):
        ZAMS_select = init_05.loc[(init_05.ID.isin(channels[k]))]
        
        if len(ZAMS_select) > 0:
            #if k != 'failed_CE':
            print(len(ZAMS_select), k)
            plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
            
            
        else:
            print(0, k)

#DWD_select = DWD.loc[DWD.ID.isin(init_09.ID)]
#DWD_ZAMS = init_09.loc[init_09.ID.isin(DWD.ID)]
#print(DWD_select)
#plt.scatter(DWD_ZAMS.porb, DWD_ZAMS.mass1, c='black', s=50, zorder=0)
print()
print()
plt.xscale('log')
plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
plt.yscale('log')
plt.xlim(min(init_05.porb)-0.1, max(init_05.porb)+1e3)
plt.ylim(min(init_05.mass1)-0.05, max(init_05.mass1)+0.5)

plt.xlabel('orbital period [days]')
plt.ylabel('M$_1$ [Msun]')
plt.show()
#plt.savefig('cosmic_q_i_09.png', facecolor='white', dpi=150)

# + id="jfMEaRnFPM37"
bpp = pd.read_hdf('data/basic.h5', key='bpp')
# -

for id in channels['evCE1_CE2'][:10]:
    print(d.loc[d.ID == id][['time', 'mass1', 'mass2', 'type1', 'type2', 'semiMajor', 'event']])
    print(bpp.loc[bpp.bin_num == id][['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'evol_type']])

RLO_1 = RLO_all.loc[RLO_all.ID.value_counts() == 1]
CE1 = d_pre_DWD.loc[(d_pre_DWD.ID.isin(RLO_1.ID)) & (d_pre_DWD.event == 511)].ID.unique()
CE1_merge = d_pre_DWD.loc[(d_pre_DWD.ID.isin(CE1)) & (d_pre_DWD.event == 52)].ID.unique()
CE1_survive = np.setxor1d(CE1, CE1_merge)



# +
SMT1_CE2_CE2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                          (RLO_3.groupby('ID', as_index=False).nth(1).event.isin([512, 513, 53])) & 
                          (RLO_3.groupby('ID', as_index=False).nth(2).event.isin([512, 513, 53])))].ID.unique()

CE_mergers = d.loc[d.event == 52]

SMT1_CE2_CE2_merge = np.intersect1d(CE_mergers, SMT1_CE2_CE2)

SMT1_CE2_CE2_survive = np.setxor1d(SMT1_CE2_CE2, SMT1_CE2_CE2_merge)
for ii in CE1_survive[:10]:
    print(d.loc[d.ID == ii][['time', 'mass1', 'mass2', 'type1', 'type2', 'semiMajor', 'event']])
    print(bpp.loc[bpp.bin_num == ii][['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'evol_type']])
# -

for ID in channels['RLO_3_other'][::50]:
    print(ID)
    print(d.loc[(d.ID == ID)][['time', 'mass1', 'mass2', 'type1', 'type2', 'semiMajor', 'event']])
    print(bpp.loc[bpp.bin_num == ID][['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'evol_type']])

d_CE2_CE2 = d.loc[(d.ID.isin(channels['CE2_CE2'])) & (d.ID.isin(init_09.ID))]

channels['CE2_CE2']

for id in d_CE2_CE2.ID.unique()[:1]:
    print(d_CE2_CE2.loc[d_CE2_CE2.ID == id][['time', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor', 'event']])


bpp.loc[bpp.bin_num == 355087][['tphys', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'sep', 'evol_type', 'RRLO_1', 'RRLO_2']]

bpp.loc[(bpp.kstar_1.isin([10,11,12])) & (bpp.kstar_2.isin([10,11,12]))][['mass_1','mass_2', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']]

CE = bpp.loc[bpp.evol_type == 7][['mass_1','mass_2', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']]

SMT = bpp.loc[(bpp.evol_type == 3)]

# +
#SMT = SMT.loc[~SMT.bin_num.isin(CE.bin_num)]
# -

for bn in SMT.bin_num.unique()[:5]:
    print(bpp.loc[bpp.bin_num == bn][['mass_1','mass_2', 'kstar_1', 'kstar_2', 'porb', 'evol_type', 'bin_num']])


# -


