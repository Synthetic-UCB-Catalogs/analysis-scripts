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
from rapid_code_load_T0 import load_BSE_data, load_COMPAS_data, load_COSMIC_data, load_SeBa_data, load_T0_data
# -
COSMIC = 'data/basic.h5'
c, c_header = load_COSMIC_data(COSMIC, metallicity=0.02)

METISSE = 'data/basic_METISSE.h5'
m, m_header = load_COSMIC_data(METISSE, metallicity=0.02)

COMPAS = 'data/COMPAS_pilot.h5'
co = load_COMPAS_data(COMPAS)


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
    first_RLO = fc.first_interaction_channels(d=d)

    init_q = ZAMS.loc[(np.round(ZAMS.mass2/ZAMS.mass1, 2) == q)]

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

first_RLO_c_05 = get_first_RLO_figure(d=c, q=0.49, savefig='first_RLO_COSMIC_pilot_qinit05.png')

first_RLO_c09 = get_first_RLO_figure(d=c, q=0.88, savefig='first_RLO_COSMIC_pilot_qinit09.png')

first_RLO_co_05 = get_first_RLO_figure(d=co, q=0.49, savefig='first_RLO_COMPAS_pilot_qinit09.png')

first_RLO_co_09 = get_first_RLO_figure(d=co, q=0.88)

first_RLO_m_05 = get_first_RLO_figure(d=m, q=0.49)

first_RLO_m_09 = get_first_RLO_figure(d=m, q=0.88)







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


