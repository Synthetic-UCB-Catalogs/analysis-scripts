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
from collections import Counter

import formation_channels as fc
from rapid_code_load_T0 import convert_BSE_data_to_T0, convert_COMPAS_data_to_T0, convert_COSMIC_data_to_T0, convert_SeBa_data_to_T0, load_T0_data


# -
def get_first_RLO_figure(d, q=0.49, savefig=None):
    ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=d)

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
        print("warning, you missed ids:", np.setxor1d(all_IDs, id_check))
        print(len(all_IDs), len(id_check))

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
                plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
                print(len(ZAMS_select), k)
                
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


COSMIC_T0, COSMIC_header = load_T0_data('data/T0_format_pilot/COSMIC/basic/COSMIC_T0.hdf5')
METISSE_T0, METISSE_header = load_T0_data('data/T0_format_pilot/METISSE-COSMIC/basic/METISSE_T0.hdf5')
COMPAS_T0, COMPAS_header = load_T0_data('data/T0_format_pilot/COMPAS/COMPAS_T0.hdf5')

first_RLO_COSMIC_05 = get_first_RLO_figure(COSMIC_T0, q=0.49, savefig='COSMIC_first_RLO_channels_qinit_05.png')
first_RLO_COSMIC_09 = get_first_RLO_figure(COSMIC_T0, q=0.88, savefig='COSMIC_first_RLO_channels_qinit_09.png')

first_RLO_METISSE_05 = get_first_RLO_figure(METISSE_T0, q=0.49, savefig='METISSE_first_RLO_channels_qinit_05.png')
first_RLO_METISSE_09 = get_first_RLO_figure(METISSE_T0, q=0.88, savefig='METISSE_first_RLO_channels_qinit_09.png')

first_RLO_COMPAS_05 = get_first_RLO_figure(COMPAS_T0, q=0.49, savefig='COMPAS_first_RLO_channels_qinit_05.png')
first_RLO_COMPAS_09 = get_first_RLO_figure(COMPAS_T0, q=0.88, savefig='COMPAS_first_RLO_channels_qinit_09.png')





# ### Ignore everything below this -- these are for not first RLO

# +
#SMT_keys = ['SMT1_SMT1', 'SMT2_SMT2', 'SMT1', 'SMT2', 'SMT1_SMT2']
#CE_keys = ['CE1_CE1', 'CE2_CE2', 'CE1_merge', 'CE1_survive', 'CE2', 'CE3', 'CE1_CE2']
#mixed_keys_2 = ['SMT1_CE1', 'SMT2_CE2', 'CE1_SMT1', 'CE2_SMT2', 'SMT1_CE2', 'CE1_SMT2']
#mixed_keys_3 = ['evCE1_SMT2', 'evCE1_CE2', 'SMT1_evCE2', 'CE1_evCE2', 'SMT1_CE2_SMT2', 'SMT1_CE2_CE2', 'SMT1_SMT1_CE2']
#other_keys = ['RLO_3_other', 'RLO_4_or_more_other', 'No_RLO']

# +
#SMT_colors = cmr.take_cmap_colors('cmr.sapphire', len(SMT_keys), cmap_range=(0.4, 0.85), return_fmt='hex')
#CE_colors = cmr.take_cmap_colors('cmr.sunburst', len(CE_keys), cmap_range=(0.3, 0.9), return_fmt='hex')
#mixed_colors_2 = cmr.take_cmap_colors('cmr.nuclear', len(mixed_keys_2), cmap_range=(0.15, 0.95), return_fmt='hex')
#mixed_colors_3 = cmr.take_cmap_colors('cmr.horizon', len(mixed_keys_3), cmap_range=(0.15, 0.95), return_fmt='hex')
#other_colors = cmr.take_cmap_colors('cmr.neutral', len(other_keys), cmap_range=(0.35, 0.85), return_fmt='hex')

# +
#keys_list = [SMT_keys, CE_keys, mixed_keys_2, mixed_keys_3, other_keys]
#colors_list = [SMT_colors, CE_colors, mixed_colors_2, mixed_colors_3, other_colors]

# + colab={"base_uri": "https://localhost:8080/"} id="q1cShMsXHZ9n" executionInfo={"status": "ok", "timestamp": 1716398731600, "user_tz": 240, "elapsed": 19009, "user": {"displayName": "Katelyn Breivik", "userId": "00438142393458917517"}} outputId="9735f1f2-99ee-4bb2-b38b-a4341eb77bfc"
#fig = plt.figure(figsize=(5.8,4), dpi=150)
#channel_keys = channels.keys()
#for colors, keys in zip(colors_list, keys_list):
#    for c, k, ii in zip(colors, keys, range(len(colors))):
#        ZAMS_select = init_09.loc[(init_09.ID.isin(channels[k]))]
#        
#        if len(ZAMS_select) > 0:
#            #if k != 'failed_CE':
#            print(len(ZAMS_select), k)
#            plt.scatter(ZAMS_select.semiMajor, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
#            
#            
#        else:
#            print(0, k)#

##DWD_select = DWD.loc[DWD.ID.isin(init_09.ID)]
##DWD_ZAMS = init_09.loc[init_09.ID.isin(DWD.ID)]
##print(DWD_select)
##plt.scatter(DWD_ZAMS.porb, DWD_ZAMS.mass1, c='black', s=50, zorder=0)
#print()
#print()
#plt.xscale('log')
#plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
#plt.yscale('log')
#plt.xlim(min(init_05.semiMajor)-0.1, max(init_05.semiMajor))
#plt.ylim(min(init_05.mass1)-0.05, max(init_05.mass1)+0.5)#

#plt.xlabel('semimajor axis [Rsun]')
#plt.ylabel('M$_1$ [Msun]')
#plt.show()
##plt.savefig('cosmic_q_i_09.png', facecolor='white', dpi=150)

# +
#fig = plt.figure(figsize=(5.8,4), dpi=150)
#channel_keys = channels.keys()
#for colors, keys in zip(colors_list, keys_list):
#    for c, k, ii in zip(colors, keys, range(len(colors))):
#        ZAMS_select = init_05.loc[(init_05.ID.isin(channels[k]))]
#        
#        if len(ZAMS_select) > 0:
#            #if k != 'failed_CE':
#            print(len(ZAMS_select), k)
#            plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
#            
#            
#        else:
#            print(0, k)
#
##DWD_select = DWD.loc[DWD.ID.isin(init_09.ID)]
##DWD_ZAMS = init_09.loc[init_09.ID.isin(DWD.ID)]
##print(DWD_select)
##plt.scatter(DWD_ZAMS.porb, DWD_ZAMS.mass1, c='black', s=50, zorder=0)
#print()
#print()
#plt.xscale('log')
#plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
#plt.yscale('log')
#plt.xlim(min(init_05.porb)-0.1, max(init_05.porb)+1e3)
#plt.ylim(min(init_05.mass1)-0.05, max(init_05.mass1)+0.5)
#
#plt.xlabel('orbital period [days]')
#plt.ylabel('M$_1$ [Msun]')
#plt.show()
#plt.savefig('cosmic_q_i_09.png', facecolor='white', dpi=150)
