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
from rapid_code_load_T0 import load_T0_data
import rapid_code_load_T0 as load
import formation_channels as fc
# -
#BSE_T0 = load.convert_BSE_data_to_T0('data/pilot_runs_raw_data/BSE/fiducial/BinaryRun.dat', 0.02, outputpath='data/T0_format_pilot/BSE/fiducial', hdf5_filename="BSE_T0.hdf5")
BSE_T0, BSE_header = load_T0_data('data/T0_format_pilot/BSE/fiducial/BSE_T0.hdf5')
#COSMIC_T0, COSMIC_header = load_T0_data('data/T0_format_pilot/COSMIC/basic/COSMIC_T0.hdf5')
#METISSE_T0, METISSE_header = load_T0_data('data/T0_format_pilot/METISSE-COSMIC/basic/METISSE_T0.hdf5')
#COMPAS_T0, COMPAS_header = load_T0_data('data/T0_format_pilot/COMPAS/COMPAS_T0.hdf5')
#SeBa_T0, SeBa_header = load_T0_data('data/IC_variations/qmin_01/SeBa_T0.hdf5')


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
    keys = ['SMT_1', 'SMT_2', 'CE_1', 'CE_2', 'DCCE', 'DCCE_merger', 'failed_CE_merger', 'contact_merger', 'nonRLO', 'leftovers']

    id_check = []
    for k in keys:
        id_check.extend(first_RLO[k])

    if len(np.setxor1d(all_IDs, id_check)) > 0:
        print("warning, you missed ids:", np.setxor1d(all_IDs, id_check))
        print(len(all_IDs), len(id_check))

    SMT_colors = cmr.take_cmap_colors('cmr.neutral', 2, cmap_range=(0.6, 0.85), return_fmt='hex')
    CE_colors = cmr.take_cmap_colors('cmr.sunburst', 3, cmap_range=(0.3, 0.9), return_fmt='hex')
    other_colors = cmr.take_cmap_colors('cmr.sapphire', 5, cmap_range=(0.3, 0.95), return_fmt='hex')

    
    keys_list = [['SMT_1', 'SMT_2'], ['CE_1', 'CE_2', 'DCCE'], ['DCCE_merger', 'failed_CE_merger', 'contact_merger', 'nonRLO', 'leftovers']]
    colors_list = [SMT_colors, CE_colors, other_colors]

    fig = plt.figure(figsize=(6,4.9))
    
    for colors, keys in zip(colors_list, keys_list):
        for c, k, ii in zip(colors, keys, range(len(colors))):
            ZAMS_select = init_q.loc[(init_q.ID.isin(first_RLO[k]))]
            
            if len(ZAMS_select) > 0:
                plt.scatter(ZAMS_select.porb, ZAMS_select.mass1, c=c, s=5.8, label=k, zorder=200 - (1+ii)*5, marker='s')
                print(len(ZAMS_select), k)
                
            else:
                print(0, k)
    print()
    print()
    plt.xscale('log')
    plt.legend(loc=(0.0, 1.01), ncol=3, prop={'size':9})
    plt.yscale('log')
    
    
    plt.xlim(1, 10000)
    plt.ylim(min(init_q.mass1), max(init_q.mass1))
    
    plt.xlabel('orbital period [day]')
    plt.ylabel('M$_1$ [Msun]')
    if savefig != None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=100, facecolor='white')
    plt.show()

    return first_RLO

initBSE = BSE_T0.groupby('ID').first()

q = (initBSE.mass2/initBSE.mass1).values

q

print(np.round(q[(q<0.9) & (q > 0.89)], 3))
print(len(np.round(q[(q<0.9) & (q > 0.89)], 3)))


first_RLO_BSE_01 = get_first_RLO_figure(BSE_T0, q=0.08, savefig='BSE_first_RLO_channels_qinit_01.png')
first_RLO_BSE_05 = get_first_RLO_figure(BSE_T0, q=0.5, savefig='BSE_first_RLO_channels_qinit_05.png')
first_RLO_BSE_09 = get_first_RLO_figure(BSE_T0, q=0.89, savefig='BSE_first_RLO_channels_qinit_09.png')

BSE_01_SMT1 = BSE_T0.loc[BSE_T0.ID.isin(first_RLO_BSE_01['SMT_1'])]

for id in first_RLO_BSE_01['SMT_1'][:10]:
    print(BSE_T0.loc[BSE_T0.ID == id][['time', 'event', 'semiMajor', 'type1', 'type2', 'mass1', 'mass2']])

BSE_05_SMT1 = BSE_T0.loc[BSE_T0.ID.isin(first_RLO_BSE_05['SMT_1'])]

ceid = BSE_01_SMT1.loc[BSE_01_SMT1.event == 511.0].ID
print(len(ceid))

for id in ceid[:10]:
    print(id)
    print(BSE_05_SMT1.loc[BSE_05_SMT1.ID == id][['time', 'type1', 'type2', 'event', 'mass1', 'mass2', 'semiMajor']])
    print()
    print()
# ### First let's load the different pilot runs for COSMIC

COSMIC_T0_basic, COSMIC_header_basic = load_T0_data('data/T0_format_pilot/COSMIC/basic/COSMIC_T0.hdf5')
COSMIC_T0_basic1b, COSMIC_header_basic1b = load_T0_data('data/T0_format_pilot/COSMIC/basic1b/COSMIC_T0.hdf5')
COSMIC_T0_intermediate, COSMIC_header_intermediate = load_T0_data('data/T0_format_pilot/COSMIC/intermediate/COSMIC_T0.hdf5')

first_RLO_COSMIC_b_05 = get_first_RLO_figure(COSMIC_T0_basic, q=0.49, savefig='COSMIC_b_first_RLO_channels_qinit_05.png')
first_RLO_COSMIC_b1_05 = get_first_RLO_figure(COSMIC_T0_basic1b, q=0.49, savefig='COSMIC_b1_first_RLO_channels_qinit_05.png')
first_RLO_COSMIC_i_05 = get_first_RLO_figure(COSMIC_T0_intermediate, q=0.49, savefig='COSMIC_b1_first_RLO_channels_qinit_05.png')


# ### There are only slight differences between basic, basic1b, and intermediate
#
# #### We can look at the different evolutions to see what is happening with the different outcomes of the first RLO 

COSMIC_b_05_merger = COSMIC_T0_basic.loc[COSMIC_T0_basic.ID.isin(first_RLO_COSMIC_b_05['failed_CE_merger'])]

print(len(COSMIC_b_05_merger.ID.unique()))

# +
print_cols = ["time", "event", "type1", "type2", "mass1", "mass2", "semiMajor"]

for ID in COSMIC_b_05_merger.ID.unique()[::100]:
    print(COSMIC_b_05_merger.loc[COSMIC_b_05_merger.ID == ID][print_cols])
# -

# ### We can also look at how the WDMS population is affected for each channel

# +
ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=COSMIC_T0_basic)

ZAMS_1b, WDMS_1b, DWD_1b = fc.select_evolutionary_states(d=COSMIC_T0_basic1b)

ZAMS_i, WDMS_i, DWD_i = fc.select_evolutionary_states(d=COSMIC_T0_intermediate)

# +
for channel in ['SMT_1', 'CE_1', 'nonRLO']:
    WDMS_select = WDMS.loc[WDMS.ID.isin(first_RLO_COSMIC_b_05[channel])]
    plt.scatter(WDMS_select.mass1, WDMS_select.mass2, label=channel+', basic', s=6, marker='o')

    WDMS_select = WDMS_1b.loc[WDMS_1b.ID.isin(first_RLO_COSMIC_b1_05[channel])]
    plt.scatter(WDMS_select.mass1, WDMS_select.mass2, label=channel+', basic 1b', s=2, marker='s')

plt.xlabel('mass 1 [Msun]')
plt.ylabel('mass 2 [Msun]')
plt.legend()

plt.tight_layout()
plt.savefig('COSMIC_compare_basic_basic1b_m1_m2.png', facecolor='white')

# +
for channel in ['SMT_1', 'CE_1', 'nonRLO']:
    WDMS_select = WDMS.loc[WDMS.ID.isin(first_RLO_COSMIC_b_05[channel])]
    plt.scatter(WDMS_select.mass1, WDMS_select.mass2, label=channel+', basic', s=6, marker='o')

    WDMS_select = WDMS_i.loc[WDMS_1b.ID.isin(first_RLO_COSMIC_i_05[channel])]
    plt.scatter(WDMS_select.mass1, WDMS_select.mass2, label=channel+', intermediate', s=2, marker='s')

plt.xlabel('mass 1 [Msun]')
plt.ylabel('mass 2 [Msun]')
plt.legend()

plt.tight_layout()
plt.savefig('COSMIC_compare_basic_intermediate_m1_m2.png', facecolor='white')

# +
for channel in ['SMT_1', 'CE_1', 'nonRLO']:
    WDMS_select = WDMS.loc[WDMS.ID.isin(first_RLO_COSMIC_b_05[channel])]
    plt.scatter(WDMS_select.semiMajor, WDMS_select.mass1, label=channel, s=2)

    WDMS_select = WDMS_1b.loc[WDMS_1b.ID.isin(first_RLO_COSMIC_b1_05[channel])]
    plt.scatter(WDMS_select.semiMajor, WDMS_select.mass1, label=channel+', basic 1b', s=2, marker='s')


plt.ylabel('mass 1 [Msun]')
plt.xlabel('semimajor axis [Rsun]')
plt.xscale('log')
plt.legend()

plt.tight_layout()
plt.savefig('COSMIC_compare_basic_basic1b_sep_m1.png', facecolor='white')

# +
for channel in ['SMT_1', 'CE_1', 'nonRLO']:
    WDMS_select = WDMS.loc[WDMS.ID.isin(first_RLO_COSMIC_b_05[channel])]
    plt.scatter(WDMS_select.semiMajor, WDMS_select.mass1, label=channel, s=2)

    WDMS_select = WDMS_i.loc[WDMS_1b.ID.isin(first_RLO_COSMIC_i_05[channel])]
    plt.scatter(WDMS_select.semiMajor, WDMS_select.mass1, label=channel+', intermediate', s=2, marker='s')


plt.ylabel('mass 1 [Msun]')
plt.xlabel('semimajor axis [Rsun]')
plt.xscale('log')
plt.legend()

plt.tight_layout()
plt.savefig('COSMIC_compare_basic_intermediate_sep_m1.png', facecolor='white')
# -





# ### We can also compare between codes

METISSE_T0, METISSE_header = load_T0_data('data/T0_format_pilot/METISSE-COSMIC/basic/METISSE_T0.hdf5')
COMPAS_T0, COMPAS_header = load_T0_data('data/T0_format_pilot/COMPAS/COMPAS_T0.hdf5')

first_RLO_COSMIC_05 = get_first_RLO_figure(COSMIC_T0_basic, q=0.49, savefig='COSMIC_first_RLO_channels_qinit_05.png')
first_RLO_COSMIC_09 = get_first_RLO_figure(COSMIC_T0_basic, q=0.88, savefig='COSMIC_first_RLO_channels_qinit_09.png')

first_RLO_METISSE_01 = get_first_RLO_figure(METISSE_T0, q=0.08, savefig='METISSE_first_RLO_channels_qinit_01.png')
first_RLO_METISSE_05 = get_first_RLO_figure(METISSE_T0, q=0.49, savefig='METISSE_first_RLO_channels_qinit_05.png')
first_RLO_METISSE_09 = get_first_RLO_figure(METISSE_T0, q=0.88, savefig='METISSE_first_RLO_channels_qinit_09.png')

first_RLO_COMPAS_01 = get_first_RLO_figure(COMPAS_T0, q=0.08, savefig='COMPAS_first_RLO_channels_qinit_01.png')
first_RLO_COMPAS_05 = get_first_RLO_figure(COMPAS_T0, q=0.49, savefig='COMPAS_first_RLO_channels_qinit_05.png')
first_RLO_COMPAS_09 = get_first_RLO_figure(COMPAS_T0, q=0.88, savefig='COMPAS_first_RLO_channels_qinit_09.png')

first_RLO_SeBa_01 = get_first_RLO_figure(SeBa_T0, q=0.08, savefig='SeBa_first_RLO_channels_qinit_01.png')
first_RLO_SeBa_05 = get_first_RLO_figure(SeBa_T0, q=0.49, savefig='SeBa_first_RLO_channels_qinit_05.png')
first_RLO_SeBa_09 = get_first_RLO_figure(SeBa_T0, q=0.88, savefig='SeBa_first_RLO_channels_qinit_09.png')

first_RLO_SeBa_01





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
# -






