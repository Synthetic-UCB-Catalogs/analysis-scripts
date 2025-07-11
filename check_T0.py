# ---
# jupyter:
#   jupytext:
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

# ## This notebook loads/converts the raw code data taken from this Pilot Runs folder: https://drive.google.com/drive/folders/1rozj_qlCgcFrDPSkH4t4JxAAV1OeM5Qg?usp=drive_link

# ### Some things we will check are whether the RLO is treated properly and what conversion to-do items are left:
#
# #### Here are the types
# <div>
#     <img src=attachment:4de3b921-5b64-46f5-9a3b-61b78b44fdae.png width=500>
# <div>
# <div>
#     <img src=attachment:af130430-4b5e-4673-9664-8ba5995a7dd0.png width=500>
# <div>
#     
# #### and here are the events
# <div>
#     <img src=attachment:6e483f22-e89e-4936-9bf9-142114a3fb55.png width=500>
# <div>
# <div>
#     <img src=attachment:a1715666-fc73-4a6c-b105-0ee574ae6231.png width=500>
# <div>

import rapid_code_load_T0 as load
import formation_channels as fc

# #### First we check BSE

BSE_T0 = load.convert_BSE_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/BSE/fiducial/BinaryRun.dat", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/BSE/fiducial/", 
    hdf5_filename="BSE_T0.hdf5")

BSE_T0

# #### Ok -- a to do for BSE is to extract the events

# #### Next up is COMPAS

COMPAS_T0 = load.convert_COMPAS_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/COMPAS/COMPAS_pilot.h5",
    outputpath="data/T0_format_pilot/COMPAS/", 
    hdf5_filename="COMPAS_T0.hdf5")

COMPAS_T0_RLO = COMPAS_T0.loc[COMPAS_T0.event.isin([31, 32, 511, 512, 513, 52, 53])]
for id in COMPAS_T0_RLO.ID.unique()[:10]:
    print(COMPAS_T0.loc[COMPAS_T0.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### For ID 3301: it looks like there is an instantaneous SMT that doesn't alter the semimajor axis; is this expected?
#
# ##### For ID 3302: I see an RLO, then CE from the initially more massive star then the CE results in a merger confirmed by event = 52 then event=84. The semiMajor axis is still 0.096 Rsun though; is this expected?
# ##### Same for 3303-33101
#
# ##### The event is 13 for both stars changing type at ZAMS -- need to implement this in other converters. 
#
# ##### Is there metallicity information in the T0 data?

# #### Next lets check COSMIC

COSMIC_T0_basic = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/COSMIC/basic.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/COSMIC/basic/", 
    hdf5_filename="COSMIC_T0.hdf5")

# +
COSMIC_T0_basic_RLO = COSMIC_T0_basic.loc[COSMIC_T0_basic.event.isin([31, 32, 511, 512, 513, 52, 53])]
COSMIC_T0_basic_event0 = COSMIC_T0_basic.loc[COSMIC_T0_basic.event.isin([0])]

for id in COSMIC_T0_basic_event0.ID.unique()[:10]:
    print(COSMIC_T0_basic.loc[COSMIC_T0_basic.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()
# -

# ##### need to make ZAMS event=13
# ##### for ID 3299: need to fix events to read as CE onset from time 12478.81 and CE merger; only the event needs to be updated since the masses and semimajor axies are good. 
#
# ### The above is fixed!

COSMIC_T0_basic1b = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/COSMIC/basic1b.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/COSMIC/basic1b/", 
    hdf5_filename="COSMIC_T0.hdf5")

COSMIC_T0_basic1b_RLO = COSMIC_T0_basic1b.loc[COSMIC_T0_basic1b.event.isin([31, 32, 511, 512, 513, 52, 53])]
for id in COSMIC_T0_basic1b_RLO.ID.unique()[:10]:
    print(COSMIC_T0_basic1b.loc[COSMIC_T0_basic1b.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### For ID 3299: undifined event at 13695; spurious event = 12 at 12339? Also showing for 3300
# ##### Same problem as 'basic' run above for 3302

COSMIC_T0_intermediate = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/COSMIC/intermediate.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/COSMIC/intermediate/", 
    hdf5_filename="COSMIC_T0.hdf5")

COSMIC_T0_intermediate_RLO = COSMIC_T0_intermediate.loc[COSMIC_T0_intermediate.event.isin([31, 32, 511, 512, 513, 52, 53])]
for id in COSMIC_T0_intermediate_RLO.ID.unique()[:10]:
    print(COSMIC_T0_intermediate.loc[COSMIC_T0_intermediate.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### For ID 0-2, 99-102, 199-201: need to log the CE merger correctly; events are currently 0.0

# #### Next up is METISSE-COSMIC
#

METISSE_T0_basic = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/METISSE-COSMIC/basic.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/METISSE-COSMIC/basic/", 
    hdf5_filename="METISSE_T0.hdf5")

METISSE_T0_basic_RLO = METISSE_T0_basic.loc[METISSE_T0_basic.event.isin([0])]
for id in METISSE_T0_basic_RLO.ID.unique()[:10]:
    print(METISSE_T0_basic.loc[METISSE_T0_basic.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### need to update the ZAMS time (will be fixed with COSMIC)
# ##### For ID 1899-1902: missing event at 13351; maybe a 41?
# ##### For ID 1906-1908: the CE is not logged and the CE merger is not logged

METISSE_T0_basic1b = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/METISSE-COSMIC/basic1b.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/METISSE-COSMIC/basic1b/", 
    hdf5_filename="METISSE_T0.hdf5")

METISSE_T0_basic1b_RLO = METISSE_T0_basic1b.loc[METISSE_T0_basic1b.event.isin([31, 32, 511, 512, 513, 52, 53])]
for id in METISSE_T0_basic1b_RLO.ID.unique()[:10]:
    print(METISSE_T0_basic1b.loc[METISSE_T0_basic1b.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### need to update the ZAMS time (will be fixed with COSMIC): same as above
# ##### For ID 1899-1902: missing event at 13351; maybe a 41?
# ##### For ID 1906-1908: the CE is not logged and the CE merger is not logged
#
# ##### Need to check whether the models are different between basic and basic1b

METISSE_T0_intermediate = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/METISSE-COSMIC/intermediate.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/METISSE-COSMIC/intermediate/", 
    hdf5_filename="METISSE_T0.hdf5")

METISSE_T0_intermediate_RLO = METISSE_T0_intermediate.loc[METISSE_T0_intermediate.event.isin([511, 512, 513, 52, 53])]
for id in METISSE_T0_intermediate_RLO.ID.unique()[:10]:
    print(METISSE_T0_intermediate.loc[METISSE_T0_intermediate.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### similar to above basic and basic1b; COSMIC is much more different from basic to intermediate

# #### Next up is SeBa

SeBa_T0_basic, header = load.load_T0_data(ifilepath="data/T0_format_pilot/SeBa/basic/SeBa_T0.hdf5")

SeBa_T0_basic

first_RLO_SeBa = fc.first_interaction_channels(SeBa_T0_basic)

SeBa_T0_basic_RLO = SeBa_T0_basic.loc[SeBa_T0_basic.event.isin([511])]
for id in SeBa_T0_basic_RLO.ID.unique()[200:210]:
    print(SeBa_T0_basic.loc[SeBa_T0_basic.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor', 'RRLO_1']])
#print(SeBa_T0_basic.loc[SeBa_T0_basic.ID == 3315])
#s = s.loc[~((s.event == 31) & (s.event.shift() == 31))]
#print(SeBa_T0_basic.loc[~((SeBa_T0_basic.event == 31) & (SeBa_T0_basic.event.shift() == 31)) & (SeBa_T0_basic.type1 == SeBa_T0_basic.type1.shift())])

s
SeBa_T0_basic.loc[SeBa_T0_basic.ID == 77131]

# ##### Ok! so for this one, we need a lot of work on the event = 32 situation. 

SeBa_T0_intermediate = load.convert_SeBa_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/SeBa/SeBa-intermediate.data", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/SeBa/intermediate", 
    hdf5_filename="SeBa_T0.hdf5")

SeBa_T0_intermediate_RLO = SeBa_T0_intermediate.loc[SeBa_T0_intermediate.event.isin([31, 32, 511, 512, 513, 52, 53])]
for id in SeBa_T0_intermediate_RLO.ID.unique()[:10]:
    print(SeBa_T0_intermediate.loc[SeBa_T0_intermediate.ID == id][['ID', 'time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])
    print()

# ##### events are messed up here too 52 should only be once. 


