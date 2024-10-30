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

# +
import rapid_code_load_T0 as load

import pandas as pd
import numpy as np
# -

COSMIC_T0_basic = load.convert_COSMIC_data_to_T0(
    ifilepath="data/pilot_runs_raw_data/COSMIC/basic.h5", 
    metallicity=0.02, 
    outputpath="data/T0_format_pilot/COSMIC/basic/", 
    hdf5_filename="COSMIC_T0.hdf5")

COSMIC_T0_basic[COSMIC_T0_basic.ID == 467096]

dat = pd.read_hdf("data/pilot_runs_raw_data/COSMIC/basic.h5", key='bpp')
bcm = pd.read_hdf("data/pilot_runs_raw_data/COSMIC/basic.h5", key='bcm')

dat.loc[dat.bin_num == 467096][['tphys', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'RRLO_1', 'RRLO_2', 'porb', 'evol_type']]




# +
dat = dat.rename(
        columns={"tphys": "time", "mass_1": "mass1", "mass_2": "mass2", 
                 "massc_1": "massHeCore1", "massc_2": "massHeCore2", 
                 "porb": "period", "sep": "semiMajor", "ecc": "eccentricity",
                 "teff_1": "Teff1", "teff_2": "Teff2",
                 "rad_1": "radius1", "rad_2": "radius2",
                 "bin_num": "UID"})
dat["radiusRL1"] = dat.radius1 / dat.RRLO_1
dat["radiusRL2"] = dat.radius2 / dat.RRLO_2
# convert evol_type to event
# grab bin_nums for SNe types from bcm
bn_1_cc = bcm.loc[bcm.SN_1 == 1].bin_num.unique()
bn_2_cc = bcm.loc[bcm.SN_2 == 1].bin_num.unique()
bn_1_ecsn = bcm.loc[bcm.SN_1 == 2].bin_num.unique()
bn_2_ecsn = bcm.loc[bcm.SN_2 == 2].bin_num.unique()
bn_1_ussn = bcm.loc[bcm.SN_1 == 3].bin_num.unique()
bn_2_ussn = bcm.loc[bcm.SN_2 == 3].bin_num.unique()
bn_1_aic = bcm.loc[bcm.SN_1 == 4].bin_num.unique()
bn_2_aic = bcm.loc[bcm.SN_2 == 4].bin_num.unique()
bn_1_mic = bcm.loc[bcm.SN_1 == 5].bin_num.unique()
bn_2_mic = bcm.loc[bcm.SN_2 == 5].bin_num.unique()
bn_1_ppisn = bcm.loc[bcm.SN_1 == 6].bin_num.unique()
bn_2_ppisn = bcm.loc[bcm.SN_2 == 6].bin_num.unique()
bn_1_pisn = bcm.loc[bcm.SN_1 == 7].bin_num.unique()
bn_2_pisn = bcm.loc[bcm.SN_2 == 7].bin_num.unique()
bn_1_DC_fryer = bcm.loc[(bcm.SN_1 == 1) & (bcm.mass_1 > 20) & (bcm.kstar_1 == 14)].bin_num.unique()
bn_2_DC_fryer = bcm.loc[(bcm.SN_2 == 1) & (bcm.mass_2 > 20) & (bcm.kstar_2 == 14)].bin_num.unique()
# grab bin_nums for mergers from bcm
bn_merger = bcm.loc[bcm.bin_state == 1].bin_num.unique()

# rename the columns that are easy to rename
dat = dat.rename(
    columns={"tphys": "time", "mass_1": "mass1", "mass_2": "mass2", 
             "massc_1": "massHeCore1", "massc_2": "massHeCore2", 
             "porb": "period", "sep": "semiMajor", "ecc": "eccentricity",
             "teff_1": "Teff1", "teff_2": "Teff2",
             "rad_1": "radius1", "rad_2": "radius2",
             "bin_num": "UID"})
dat["radiusRL1"] = dat.radius1 / dat.RRLO_1
dat["radiusRL2"] = dat.radius2 / dat.RRLO_2
# -

# convert evol_type to event
dat["event"] = np.zeros(len(dat))
dat.loc[dat.evol_type == 1, "event"] = 13
dat.loc[(dat.evol_type == 2) & (dat.kstar_1.shift() < dat.kstar_1), "event"] = 11
dat.loc[(dat.evol_type == 2) & (dat.kstar_2.shift() < dat.kstar_2), "event"] = 12
dat.loc[(dat.evol_type == 2) & (dat.kstar_2.shift() > dat.kstar_2), "event"] = 12
dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 > 1) & (dat.RRLO_2 < 1), "event"] = 31
dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 < 1) & (dat.RRLO_2 > 1), "event"] = 32
dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 > 1) & (dat.RRLO_2 > 1), "event"] = 33
dat.loc[(dat.evol_type == 4) & (dat.RRLO_1 > 1) & (dat.period > 0), "event"] = 41
dat.loc[(dat.evol_type == 4) & (dat.RRLO_1 > 1) & (dat.period > 0), "event"] = 42
dat.loc[(dat.evol_type == 4) & (dat.kstar_1 > 7) & (dat.kstar_2 > 7) & (dat.RRLO_1 == 0.99) & (dat.RRLO_2 == 0.99), "event"] = 43
dat.loc[(dat.evol_type == 4) & (dat.period == 0), "event"] = 52
dat.loc[(dat.evol_type == 5), "event"] = 53
dat.loc[(dat.evol_type == 6) & ((dat.RRLO_1 > 1) | (dat.RRLO_2 > 1)), "event"] = 52
dat.loc[(dat.evol_type == 6) & (dat.RRLO_2 == -1), "event"] = 52
dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 > 1) & (dat.RRLO_2 < 1), "event"] = 511
dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 < 1) & (dat.RRLO_2 > 1), "event"] = 512
dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 > 1) & (dat.RRLO_2 > 1), "event"] = 513
dat.loc[(dat.evol_type == 8) & (dat.RRLO_1 > 1) & (dat.period > 0), "event"] = 41
dat.loc[(dat.evol_type == 8) & (dat.RRLO_2 > 1) & (dat.period > 0), "event"] = 42
dat.loc[(dat.evol_type == 8) & (dat.kstar_1 > 7) & (dat.kstar_2 > 7) & (dat.RRLO_1 == 0.99) & (dat.RRLO_2 == 0.99), "event"] = 43
dat.loc[(dat.evol_type == 8) & (dat.period == 0), "event"] = 52
dat.loc[(dat.evol_type == 15) & (dat.UID.isin(bn_1_cc)), "event"] = 212
dat.loc[(dat.evol_type == 8) & (dat.period == -1), "event"] = 52
dat.loc[(dat.evol_type == 16) & (dat.UID.isin(bn_2_cc)), "event"] = 222
dat.loc[(dat.evol_type == 15) & (dat.UID.isin(bn_1_ecsn)), "event"] = 213
dat.loc[(dat.evol_type == 16) & (dat.UID.isin(bn_2_ecsn)), "event"] = 223
dat.loc[(dat.evol_type == 15) & (dat.UID.isin(bn_1_pisn)), "event"] = 214
dat.loc[(dat.evol_type == 16) & (dat.UID.isin(bn_2_pisn)), "event"] = 224
dat.loc[(dat.evol_type == 15) & (dat.UID.isin(bn_1_ppisn)), "event"] = 215
dat.loc[(dat.evol_type == 16) & (dat.UID.isin(bn_2_ppisn)), "event"] = 225
dat.loc[(dat.evol_type == 15) & (dat.UID.isin(bn_1_DC_fryer)), "event"] = 216
dat.loc[(dat.evol_type == 16) & (dat.UID.isin(bn_2_DC_fryer)), "event"] = 226
dat.loc[(dat.evol_type == 10) & (dat.semiMajor < 0), "event"] = 83
dat.loc[(dat.evol_type == 10) & (dat.semiMajor > 0), "event"] = 81
dat.loc[(dat.evol_type == 10) & (dat.semiMajor == 0) & (dat.UID.isin(bn_merger)), "event"] = 84
dat.loc[(dat.evol_type == 10) & (dat.semiMajor == 0) & (dat.kstar_1.isin([13,14])) & (dat.kstar_2 == 15) & (dat.UID.isin(bn_merger)), "event"] = 82

bcm.loc[bcm.bin_num == 278665].SN_1

dat.loc[dat.UID == 350674][['time', 'kstar_1', 'kstar_2', 'mass1', 'mass2', 'semiMajor', 'RRLO_1', 'RRLO_2', 'evol_type', 'event','period']]

bpp.loc[(bpp.evol_type == 8) &(bpp.sep > 0) & (bpp.RRLO_1 ==0.99) & (bpp.RRLO_2 ==0.99)][['bin_num', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'sep', 'RRLO_1', 'RRLO_2', 'evol_type']]

bpp.loc[(bpp.evol_type == 7) & (bpp.RRLO_2 < 1) & (bpp.RRLO_1 > 1)]

bpp.loc[bpp.bin_num == 3299][['tphys', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'sep', 'RRLO_1', 'RRLO_2', 'evol_type']]

bpp.loc[(bpp.evol_type == 8) & (bpp.kstar_1.isin([7,9])) & (bpp.kstar_2 != 7) & (bpp.kstar_2 != 9), "event"]
