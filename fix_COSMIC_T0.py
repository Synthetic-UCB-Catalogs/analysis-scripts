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

COSMIC_T0_basic[COSMIC_T0_basic.ID == 3299]

dat = pd.read_hdf("data/pilot_runs_raw_data/COSMIC/basic.h5", key='bpp')

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
dat["event"] = np.zeros(len(dat))

dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 > 1) & (dat.RRLO_2 < 1), "event"] = 31


dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 > 1) & (dat.RRLO_2 < 1), "event"] = 511
dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 < 1) & (dat.RRLO_2 > 1), "event"] = 512
dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 > 1) & (dat.RRLO_2 > 1), "event"] = 513

dat.loc[dat.UID == 3299][['time', 'kstar_1', 'kstar_2', 'mass1', 'mass1', 'semiMajor', 'RRLO_1', 'RRLO_2', 'evol_type', 'event']]

bpp.loc[(bpp.evol_type == 8) &(bpp.sep > 0) & (bpp.RRLO_1 ==0.99) & (bpp.RRLO_2 ==0.99)][['bin_num', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'sep', 'RRLO_1', 'RRLO_2', 'evol_type']]

bpp.loc[(bpp.evol_type == 7) & (bpp.RRLO_2 < 1) & (bpp.RRLO_1 > 1)]

bpp.loc[bpp.bin_num == 3299][['tphys', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2', 'sep', 'RRLO_1', 'RRLO_2', 'evol_type']]

bpp.loc[(bpp.evol_type == 8) & (bpp.kstar_1.isin([7,9])) & (bpp.kstar_2 != 7) & (bpp.kstar_2 != 9), "event"]
