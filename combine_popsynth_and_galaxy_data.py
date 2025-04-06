#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:05:24 2024

@author: alexey
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import legwork as lw
from astropy import units as u

sys.path.insert(0, os.path.abspath(os.path.join(os.environ["PWD"])))
from utils import MWConsts, fGW_from_A, calculateSeparationAfterSomeTime, inspiral_time, chirp_mass
from galaxy_models.draw_samples_from_galaxy import createGalaxyModel
from population_synthesis.get_popsynth_lisa_dwds import getLisaDwdProperties

# This environment variable must be set by the user to point to the root of the google drive folder
ROOT_DIR = os.environ['UCB_ROOT_DIR']
SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']

# Units are MSun, kpc, Gyr
# FOR VISUALS, WE USE A RIGHT-HANDED SYSTEM, WITH X POINTING AT THE SUN, Z ALIGNED WITH THE SPIN, AND THE ORIGIN AT THE GALACTIC CENTER


def matchDwdsToGalacticPositions(
        pathToPopSynthData=None,
        pathToGalacticSamples=None,
        useLegworkMask=False, # If true, calculate LISA visibile sources according to Legwork                                                                                                   
        applyInitialLisaBandFilter=True,
    ):

    # Import Population Synthesis data 
    DWDs, Z = getLisaDwdProperties(pathToPopSynthData, applyInitialLisaBandFilter=applyInitialLisaBandFilter)
    m1,   m2,  a_birth, t_DWD = DWDs
    #Msun Msun Rsun     Myr 
    nBinaries = m1.shape[0]

    # WD mass-radius relation
    WD_MR_relation = UnivariateSpline(*np.loadtxt(os.path.join(ROOT_DIR, 'population_synthesis/WDMassRadiusRelation.dat')).T, k=4, s=0)
    r1 = WD_MR_relation(m1) # R_sun 
    r2 = WD_MR_relation(m2) # R_sun

    # Import Galactic position samples
    galaxyModel = createGalaxyModel('Besancon', Z, saveOutput=False) 
    drawn_samples = galaxyModel.DrawSamples(nBinaries)

    #drawn_samples = getGalaxySamples(pathToGalacticSamples)
    #b_gal, l_gal, d_gal, t_birth, which_component = drawn_samples[:,:nBinaries]
    b_gal, l_gal, d_gal, t_birth, _ = drawn_samples #[:,:nBinaries]
    
    # Calculate present day properties of WDs 
    dt = t_birth - t_DWD # Myr
    a_today = calculateSeparationAfterSomeTime(m1, m2, a_birth, dt) # R_sun
    fGW_today = fGW_from_A(m1, m2, a_today) # Hz

    dwd_properties = np.vstack((m1, m2, a_today, fGW_today, b_gal, l_gal, d_gal))

    # if desired, use legwork to apply additional masking for LISA-visible DWDs
    if useLegworkMask:

        cutoff = 2 # user should set this

        sources = lw.source.Source(
            m_1   = m1 *u.Msun, 
            m_2   = m2 *u.Msun, 
            dist  = d_gal *u.kpc, 
            f_orb = fGW_today/2 *u.Hz,
            ecc   = np.zeros(nBinaries)
            )

        snr = sources.get_snr(verbose=True)
        print(np.sort(snr))
        mask_detectable_sources = sources.snr > cutoff
        print(np.sum(mask_detectable_sources))
        mask = mask_detectable_sources
    else:
        # Remove systems that would have merged by now 
        mask_mergers = a_today < r1+r2
        f_min = 1e-4 # Minimum frequency bin for binary to reach LISA within Hubble time
        f_max = 1e-1 # Maximum frequency bin for LISA 
        mask_lisa_band = (fGW_today > f_min) & (fGW_today < f_max)
        mask = ~mask_mergers #& mask_lisa_band


    masked_dwds = dwd_properties[:,mask]
    masks = [mask, mask_lisa_band]
    return masked_dwds, masks




if __name__ == "__main__":
    matchDwdsToGalacticPositions( pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"),
                                 #pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5",
                                  useLegworkMask=False, 
                                  applyInitialLisaBandFilter=False)
