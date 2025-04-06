#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %matplotlib inline

# +
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey, reinhold
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.environ["PWD"])))
from combine_popsynth_and_galaxy_data import matchDwdsToGalacticPositions
from utils import chirp_mass

SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']
# -

from galaxy_models.draw_samples_from_galaxy import getGalaxySamples, createGalaxyModel 

def plotSimulatedGalaxySamples(
        pathToExistingData=None,         # (str) Path to existing Galaxy samples, useful for just plotting
        galaxyModelName=None,            # (str) The name of the Galaxy Model to use, if generating new data
        nSamples=1e6,                    # (int) Number of stars to sample - memory issues around 1e8
        Z=0.0142,                        # (float) Metallicity of samples
        fnameOutput=None,                # (str) File name to save (only relevant if saveOutput is true)
        saveOutput=False,                # (bool) Whether to save the output to a file (if new samples are drawn)
        singleComponentToUse=None,       # (int) Number of the single component to model (for visualizations). If None, use full model.
    ):

    # if pathToExistingData specified, plot existing data, otherwise create new Galaxy model and draw fresh samples
    if pathToExistingData is not None:
        drawn_samples = getGalaxySamples(pathToExistingData)
    else: 
        # Create Galaxy model and draw (or import) samples of binary locations and birth times
        galaxyModel = createGalaxyModel(galaxyModelName, Z, fnameOutput, saveOutput, singleComponentToUse) 
        drawn_samples = galaxyModel.DrawSamples(nSamples)

    b_gal, l_gal, d_gal, t_birth, which_component = drawn_samples

    z_gal = d_gal*np.sin(b_gal) 
    x_gal = d_gal*np.cos(b_gal)*np.cos(l_gal)
    y_gal = d_gal*np.cos(b_gal)*np.sin(l_gal)

    cmap = mpl.cm.rainbow
    color = cmap(which_component/10) # color by component
    #color = cmap(np.arange(len(d_gal))/len(d_gal)) # color by position in the samples list - which is ordered by distance from the sun

    # Plot all components together in 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_gal, y_gal, z_gal, s=1, c=color) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_zlim(-30,30)

    # Plot R vs Z for individual components separately
    #fig, axes = plt.subplots(ncols=5, nrows=2)
    #axs = axes.flatten()
    #for ii in range(10):
    #    mask = which_component == ii
    #    label = ['ThinDisk1',  'ThinDisk2',  'ThinDisk3',  'ThinDisk4',  'ThinDisk5',  'ThinDisk6',  'ThinDisk7',  'ThickDisk',      'Halo',   'Bulge'][ii]
    #    ax = axs[ii]
    #    ax.scatter(r_gal[mask], z_gal[mask], s=1, c=color[mask], label=label)
    #    ax.legend(fontsize=12)
    #    ax.set_xlim(0, 30)
    #    ax.set_ylim(-30,30)
    plt.show()
    return fig
    

if __name__ == "__main__":
    plotSimulatedGalaxySamples(galaxyModelName='Besancon', nSamples=1e6, saveOutput=False) #, singleComponentToUse=8) 
    #plotSimulatedGalaxySamples(galaxyModelName='Besancon', nSamples=1e8, saveOutput=True) 
    #plotSimulatedGalaxySamples(galaxyModelName='Besancon', nSamples=1e6, saveOutput=False, fnameOutput="small_test.h5", singleComponentToUse=8) 
    #plotSimulatedGalaxySamples(pathToExistingData="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5")


