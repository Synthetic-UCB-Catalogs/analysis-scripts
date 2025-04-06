#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey, reinhold
"""

import os
import h5py as h5
import numpy as np
import scipy.stats as ss
from utils import MWConsts


class GalaxyModel():
    def __init__(self, name, Z=0.0142, fnameOutput=None, saveOutput=False, singleComponentToUse=None): 
        self.name = name
        self.Z = Z
        self.fnameOutput = os.path.abspath(self.GetFnameOutput(fnameOutput))
        self.saveOutput = saveOutput

        self.SetModelParameters()
        self.components = self.CreateComponents()
        self.componentWeights = self.CalculateGalacticComponentMassFractions()
        if singleComponentToUse is not None:
            self.componentWeights = np.zeros_like(self.componentWeights) 
            self.componentWeights[singleComponentToUse] = 1

    # Functions passed to children
    def SetModelParameters(self):
        pass

    def CreateComponents(self):
        pass

    def CalculateGalacticComponentMassFractions(self):
        pass

    # Functions that apply for all Galaxy models
    def GetFnameOutput(self, fnameOutput):
        if fnameOutput is not None:
            return fnameOutput
        else:
            return "SampledGalacticLocations_{}_{}.h5".format(self.name, self.Z)

    def DrawSamples(self, nSystems):

        print("Generating new data")
        nSystems = int(nSystems) # in case of weird input type issues

        if not self.saveOutput:                     # Just draw the subsamples, if the number is too large, it will crash
            return self.DrawSubSamples(nSystems, include_component=True)
        else:                                       # Saving potentially large data, need to chunk it
            print("Saving to: {}".format(self.fnameOutput))

            # Need to loop over large chunks of samples, since the datafile gets very large otherwise
            nSystemsPerChunk = int(1e7)
            nChunks = nSystems // nSystemsPerChunk
            print(nChunks)

            with h5.File(self.fnameOutput, "w") as f:
                dset = f.create_dataset('data', dtype='float32', shape=(4, nSystems), chunks=(4, nSystemsPerChunk))

                for iChunk in range(nChunks):
                    print("iChunk = {}".format(iChunk))
                    drawn_samples = self.DrawSubSamples(nSystemsPerChunk)
                    start = nSystemsPerChunk *(iChunk)   
                    stop  = nSystemsPerChunk *(iChunk+1)
                    print(start, stop)
                    dset[:,start:stop] = drawn_samples
            return

    
    def DrawSubSamples(self, nSystems, include_component=False):

        # the component weight is the mass fraction multiplied by the metallicity weight
        component_weights = self.componentWeights

        empty_output_array_all_components = np.zeros(
            (self.nComponents, nSystems))
        r_matrix = empty_output_array_all_components.copy()
        z_matrix = empty_output_array_all_components.copy()
        t_matrix = empty_output_array_all_components.copy()

        # For each Galacitc Component, get the metallicity weight factor, and randomly draw the position and age for every binary from the popsynth
        for ii in range(self.nComponents):
            # Get the metallicity factor for the Component weight
            mean_FeH, std_FeH = self.components[ii].mean_FeH, self.components[ii].std_FeH
            # This is a constant extra weight for this Component
            metallicity_weight = ss.norm.pdf(x=self.Z, loc=mean_FeH, scale=std_FeH)
            component_weights[ii] = component_weights[ii] * metallicity_weight

            # Get the r,z values randomly sampled, weighted by the density distribution of the component
            # Note that these are locally defined r,z, and are not necessarily consistent between components
            rzDensityGrid = self.components[ii].GetDensityGrid()
            rVals, zVals, mass_weights = rzDensityGrid
            indexChoices = np.random.choice(
                len(rVals), size=nSystems, p=mass_weights/np.sum(mass_weights))
            r_matrix[ii] = np.take(rVals, indexChoices, axis=0)
            z_matrix[ii] = np.take(zVals, indexChoices, axis=0)

            # Sample the age uniformly from the min and max values for this Component
            t_matrix[ii] = np.random.uniform(
                low=self.components[ii].ageMin, 
                high=self.components[ii].ageMax, 
                size=nSystems)

        # Choose which component to use for each binary according to a weighted random draw.
        which_component = np.random.choice(
            self.nComponents, size=nSystems, p=component_weights/np.sum(component_weights))
        r = np.choose(which_component, r_matrix)
        z = np.choose(which_component, z_matrix)
        t = np.choose(which_component, t_matrix)

        # Calculate related quantities
        bld_gal = np.zeros((r.shape[0], 3)).T
        for ii in range(self.nComponents):
            mask = which_component == ii
            bld_gal[:,mask] = self.components[ii].ConvertToGalacticFrame(r[mask], z[mask])
        b_gal, l_gal, d_gal = bld_gal
        t_birth = t 

        if include_component:
            drawn_samples = np.vstack((b_gal, l_gal, d_gal, t_birth, which_component))
        else:
            drawn_samples = np.vstack((b_gal, l_gal, d_gal, t_birth))
        return drawn_samples



class GalaxyComponent():
    def __init__(self,
                 componentName, ageMin, ageMax,
                 rMax, zMax, mean_FeH, std_FeH,
                 RhoFunction, RotationFunction):

        self.componentName = componentName
        self.ageMin = ageMin
        self.ageMax = ageMax
        self.rMax = rMax
        self.zMax = zMax
        self.mean_FeH = mean_FeH
        self.std_FeH = std_FeH
        self.RhoFunction = RhoFunction
        self.Rotate = RotationFunction 

    # Functions passed to children
    def RhoFunction(self, r, z):
        pass

    # Functions that apply for all Galaxy Components
    def ConvertToGalacticFrame(self, r, z):
        theta = 2*np.pi*np.random.uniform(size=r.size)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        x, y, z = self.Rotate(x, y, z)  # Rotate here, if necessary

        x_rel = x - MWConsts['RGalSun']
        y_rel = y
        z_rel = z + MWConsts['ZGalSun']
        r_rel = np.sqrt(x_rel*x_rel + y_rel*y_rel + z_rel*z_rel)
        b_gal = np.arcsin(z_rel/r_rel)
        l_gal = np.arccos(x_rel/(np.sqrt((r_rel)**2 - (z_rel)**2)))
        l_gal[y_rel < 0] = 2*np.pi - l_gal[y_rel < 0]
        d_gal = r_rel
        return b_gal, l_gal, d_gal

    # These are universal to all Galaxy components
    def GetDensityGrid(self, nPoints=100):

        rArray = np.linspace(0, self.rMax, nPoints)
        zArray = np.linspace(-self.zMax, self.zMax, nPoints)
        dr = self.rMax / (nPoints - 1)
        dz = self.zMax*2 / (nPoints - 1)

        rGrid, zGrid = np.meshgrid(rArray, zArray, indexing='ij')
        rGrid = rGrid.flatten()
        zGrid = zGrid.flatten()
        rhoGrid = self.RhoFunction(rGrid, zGrid)
        integrand = rhoGrid*rGrid * 2*np.pi * dr * dz
        return np.vstack((rGrid, zGrid, integrand))

    def GetVolumeIntegral(self, nPoints=100):
        return np.sum(self.GetDensityGrid(nPoints=nPoints))

