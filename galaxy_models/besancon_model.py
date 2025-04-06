#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey, reinhold
"""

import numpy as np
from utils import MWConsts
from galaxy_models.galaxy_model_class import GalaxyModel, GalaxyComponent


# For Besancon model, see the full description at https://model.obs-besancon.fr/modele_descrip.php

# Here we use:
# 1) eps values from Robin+2003 (https://www.aanda.org/articles/aa/abs/2003/38/aa3188/aa3188.html)
# 2) Density weights are from Czekaj+2014 (https://www.aanda.org/articles/aa/full_html/2014/04/aa22139-13/aa22139-13.html)

class BesanconModel(GalaxyModel):
    # Redefine the necessary functions

    def SetModelParameters(self):
        # TODO: add units
        self.componentNames  =           ['YoungThinDisk',  'ThinDisk2',  'ThinDisk3',  'ThinDisk4',  'ThinDisk5',  'ThinDisk6',  'ThinDisk7',  'ThickDisk',      'Halo',   'Bulge']                                 
        self.ageMins         =  np.array([0,                0.15,         1,            2,            3,            5,            7,            10,               14,       8],               dtype='float64')*1000  
        self.ageMaxs         =  np.array([0.15,             1.,           2.,           3.,           5.,           7.,           10.,          10.,              14.,      10],              dtype='float64')*1000  
        self.rMaxs           =  np.array([30,               30,           30,           30,           30,           30,           30,           30,               50,       5],               dtype='float64')       
        self.zMaxs           =  np.array([4,                4,            4,            4,            4,            4,            4,            8,                50,       3],               dtype='float64')       
        self.mean_FeH        =  np.array([0.01,             0.03,         0.03,         0.01,         -0.07,        -0.14,        -0.37,        -0.78,            -1.78,    0.00],            dtype='float64')       
        self.std_FeH         =  np.array([0.12,             0.12,         0.10,         0.11,         0.18,         0.17,         0.20,         0.30,             0.50,     0.40],            dtype='float64')       
        self.dFedR           =  np.array([-0.07,            -0.07,        -0.07,        -0.07,        -0.07,        -0.07,        -0.07,        0,                0,        0],               dtype='float64')       
        self.Rho0Array       =  np.array([1.888e-3,         5.04e-3,      4.11e-3,      2.84e-3,      4.88e-3,      5.02e-3,      9.32e-3,      2.91e-3,          9.2e-6],  dtype='float64')  # Czekaj2014
        self.epsArrayThin    =  np.array([0.0140,           0.0268,       0.0375,       0.0551,       0.0696,       0.0785,       0.0791],      dtype='float64')                                                     
        self.nComponents = len(self.componentNames)

    def CreateComponents(self):
        components = []
        for iComponent in range(self.nComponents):
            if (self.componentNames[iComponent] == 'Bulge'):
                RotationFunction = self.BulgeRotationFunction
            else:
                RotationFunction = self.DefaultRotationFunction

            # Append the component, including the Rho Function
            components.append(GalaxyComponent(
                componentName=self.componentNames[iComponent],
                ageMin=self.ageMins[iComponent],
                ageMax=self.ageMaxs[iComponent],
                rMax=self.rMaxs[iComponent],
                zMax=self.zMaxs[iComponent],
                mean_FeH=self.mean_FeH[iComponent],
                std_FeH=self.std_FeH[iComponent],
                RhoFunction=self.RhoFunctions(iComponent),
                RotationFunction=RotationFunction))
        return components

    def RhoFunctions(self, iComponent):
        # Define RhoFunction(r,z), for each Component in the Besancon model, weights are defined later
        # Young thin disc
        if iComponent == 0:
            hPlus = 5
            hMinus = 3
            eps = self.epsArrayThin[iComponent]

            def RhoFunction(r, z):
                aParam = np.sqrt(np.power(r, 2) + np.power(z/eps, 2))
                Rho = np.exp(-np.power(aParam/hPlus, 2)) - \
                    np.exp(-np.power(aParam/hMinus, 2))
                return Rho
        # Thin disk - other components
        elif (iComponent >= 1) and (iComponent <= 6):
            hPlus = 2.53
            hMinus = 1.32
            eps = self.epsArrayThin[iComponent]

            def RhoFunction(r, z):
                aParam = np.sqrt(np.power(r, 2) + np.power(z/eps, 2))
                Rho = np.exp(-np.sqrt((0.25 + np.power(aParam/hPlus, 2)))) - \
                    np.exp(-np.sqrt(0.25 + np.power(aParam/hMinus, 2)))
                return Rho
        # Thick disc
        elif (iComponent == 7):
            xl = 0.4
            RSun = 8
            hR = 2.5
            hz = 0.8

            def RhoFunction(r, z):
                mask = np.abs(z) <= xl
                Rho = np.zeros_like(r)
                Rho[mask] = ((np.exp(-(r - RSun)/hR)) * (1 -
                             ((1/hz)/(xl*(2 + xl/hz))) * (np.power(z, 2))))[mask]
                Rho[~mask] = ((np.exp(-(r - RSun)/hR)) * ((np.exp(xl/hz)) /
                              (1 + xl/(2*hz))) * (np.exp(-np.abs(z)/hz)))[~mask]
                return Rho
        # Halo
        elif (iComponent == 8):
            ac = 0.5
            eps = 0.76
            RSun = 8

            def RhoFunction(r, z):
                aParam = np.sqrt(np.power(r, 2) + np.power(z/eps, 2))
                aLarger = np.maximum(aParam, ac)
                Rho = np.power(aLarger/RSun, -2.44)
                return Rho
        # Bulge
        elif (iComponent == 9):
            # Bulge coordinates are weird, the x-axis is defined as the axis of rotation
            x0 = 1.59
            yz0 = 0.424  # -- y0 and z0 are equal, use that to sample the bulge stars in the coordinates where z-axis is aligned with the x-axis of the bugle
            Rc = 2.54
            N = 13.7
            # See Robin Tab 5
            # Orientation angles:
            # α (angle between the bulge major axis and the line perpendicular to the Sun – Galactic Center line),
            # β (tilt angle between the bulge plane and the Galactic plane) and
            # γ (roll angle around the bulge major axis);

            def RhoFunction(r, z):
                # We assume z is the axis of symmetry, but in the bulge coordinates it is x; use rotation
                xbulge = -z
                rbulge = r
                # Note, bulge is not fully axisymmetric, and minor axes y and z contribute differently to the equation
                # REVISE AND SAMPLE FROM 3D

                rs2 = np.sqrt(np.power(rbulge/yz0, 4) + np.power(xbulge/x0, 4))
                # rParam = rbulge
                mask = r > Rc
                Rho = np.exp(-0.5*rs2)
                Rho[mask] = (np.exp(-0.5*rs2) * np.exp(-0.5 *
                             np.power((rbulge - Rc)/(0.5), 2)))[mask]
                # np.exp(-0.5*rs2)*np.exp(-0.5*((np.sqrt(x**2 + y**2) - Rc)/(0.5))**2)
                return Rho

        return RhoFunction

    def DefaultRotationFunction(self, x, y, z):
        return x, y, z

    def BulgeRotationFunction(self, xp, yp, zp):
        # Bulge coordinates are weird, need to convert from prime values back to normal Galactic frame
        # See Dwek et al. (1995)
        alpha = 78.9*(np.pi/180)
        x = -zp*np.sin(alpha) + xp*np.cos(alpha)
        y =  zp*np.cos(alpha) + xp*np.sin(alpha)
        z = -yp
        return x, y, z

    def CalculateGalacticComponentMassFractions(self):

        componentIntegrals = [component.GetVolumeIntegral()
                              for component in self.components]

        NormCArray = np.zeros(self.nComponents)
        # Halo mass:
        iHalo = np.where(np.in1d(self.componentNames, ['Halo']))[0][0]
        NormCArray[iHalo] = MWConsts['MHalo']/componentIntegrals[iHalo]
        # Bulge mass:
        iBulge = np.where(np.in1d(self.componentNames, ['Bulge']))[0][0]
        NormCArray[iBulge] = MWConsts['MBulge']/componentIntegrals[iBulge]

        # Thin/Thick disc masses:
        # First, get the non-weighted local densities
        RhoTildeArray = np.array([self.components[ii].RhoFunction(
            MWConsts['RGalSun'], MWConsts['ZGalSun']) for ii in range(8)])
        # Then, get the weights so that the local densities are reproduced
        NormArrayPre = self.Rho0Array[:8]/RhoTildeArray
        # Then renormalise the whole thin/thick disc to match the Galactic stellar mass and finalise the weights
        NormCArray[:8] = NormArrayPre*(MWConsts['MGal'] - MWConsts['MBulge'] -
                                       MWConsts['MHalo'])/np.sum(NormArrayPre*componentIntegrals[:8])

        # Compute derived quantities
        # Masses for each component
        componentMasses = NormCArray*componentIntegrals
        # Mass fractions in each component
        componentMassFractions = componentMasses/MWConsts['MGal']

        return componentMassFractions

