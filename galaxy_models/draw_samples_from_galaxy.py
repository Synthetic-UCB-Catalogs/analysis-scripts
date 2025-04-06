#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:41:43 2024

@author: alexey, reinhold
"""

import os
import numpy as np
import h5py as h5

from galaxy_models.besancon_model import BesanconModel
# TODO: implement other models

# This environment variable must be set by the user to point to the root of the google drive folder
SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']

def createGalaxyModel(galaxyModelName=None, *args, **kwargs):
    if galaxyModelName == "Besancon":
        return BesanconModel("Besancon", *args, **kwargs)
    else:
        raise Exception("Model not configured")

def getGalaxySamples(pathToExistingData=None):
   fullPath = os.path.abspath(pathToExistingData)
   if os.path.isfile(fullPath): 
       print("Importing galaxy data from existing file: {}".format(fullPath))
       drawn_samples = h5.File(fullPath, 'r')['data']
       return drawn_samples
   else:
       raise Exception("pathToExistingData does not exist: {}".format(fullPath))



if __name__ == "__main__":
    createGalaxyModel(galaxyModelName='Besancon', nSamples=1e8, saveOutput=True) 
