"""
Module for interpolating white dwarf masses, radii and various other important
stellar parameters. The code currently uses the Bedard et al. (2020) grid of
evolutionary models, by default. Though it is still possible to use the older
Fontaine et al. (2001) grid if required.
"""
__author__ = "Mark Hollands"
__email__ = "M.Hollands.1@warwick.ac.uk"

from .MR_relation import *
