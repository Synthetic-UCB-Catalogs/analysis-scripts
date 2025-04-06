from astropy import units as u
from astropy import constants as const
import numpy as np

# Common astrophysical functions and constants
# Units
DaysToSec = float(str((u.d/u.s).decompose()))
YearToSec = float(str((u.yr/u.s).decompose()))
MyrToSec = 1.e6*YearToSec
KmToCM = float(str((u.km/u.cm).decompose()))
MSunToG = ((const.M_sun/u.g).decompose()).value
RSunToCm = ((const.R_sun/u.cm).decompose()).value

# Constants
GNewtCGS = ((const.G*u.g*((u.s)**2)/(u.cm)**3).decompose()).value
CLightCGS = ((const.c*(u.s/u.cm)).decompose()).value
RGravSun = 2.*GNewtCGS*MSunToG/CLightCGS**2
RhoConv = (MSunToG/RSunToCm**3)


# RTW: Do we need both?
# MW parameters
use_alt_params = False
if not use_alt_params:
    MWConsts = {'MGal': 6.43e10,  # From Licquia and Newman 2015
                'MBulge1': 6.1e9,  # From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8,  # From Robin+ 2012, metal-poor bulge
                # From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'MHalo': 1.4e9,
                'RGalSun': 8.2,  # Bland-Hawthorn, Gerhard 2016
                'ZGalSun': 0.025  # Bland-Hawthorn, Gerhard 2016
                }
else:
    MWConsts = {'MGal': 6.43e10,  # From Licquia and Newman 2015
                'MBulge1': 6.1e9,  # From Robin+ 2012, metal-rich bulge
                'MBulge2': 2.6e8,  # From Robin+ 2012, metal-poor bulge
                # From Deason+ 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3426D/abstract)
                'MHalo': 1.4e9,
                'RGalSun': 8.122,  # GRAVITY Collaboration et al. 2018
                'ZGalSun': 0.028  # Bennett & Bovy, 2019
                }
MWConsts.update({'MBulge': MWConsts['MBulge1'] + MWConsts['MBulge2']})
# 3D components of the Sun’s velocity (U ; V ;W ) =(12:9; 245:6; 7:78) km s^1 (Drimmel & Poggio 2018)
# The GW inspiral time in megayears

def P_from_A(m1, m2, a):
    # m1, m2 [Msun], a [Rsun]
    # P in years
    prefactor = 3153.23 #np.power((u.AU/u.Rsun),(3/2)).decompose()
    return prefactor*np.sqrt(a**3 / (m1+m2))

def fGW_from_A(m1, m2, a):
    # m1, m2 [Msun], a [Rsun]
    # fGW in Hz = 2/Porb
    unit_conversion = 3153.23 / (365*24*60*60) #np.power((u.AU/u.Rsun),(3/2)).decompose() * yr->s
    return 2*np.sqrt(m2+m2)*np.power(a, -3/2) *unit_conversion

def inspiral_time(m1, m2, a):
    # m1, m2 [Msun], a [Rsun]
    # return tau [Myr]
    prefactor = 150.2044149170097 #solMass3 Myr / solRad4 #(5*np.power(const.c, 5) / (256*np.power(const.G, 3))).to(u.M_sun**3 *u.yr /u.R_sun**4 )
    return prefactor *np.power(a, 4) / (m1*m2*(m1 + m2)) 

# Separation after some amount of time of inspiral
def calculateSeparationAfterSomeTime(m1, m2, a_birth, dt):
    tau_GW = inspiral_time(m1, m2, a_birth)
    return a_birth * np.power(1 - dt/tau_GW, 0.25)

def chirp_mass(m1, m2):
    return np.power(m1*m2, 3/5)/np.power(m1+m2, 1/5)

if __name__ == "__main__":
    print()

