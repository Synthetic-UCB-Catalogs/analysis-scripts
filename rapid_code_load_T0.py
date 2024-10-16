import os, sys
import pandas as pd
import numpy as np
import h5py as h5

#######################################################################
### Functions to load T0.hdf5 data
#######################################################################

def load_T0_data(ifilepath, code=None, **kwargs):
    """Read in standardized output Common Core data and select at DWD formation

    Note: all codes should save their T0 dataframes as hdf, for speed and storage
    
    Parameters
    ----------
    ifilepath : `str`
        ifilepath to T0 datafile 

    code : 
        name of code (only required for non-standard codes, ComBinE and SEVN)

    **kwargs
        metallicty : `float`
            metallicity of the data if code=='SEVN'; this is usually encoded in the path

    Returns
    -------
    dat : `pandas.DataFrame`
        all data in T0 format
        
    header : `pandas.DataFrame`
        header for dat
    """
    if code == "ComBinE":
        col_standard = ["ID","UID","SID","time","event",
                        "semiMajor","eccentricity","type1",
                        "mass1","radius1","Teff1","massHeCore1",
                        "type2","mass2","radius2","Teff2","massHeCore2",
                        "envBindEn","massCOCore1","massCOCore2",
                        "radiusRL1","radiusRL2","period",
                        "luminosity1","luminosity2"]

        # load the data
        dat = pd.read_csv(ifilepath, skiprows=6, names=col_standard)
        lines_number = 6
        with open(ifilepath) as input_file:
            head = [next(input_file) for _ in range(lines_number)]
            T0_info = head[4].replace(" ", "").split(",")
        
            header_info = {"cofVer" : float(T0_info[0]), 
                           "cofLevel": T0_info[1],
                           "cofExtension": "None", 
                           "bpsName": T0_info[3],
                           "bpsVer": T0_info[4], 
                           "contact": T0_info[5], 
                           "NSYS": int(T0_info[6]), 
                           "NLINES": int(T0_info[7]),
                           "Z": float(T0_info[8].replace("\n",""))}
    elif code == "SEVN":
        metallicity = kwargs.pop('metallicity')
        col_standard = ["ID","UID","time","event","semiMajor","eccentricity",
                        "type1","mass1","radius1","Teff1","massHecore1",
                        "type2","mass2","radius2","Teff2","massHecore2"]
        #read in the data with the columns
        dat = pd.read_csv(ifilepath, skiprows=3, names=col_standard)

        #read in the T0 info in the header
        lines_number = 3
        with open(ifilepath) as input_file:
            head = [next(input_file) for _ in range(lines_number)]
            T0_info = head[1].replace(" ", "").split(",")
            header_info = {"cofVer" : float(T0_info[0]), 
                           "cofLevel": T0_info[1],
                           "cofExtension": "None", 
                           "bpsName": T0_info[3],
                           "bpsVer": T0_info[4], 
                           "contact": T0_info[5], 
                           "NSYS": int(T0_info[6]), 
                           "NLINES": int(T0_info[7]),
                           "Z": metallicity}

    #elif code in ["COMPAS", "COSMIC", "SeBa", "BSE"]:
    else:
        with pd.HDFStore(ifilepath) as hdf_store:
            header_info = hdf_store.get_storer('data').attrs.metadata
            dat = hdf_store.get('data')

    header = pd.DataFrame.from_dict([header_info])
    return dat, header

def load_IC(ifilepath):
    """Read in initial data
    
    Parameters
    ----------
    ifilepath : `str`
        name of file including path

    Returns
    -------
    ICs : `pandas.DataFrame`
        all ICs at Zero Age Main Sequence
    """
    try:
        ICs = pd.read_csv(filename, skiprows=1, names=["mass1", "mass2", "period"])
    except:
        ICs = pd.read_csv(filename, skiprows=1, names=["mass1", "mass2", "eccentricity"])

    return ICs


#######################################################################
### Functions to convert from original code outputs into T0.hdf5 format
#######################################################################

def convert_COSMIC_data_to_T0(ifilepath, metallicity, outputpath=None, hdf5_filename="COSMIC_T0.hdf5"):
    """Read in COSMIC data and convert to L0
    
    Parameters
    ----------
    ifilepath : `str`
        name of file including path
        
    metallicity : `float`
        metallicity of the population

    Returns
    -------
    dat : `pandas.DataFrame`
        all data in T0 format
        
    header : `pandas.DataFrame`
        header for dat
    """

    # load the data
    dat = pd.read_hdf(ifilepath, key="bpp")
    bcm = pd.read_hdf(ifilepath, key="bcm")
    
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

    # convert evol_type to event
    dat["event"] = np.zeros(len(dat))
    dat.loc[dat.evol_type == 1, "event"] = -1
    dat.loc[(dat.evol_type == 2) & (dat.kstar_1.shift() < dat.kstar_1), "event"] = 11
    dat.loc[(dat.evol_type == 2) & (dat.kstar_2.shift() < dat.kstar_2), "event"] = 12
    dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 > 1) & (dat.RRLO_2 < 1), "event"] = 31
    dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 < 1) & (dat.RRLO_2 > 1), "event"] = 32
    dat.loc[(dat.evol_type == 3) & (dat.RRLO_1 > 1) & (dat.RRLO_2 > 1), "event"] = 33
    dat.loc[(dat.evol_type == 4) & (dat.kstar_1.isin([7,9])) & (dat.kstar_2 != 7) & (dat.kstar_2 != 9), "event"] = 41
    dat.loc[(dat.evol_type == 4) & (dat.kstar_2.isin([7,9])) & (dat.kstar_1 != 7) & (dat.kstar_1 != 9), "event"] = 42
    dat.loc[(dat.evol_type == 4) & (dat.kstar_1.isin([7,9])) & (dat.kstar_2.isin([7,9])), "event"] = 43
    dat.loc[(dat.evol_type == 5), "event"] == 52
    dat.loc[(dat.evol_type == 6) & ((dat.RRLO_1 > 1) | (dat.RRLO_2 > 1)), "event"] == 52
    dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 > 1) & (dat.RRLO_2 < 1), "event"] == 511
    dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 < 1) & (dat.RRLO_2 > 1), "event"] == 512
    dat.loc[(dat.evol_type == 7) & (dat.RRLO_1 > 1) & (dat.RRLO_2 > 1), "event"] == 513
    dat.loc[(dat.evol_type == 8) & (dat.kstar_1.isin([7,9])) & (dat.kstar_2 != 7) & (dat.kstar_2 != 9), "event"] = 41
    dat.loc[(dat.evol_type == 8) & (dat.kstar_2.isin([7,9])) & (dat.kstar_1 != 7) & (dat.kstar_1 != 9), "event"] = 42
    dat.loc[(dat.evol_type == 8) & (dat.kstar_1.isin([7,9])) & (dat.kstar_2.isin([7,9])), "event"] = 43
    dat.loc[(dat.evol_type == 15) & (dat.UID.isin(bn_1_cc)), "event"] = 212
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

    # convert kstar types to system states
    dat["type1"] = np.zeros(len(dat))
    dat["type2"] = np.zeros(len(dat))

    dat.loc[dat.kstar_1.isin([0,1]), "type1"] = 121
    dat.loc[dat.kstar_1 == 2, "type1"] = 122
    dat.loc[dat.kstar_1 == 3, "type1"] = 123
    dat.loc[dat.kstar_1 == 4, "type1"] = 124
    dat.loc[dat.kstar_1 == 5, "type1"] = 1251
    dat.loc[dat.kstar_1 == 6, "type1"] = 1252
    dat.loc[dat.kstar_1 == 7, "type1"] = 131
    dat.loc[dat.kstar_1 == 8, "type1"] = 132
    dat.loc[dat.kstar_1 == 9, "type1"] = 133
    dat.loc[dat.kstar_1 == 10, "type1"] = 21
    dat.loc[dat.kstar_1 == 11, "type1"] = 22
    dat.loc[dat.kstar_1 == 12, "type1"] = 23
    dat.loc[dat.kstar_1 == 13, "type1"] = 3
    dat.loc[dat.kstar_1 == 14, "type1"] = 4
    dat.loc[dat.kstar_1 == 15, "type1"] = -1

    dat.loc[dat.kstar_2.isin([0,1]), "type2"] = 121
    dat.loc[dat.kstar_2 == 2, "type2"] = 122
    dat.loc[dat.kstar_2 == 3, "type2"] = 123
    dat.loc[dat.kstar_2 == 4, "type2"] = 124
    dat.loc[dat.kstar_2 == 5, "type2"] = 1251
    dat.loc[dat.kstar_2 == 6, "type2"] = 1252
    dat.loc[dat.kstar_2 == 7, "type2"] = 131
    dat.loc[dat.kstar_2 == 8, "type2"] = 132
    dat.loc[dat.kstar_2 == 9, "type2"] = 133
    dat.loc[dat.kstar_2 == 10, "type2"] = 21
    dat.loc[dat.kstar_2 == 11, "type2"] = 22
    dat.loc[dat.kstar_2 == 12, "type2"] = 23
    dat.loc[dat.kstar_2 == 13, "type2"] = 3
    dat.loc[dat.kstar_2 == 14, "type2"] = 4
    dat.loc[dat.kstar_2 == 15, "type2"] = -1    

    # get bin_num/UID into progressiv ID
    ID = np.arange(0, len(dat.UID.unique()), 1)
    UID_counts = dat.UID.value_counts().sort_index()
    dat["ID"] = np.repeat(ID, UID_counts)
    
    dat = dat[["ID","UID","time","event",
               "semiMajor","eccentricity","type1",
               "mass1","radius1","Teff1","massHeCore1",
               "type2","mass2","radius2","Teff2","massHeCore2"]]

    header = {"cofVer" : 1.0, 
                   "cofLevel": "L0",
                   "cofExtension": "None", 
                   "bpsName": "COSMIC",
                   "bpsVer": "3.4.8", 
                   "contact": "kbreivik@andrew.cmu.edu", 
                   "NSYS": len(dat.UID.unique()), 
                   "NLINES": len(dat),
                   "Z": metallicity}
                       
    # Save in hdf5 format
    if outputpath is None:
        outputpath = os.path.split(ifilepath)[0]
    ofilepath = os.path.join(outputpath, hdf5_filename)
    dat.to_hdf(ofilepath, key='data', mode='w')
    with pd.HDFStore(ofilepath) as hdf_store:
        hdf_store.put('data', dat, format='table') 
        hdf_store.get_storer('data').attrs.metadata = header
    return dat # typically not needed, but possibly good for testing

def Eggleton_Roche_lobe(q, sep):
    """Use the Eggleton Formula to calculate the Roche factor

    Parameters
    ----------
    q : `float`, `array`
        mass ratio where m1/m2 gives the factor for r1
        and m2/m1 gives the factor for r2

    sep : `float, `array`
        orbital semimajor axis

    Returns
    -------
    Roche_lobe : `float`, `array`
        Radius of the Roche lobe in units of 
        the supplied semimajor axis
    """
    Roche_fac = 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))

    Roche_lobe = sep * Roche_fac
    return Roche_lobe


def convert_SeBa_data_to_T0(ifilepath, metallicity, outputpath=None, hdf5_filename="SeBa_T0.hdf5"):
    """Read in SeBa data and select at DWD formation

    Parameters
    ----------
    ifilepath : `str`
        name of file including path

    Returns
    -------
    dat : `pandas.DataFrame`
        all data in T0 format

    header : `pandas.DataFrame`
        header for dat
    """
    # load data
    dat = pd.read_csv(ifilepath, sep="\s+",
        names=["UID", "SID", "mass_transfer_type", "time", "semiMajor", "eccentricity",
               "stellar_indentity1", "star_type1", "mass1", "radius1", "Teff1", "massHeCore1",
               "stellar_indentity2", "star_type2", "mass2", "radius2", "Teff2", "massHeCore2"])

    # compute the Roche radii at all times
    dat["RRLO_1"] = Eggleton_Roche_lobe(dat["mass1"]/dat["mass2"], dat["semiMajor"])
    dat["RRLO_2"] = Eggleton_Roche_lobe(dat["mass2"]/dat["mass1"], dat["semiMajor"])

    # convert UID to ID
    ID = np.arange(0, len(dat.UID.unique()), 1)
    UID_counts = dat.UID.value_counts().sort_index()
    dat['ID'] = np.repeat(ID, UID_counts)

    # convert star_types to L0
    dat["type1"] = np.zeros(len(dat))
    dat["type2"] = np.zeros(len(dat))

    dat.loc[dat.star_type1 == 1, "type1"] = 5
    dat.loc[dat.star_type2 == 2, "type2"] = 6
    dat.loc[dat.star_type1 == 3, "type1"] = 121
    dat.loc[dat.star_type1 == 5, "type1"] = 122
    dat.loc[dat.star_type1 == 6, "type1"] = 123
    dat.loc[dat.star_type1 == 7, "type1"] = 124
    dat.loc[dat.star_type1 == 8, "type1"] = 125
    dat.loc[dat.star_type1 == 10, "type1"] = 131
    dat.loc[dat.star_type1 == 11, "type1"] = 132
    dat.loc[dat.star_type1 == 12, "type1"] = 21
    dat.loc[dat.star_type1 == 13, "type1"] = 22
    dat.loc[dat.star_type1 == 14, "type1"] = 23
    dat.loc[dat.star_type1 == 18, "type1"] = 3
    dat.loc[dat.star_type1 == 19, "type1"] = 4
    dat.loc[dat.star_type1 == 20, "type1"] = -1

    dat.loc[dat.star_type2 == 1, "type2"] = 5
    dat.loc[dat.star_type2 == 2, "type2"] = 6
    dat.loc[dat.star_type2 == 3, "type2"] = 121
    dat.loc[dat.star_type2 == 5, "type2"] = 122
    dat.loc[dat.star_type2 == 6, "type2"] = 123
    dat.loc[dat.star_type2 == 7, "type2"] = 124
    dat.loc[dat.star_type2 == 8, "type2"] = 125
    dat.loc[dat.star_type2 == 10, "type2"] = 131
    dat.loc[dat.star_type2 == 11, "type2"] = 132
    dat.loc[dat.star_type2 == 12, "type2"] = 21
    dat.loc[dat.star_type2 == 13, "type2"] = 22
    dat.loc[dat.star_type2 == 14, "type2"] = 23
    dat.loc[dat.star_type2 == 18, "type2"] = 3
    dat.loc[dat.star_type2 == 19, "type2"] = 4
    dat.loc[dat.star_type2 == 20, "type2"] = -1


    dat["event"] = np.zeros(len(dat))
    # convert mass transfer events to L0 events

    #### Should only use SID for these collections
    dat.loc[dat.time == 0.0, "event"] = -1
    dat.loc[(dat.star_type1.shift() < dat.star_type1), "event"] = 11
    dat.loc[(dat.star_type2.shift() < dat.star_type2), "event"] = 12
    dat.loc[(dat.SID == 3) & (dat.RRLO_1 > 1), "event"] = 31
    dat.loc[(dat.SID == 3) & (dat.RRLO_2 > 1), "event"] = 32

    dat.loc[(dat.event.shift() == 31) & (dat.SID == 2), "event"] = 41
    dat.loc[(dat.event.shift() == 32) & (dat.SID == 2), "event"] = 42

    dat.loc[(dat.SID.isin([5,9])) & (dat.RRLO_1 > 1), "event"] = 511
    dat.loc[(dat.SID.isin([5,9])) & (dat.RRLO_2 > 1), "event"] = 512
    dat.loc[(dat.SID == 6), "event"] = 513

    # 4 is contact
    dat.loc[(dat.SID == 4), "event"] = 53

    # 7 is a merger
    dat.loc[(dat.SID == 7), "event"] = 52

    dat.loc[(dat.time == max(dat.time)), "event"] = 81
    dat.loc[(dat.time == max(dat.time)) & (dat.star_type1 > 11) & (dat.star_type2 > 11), "event"] = 82
    dat.loc[(dat.time == max(dat.time)) & (dat.semiMajor == 0.0), "event"] = 84


    dat = dat[["ID","UID","SID","time","event",
               "semiMajor","eccentricity","type1",
               "mass1","radius1","Teff1","massHeCore1",
               "type2","mass2","radius2","Teff2","massHeCore2"]]

    header_info = {"cofVer" : 1.0,
                   "cofLevel": "L0",
                   "cofExtension": "None",
                   "bpsName": "SeBa",
                   "bpsVer": "XX",
                   "contact": "toonen@uva.nl",
                   "NSYS": len(dat.UID.unique()),
                   "NLINES": len(dat),
                   "Z": metallicity}

    header = pd.DataFrame.from_dict([header_info])

    # Save in hdf5 format
    if outputpath is None:
        outputpath = os.path.split(ifilepath)[0]
    ofilepath = os.path.join(outputpath, hdf5_filename)
    dat.to_hdf(ofilepath, key='data', mode='w')
    with pd.HDFStore(ofilepath) as hdf_store:
        hdf_store.put('data', dat, format='table') 
        hdf_store.get_storer('data').attrs.metadata = header
    return dat # typically not needed, but possibly good for testing


def convert_BSE_data_to_T0(ifilepath, metallicity, outputpath=None, hdf5_filename="BSE_T0.hdf5"):
    """Read in BSE data and select at DWD formation
    
    Parameters
    ----------
    ifilepath : `str`
        name of file including path

    Returns
    -------
    dat : `pandas.DataFrame`
        all data in T0 format
        
    header : `pandas.DataFrame`
        header for dat
    """
    # load data
    try:
        cols = ["UID", "time", "kstar_1", "kstar_2", "mass1", "mass2", "period"]
        dat = pd.read_csv(ifilepath, sep='\s+',
            names=cols
        )
    except Error: 
        cols = ["UID", "time", "kstar_1", "kstar_2", "mass1", "mass2", "period", "eccentricity"]
        dat = pd.read_csv(ifilepath, sep='\s+',
            names=cols
        )

    if type(dat.iloc[0].UID) == str:
        dat = pd.read_csv(ifilepath, sep='\s+',
        names=cols, header=1
        )
        
    dat["semiMajor"] = ((dat.period/365.25)**2 * (dat.mass1 + dat.mass2))**(1/3) * 214.94
    if "eccentricity" not in dat.columns:
        dat["eccentricity"] = np.zeros(len(dat))
        
    # convert kstar types to system states
    dat["type1"] = np.zeros(len(dat))
    dat["type2"] = np.zeros(len(dat))

    dat.loc[dat.kstar_1.isin([0,1]), "type1"] = 121
    dat.loc[dat.kstar_1 == 2, "type1"] = 122
    dat.loc[dat.kstar_1 == 3, "type1"] = 123
    dat.loc[dat.kstar_1 == 4, "type1"] = 124
    dat.loc[dat.kstar_1 == 5, "type1"] = 1251
    dat.loc[dat.kstar_1 == 6, "type1"] = 1252
    dat.loc[dat.kstar_1 == 7, "type1"] = 131
    dat.loc[dat.kstar_1 == 8, "type1"] = 132
    dat.loc[dat.kstar_1 == 9, "type1"] = 133
    dat.loc[dat.kstar_1 == 10, "type1"] = 21
    dat.loc[dat.kstar_1 == 11, "type1"] = 22
    dat.loc[dat.kstar_1 == 12, "type1"] = 23
    dat.loc[dat.kstar_1 == 13, "type1"] = 3
    dat.loc[dat.kstar_1 == 14, "type1"] = 4
    dat.loc[dat.kstar_1 == 15, "type1"] = -1

    dat.loc[dat.kstar_2.isin([0,1]), "type2"] = 121
    dat.loc[dat.kstar_2 == 2, "type2"] = 122
    dat.loc[dat.kstar_2 == 3, "type2"] = 123
    dat.loc[dat.kstar_2 == 4, "type2"] = 124
    dat.loc[dat.kstar_2 == 5, "type2"] = 1251
    dat.loc[dat.kstar_2 == 6, "type2"] = 1252
    dat.loc[dat.kstar_2 == 7, "type2"] = 131
    dat.loc[dat.kstar_2 == 8, "type2"] = 132
    dat.loc[dat.kstar_2 == 9, "type2"] = 133
    dat.loc[dat.kstar_2 == 10, "type2"] = 21
    dat.loc[dat.kstar_2 == 11, "type2"] = 22
    dat.loc[dat.kstar_2 == 12, "type2"] = 23
    dat.loc[dat.kstar_2 == 13, "type2"] = 3
    dat.loc[dat.kstar_2 == 14, "type2"] = 4
    dat.loc[dat.kstar_2 == 15, "type2"] = -1    

    # get bin_num/UID into progressiv ID
    ID = np.arange(0, len(dat.UID.unique()), 1)
    UID_counts = dat.UID.value_counts().sort_index()
    dat["ID"] = np.repeat(ID, UID_counts)
    dat["SID"] = ""
    dat["radius1"] = ""
    dat["radius2"] = ""
    dat["Teff1"] = ""
    dat["Teff2"] = ""
    dat["massHeCore1"] = ""
    dat["massHeCore2"] = ""
    dat["event"] = ""
    
    
    
    dat = dat[["ID","UID","SID","time","event",
               "semiMajor","eccentricity","type1",
               "mass1","radius1","Teff1","massHeCore1",
               "type2","mass2","radius2","Teff2","massHeCore2"]]

    header_info = {"cofVer" : 1.0, 
                   "cofLevel": "L0",
                   "cofExtension": "None", 
                   "bpsName": "BSE",
                   "bpsVer": "XX", 
                   "contact": "lizw@ynao.ac.cn", 
                   "NSYS": len(dat.UID.unique()), 
                   "NLINES": len(dat),
                   "Z": metallicity}
    
    header = pd.DataFrame.from_dict([header_info])

    # Save in hdf5 format
    if outputpath is None:
        outputpath = os.path.split(ifilepath)[0]
    ofilepath = os.path.join(outputpath, hdf5_filename)
    dat.to_hdf(ofilepath, key='data', mode='w')
    with pd.HDFStore(ofilepath) as hdf_store:
        hdf_store.put('data', dat, format='table') 
        hdf_store.get_storer('data').attrs.metadata = header
    return dat # typically not needed, but possibly good for testing




def convert_COMPAS_data_to_T0(ifilepath, outputpath=None, hdf5_filename="COMPAS_T0.hdf5", testing=False):
    ucb_events_obj = COMPAS_UCB_Events(ifilepath, testing)
    df = ucb_events_obj.getEvents()
    if outputpath is None:
        outputpath = os.path.split(ifilepath)[0]
    ofilepath = os.path.join(outputpath, hdf5_filename)
    df.to_hdf(ofilepath, key='data', mode='w')
    with pd.HDFStore(ofilepath) as hdf_store:
        hdf_store.put('data', df, format='table') 
        hdf_store.get_storer('data').attrs.metadata = df.attrs
    return df # typically not needed, but possibly good for testing

class COMPAS_UCB_Events(object):

    """
    COMPAS has many different output file types, and does not print a chronological set of events.
    To retrieve this for the UCB standardized outputs, it is easiest to process each entry in the 
    output files 'BSE_Switch_Log', 'BSE_RLOF', 'BSE_Supernovae', and 'BSE_System_Parameters' separately, 
    and then combine and reorder these afterwards. Notably, ordering needs to be done on seeds first, 
    and time second. This is done in getEvents().
    
    Star IDs are assigned at the end, COMPAS IDs (i.e SEED) is not used. This is because COMPAS IDs are 
    not guarunteed to be unique (if multiple hdf5 runs are combied together), and there may be some missing 
    if, e.g., the sampled initial conditions of a system were invalid. 
    
    Events require:
    ID UID time event semiMajor eccentricity type1 mass1 radius1 Teff1 massHecore1 type2 mass2 radius2 Teff2 massHeCore2
    """

    def __init__(self, ifilepath, testing=False):
        self.ifilepath = ifilepath
        self.testing = testing
        self.Data = h5.File(ifilepath, 'r')
        self.all_UCB_events = None
        self.initialiaze_header()
        # Calculate the events as prescribed for the UCBs
        self.getUCBEventForStellarTypeChanges()
        self.getUCBEventForSupernova()
        self.getUCBEventForMassTransfer()
        self.getUCBEventForStartAndEndConditions()

        
    def initialiaze_header(self):
        self.header = {
            "cofVer" : 1.0, 
            "cofLevel": "L0",
            "cofExtension": "None", 
            "bpsName": "COMPAS",
            "bpsVer": self.Data['Run_Details']['COMPAS-Version'][()][0].decode('UTF-8'),
            "contact": "reinhold.willcox@gmail.com", 
            "NSYS": 0, 
            "NLINES": 0,
            #"Z": metallicity
        }
                       
    def update_header(self):
        df = self.all_UCB_events
        ids = df.loc[:,'ID']
        self.header.update({
            "NSYS": len(np.unique(ids)),
            "NLINES": len(ids)
        })

    def addEvents(self, uid=None, time=None, event=None, semiMajor=None, eccentricity=None, 
                        stellarType2=None, mass2=None, radius2=None, teff2=None, massHeCore2=None,
                        stellarType1=None, mass1=None, radius1=None, teff1=None, massHeCore1=None, 
                        scrapSeeds=None):
        columns   = [  "UID", "time", "event", "semiMajor", "eccentricity", 
                       "type1", "mass1", "radius1", "Teff1", "massHeCore1", 
                       "type2", "mass2", "radius2", "Teff2", "massHeCore2", 
                       "scrapSeeds" ] 
        data_list = [   uid,   time,   event,   semiMajor,   eccentricity, 
                        stellarType1,   mass1,   radius1,   teff1,   massHeCore1, 
                        stellarType2,   mass2,   radius2,   teff2,   massHeCore2, 
                        scrapSeeds   ] 
    
        # Want to enter data using name keywords, but all of them are required
        bad_params = []
        for ii, param in enumerate(data_list):
            if param is None:
                bad_params.append(columns[ii])
        if len(bad_params) > 0:
            raise Exception("Can't skip any of the required input values. Currently missing: {}".format(', '.join(bad_params)))
        new_events = pd.DataFrame(np.vstack(data_list).T, columns=columns)    
        if self.all_UCB_events is None:
            self.all_UCB_events = new_events
        else:
            self.all_UCB_events = pd.concat([self.all_UCB_events, new_events])

    def getEvents(self): 


        df = self.all_UCB_events                                        # Convert to df for convenience
        
        # Clean up events - remove bad seeds
        allSeeds = df.loc[:,"UID"]                                      # Identify all seeds (or UIDs)
        scrapSeedMask = df.loc[:,"scrapSeeds"] == 1                     # Identify and create mask from the scrapSeeds column
        badSeedMask = np.in1d(allSeeds, allSeeds[scrapSeedMask])        # Create mask for all seeds to be scrapped, including rows with scrappable seeds that were not masked for it
        df = df[~badSeedMask]                                           # Return df without scrapped seeds
        df = df.drop(columns='scrapSeeds')                              # Remove scrap seeds column

        # Reorder the cells, add ID column
        df = df.sort_values(["UID", 'time'])                            # Reorder by uid (seed) first, then time second        
        uid_arr = df.loc[:,"UID"]                                       # Get list of UIDs
        uniq_uid = np.unique(uid_arr)                                   # Get the unique sorted list of UIDs
        uniq_id = np.arange(len(uniq_uid)) + 1                          # Start IDs counter at 1
        dict_uid_id = dict(zip(uniq_uid, uniq_id))                      # Map the UIDs to the IDs
        id_arr = np.vectorize(dict_uid_id.__getitem__)(uid_arr)         # Apply the map to the list of UIDs (with repeats)
        df.insert(0, "ID", id_arr)                                      # Insert the full IDs list at the front of the df
        
        self.all_UCB_events = df                                        # Convert back from df
        self.update_header()                                            # Update the header with new information
        self.all_UCB_events = self.all_UCB_events.reset_index(drop=True)

        self.all_UCB_events.attrs = self.header # Set header for the df
        return self.all_UCB_events

    def verifyAndConvertCompasDataToUcbUsingDict(self, compasData, conversionDict):
        """
        General convenience function to verify and convert compas data arrays to their 
        equivalent values in UCB format, using the dictionaries defined variously below
        """
        try: 
            valid_types = np.array(list(conversionDict.keys()))
            assert np.all(np.in1d(compasData, valid_types))
        except:
            raise Exception('Invalid input array')
        return np.vectorize(conversionDict.get)(compasData)  # Quickly process entire input vector through converter dict

    
    ########################################################################################
    ### 
    ### Convert COMPAS output to UCB events 
    ### 
    ########################################################################################
    
    ############################################
    ### 
    ### BSE_Switch_Log output processing
    ### 
    ############################################
    
    # Stellar type conversion dictionary
    compasStellarTypeToUCBdict = {
        # COMPAS : UCB
        0: 121,
        1: 121,
        2: 122,
        3: 123,
        4: 124,
        5: 1251,
        6: 1252,
        7: 131,
        8: 132,
        9: 133,
        10: 21,
        11: 22,
        12: 23,
        13: 3,
        14: 4,
        15: -1,
        16: 9,  # CHE star - doesn't exist in UCB notation
    }
    
    def getUCBEventForStellarTypeChanges(self):
        
        SL = self.Data["BSE_Switch_Log"]
        
        # Direct output
        uid = SL["SEED"][()]
        time = SL["Time"][()]
        semiMajorAxis = SL["SemiMajorAxis"][()]
        eccentricity = SL["Eccentricity"][()]
        mass1 = SL["Mass(1)"][()]
        mass2 = SL["Mass(2)"][()]
        radius1 = SL["Radius(1)"][()]
        radius2 = SL["Radius(2)"][()]
        teff1 = SL["Teff(1)"][()]
        teff2 = SL["Teff(2)"][()]
        massHeCore1 = SL["Mass_He_Core(1)"][()]
        massHeCore2 = SL["Mass_He_Core(2)"][()]
        stellarType1 = self.verifyAndConvertCompasDataToUcbUsingDict(SL["Stellar_Type(1)"][()], self.compasStellarTypeToUCBdict)
        stellarType2 = self.verifyAndConvertCompasDataToUcbUsingDict(SL["Stellar_Type(2)"][()], self.compasStellarTypeToUCBdict)
        
        # Indirect output
        whichStar = SL['Star_Switching'][()]
        event = 10 + whichStar
        scrapSeeds = np.zeros_like(uid).astype(bool) 
    
        self.addEvents(  uid=uid, time=time, event=event, semiMajor=semiMajorAxis, eccentricity=eccentricity, 
                         stellarType1=stellarType1, mass1=mass1, radius1=radius1, teff1=teff1, massHeCore1=massHeCore1, 
                         stellarType2=stellarType2, mass2=mass2, radius2=radius2, teff2=teff2, massHeCore2=massHeCore2,
                         scrapSeeds=scrapSeeds)
    
    ############################################
    ### 
    ### BSE_RLOF output processing
    ### 
    ############################################
    
    def getUCBEventForMassTransfer(self):
    
        MT = self.Data["BSE_RLOF"]
        # Need to distinguish:
        # 1. Start of RLOF
        # 2. End of RLOF
        # 3. CEE events
        # 4. Mergers
        # 5. Contact phase (do we do this?)
        
        # Direct output
        uid = MT["SEED"][()]
        time = MT["Time>MT"][()]
        semiMajorAxis = MT["SemiMajorAxis>MT"][()]
        eccentricity = MT["Eccentricity>MT"][()]
        mass1 = MT["Mass(1)>MT"][()]
        mass2 = MT["Mass(2)>MT"][()]
        radius1 = MT["Radius(1)>MT"][()]
        radius2 = MT["Radius(2)>MT"][()]
        teff1 = MT["Teff(1)"][()]
        teff2 = MT["Teff(2)"][()]
        massHeCore1 = MT["Mass_He_Core(1)"][()]
        massHeCore2 = MT["Mass_He_Core(2)"][()]
        stellarType1 = self.verifyAndConvertCompasDataToUcbUsingDict(MT["Stellar_Type(1)>MT"][()], self.compasStellarTypeToUCBdict)
        stellarType2 = self.verifyAndConvertCompasDataToUcbUsingDict(MT["Stellar_Type(2)>MT"][()], self.compasStellarTypeToUCBdict)
        
        # Indirect output
        isRlof1 = MT["RLOF(1)>MT"][()] == 1
        isRlof2 = MT["RLOF(2)>MT"][()] == 1
        wasRlof1 = MT["RLOF(1)<MT"][()] == 1
        wasRlof2 = MT["RLOF(2)<MT"][()] == 1
        isCEE = MT["CEE>MT"][()] == 1
        isMerger = MT["Merger"][()] == 1
        scrapSeeds = np.zeros_like(uid).astype(bool) # TODO Scrap seeds if start of RLOF for both in the same timestep - is there any way to work with these??
    
        # Every mask in allmasks corresponds to an event in allevents
        allmasks = []
        allevents = []
        
        # Could make an events array of Nones, and then fill as they come up
        # The advantage of this is that for timesteps that qualify as 2 different events, you overwrite the wrong one...
        # Maybe I should just include the flags explicitly, that's probably more careful
        # So instead of doing a bunch of final MT timesteps and overwriting with any CEEs, I just include ~CEE in the condition.
        
        # 1. Start of RLOF.
        maskStartOfRlof1 = isRlof1 & ~wasRlof1 
        maskStartOfRlof2 = isRlof2 & ~wasRlof2 
    
        for ii in range(2):
            whichStar = ii+1 # either star 1 or 2
            allmasks.append([ maskStartOfRlof1, maskStartOfRlof2 ][ii])
            allevents.append( 3*10 + whichStar )
    
        # 2. End of RLOF
        maskFirstMtInParade1 = isRlof1 & ~wasRlof1
        maskFirstMtInParade2 = isRlof2 & ~wasRlof2
        for ii in range(2):
            whichStar = ii+1 # either star 1 or 2
            maskFirstMtInParade = [ maskFirstMtInParade1, maskFirstMtInParade2][ii]
            idxLastMtInParade = maskFirstMtInParade.nonzero()[0] - 1
            maskLastMtInParade = np.zeros_like(uid).astype(bool)
            maskLastMtInParade[idxLastMtInParade] = True
            allmasks.append(maskLastMtInParade & ~isCEE)
            allevents.append(4*10 + whichStar)

        # 3. CEE events - Process each CEE donor separately, plus double CEE for both
        maskAnyCEE = isCEE 
        whichStar = 1
        maskCEE1 = isRlof1 & ~isRlof2 & maskAnyCEE
        allmasks.append(maskCEE1)
        allevents.append(510 + whichStar)
        whichStar = 2
        maskCEE2 = isRlof2 & ~isRlof1 & maskAnyCEE
        allmasks.append(maskCEE2)
        allevents.append(510 + whichStar)
        whichStar = 3
        maskCEE3 = isRlof2 & isRlof1 & maskAnyCEE
        allmasks.append(maskCEE3)
        allevents.append(510 + whichStar)

        # 4. Mergers
        allmasks.append(isMerger)
        allevents.append(52)
        
    
        # 5. Contact phase (do we do this?)
        # TBD
        
        
        # Use masks to add all the events back into the array
        for mask, event in zip(allmasks, allevents):
    
            self.addEvents(  uid=uid[mask], time=time[mask], event=event*np.ones_like(uid)[mask], semiMajor=semiMajorAxis[mask], eccentricity=eccentricity[mask], 
                             stellarType1=stellarType1[mask], mass1=mass1[mask], radius1=radius1[mask], teff1=teff1[mask], massHeCore1=massHeCore1[mask], 
                             stellarType2=stellarType2[mask], mass2=mass2[mask], radius2=radius2[mask], teff2=teff2[mask], massHeCore2=massHeCore2[mask],
                             scrapSeeds=scrapSeeds[mask])
    
    ############################################
    ### 
    ### BSE_Supernovae output processing
    ### 
    ############################################
    
    # Supernova conversion dictionary
    compasSupernovaToUCBdict = {
        # COMPAS : UCB
        1:   2, # CCSN
        2:   3, # ECSN
        4:   4, # PISN
        8:   5, # PPISN
        16:  7, # USSN 
        32:  8, # AIC 
        64:  1, # Type Ia
        128: 9, # HeSD
    }
    UCB_SN_TYPE_DIRECT_COLLAPSE = 6 # Need to separately treat failed SNe (i.e direct collapse)
    
    def getUCBEventForSupernova(self):
    
        SN = self.Data["BSE_Supernovae"]
        
        # Direct output
        uid = SN["SEED"][()]
        time = SN["Time"][()]
        semiMajorAxis = SN["SemiMajorAxis"][()]
        eccentricity = SN["Eccentricity"][()]
        mass1 = SN["Mass(1)"][()]
        mass2 = SN["Mass(2)"][()]
        radius1 = SN["Radius(1)"][()]
        radius2 = SN["Radius(2)"][()]
        teff1 = SN["Teff(1)"][()]
        teff2 = SN["Teff(2)"][()]
        massHeCore1 = SN["Mass_He_Core(1)"][()]
        massHeCore2 = SN["Mass_He_Core(2)"][()]
        stellarType1 = self.verifyAndConvertCompasDataToUcbUsingDict(SN["Stellar_Type(1)"][()], self.compasStellarTypeToUCBdict)
        stellarType2 = self.verifyAndConvertCompasDataToUcbUsingDict(SN["Stellar_Type(2)"][()], self.compasStellarTypeToUCBdict)
        
        # Indirect output
        whichStar = SN["Supernova_State"][()] 
        assert np.all(np.in1d(whichStar, np.array([1, 2, 3]))) # TODO: need to address State 3 systems somehow.
        scrapSeeds = whichStar == 3 # need to remove these seeds at the end
        if self.testing:
            if np.any(scrapSeeds):
                print("There were {} simultaneous SNe".format(np.sum(scrapSeeds)))
                print("Their seeds were:")
                print("[" + ", ".join(uid[scrapSeeds].astype(str)) + "]")
    
        snType = self.verifyAndConvertCompasDataToUcbUsingDict(SN["SN_Type(SN)"][()], self.compasSupernovaToUCBdict)
        fb = SN['Fallback_Fraction(SN)'][()]
        snType[fb == 1] == self.UCB_SN_TYPE_DIRECT_COLLAPSE
        event = 2*100 + whichStar*10 + snType    
        
        self.addEvents(  uid=uid, time=time, event=event, semiMajor=semiMajorAxis, eccentricity=eccentricity, 
                         stellarType1=stellarType1, mass1=mass1, radius1=radius1, teff1=teff1, massHeCore1=massHeCore1, 
                         stellarType2=stellarType2, mass2=mass2, radius2=radius2, teff2=teff2, massHeCore2=massHeCore2,
                         scrapSeeds=scrapSeeds)
    
    ############################################
    ### 
    ### BSE_System_Parameters output processing
    ### 
    ############################################
    
    # End condition conversion dictionary
    compasOutcomeToUCBdict = {
        # COMPAS : UCB
        1:  -1, # simulation completed?
        2:  9, # error
        3:  1, # max time reached
        4:  1, # max timesteps reached, kind of the same as above
        5: -1, # no timesteps?
        6: -1, # timesteps exhausted?
        7: -1, # timesteps not consumed?
        8:  9, # error
        9:  9, # error
        10:  -1, # time exceeded dco merger time?
        11:  -1, # stars touching ?
        12:  4, # merger
        13: 4, # merger
        14: 2, # dco formed
        15: 2, # dwd formed
        16: 4, # massless remnant
        17: 3, # unbound    
    }
    
    # -1 applies if the compas output description is unclear, just toss these seeds for now.
    # Q: how does 85 happen and not 84? Wouldn't simulations stop at 84?
        
    # UCBs
    # 81 - max time reached
    # 82 - both components are compact remnants - RTW: including WDs?
    # 83 - the binary system is dissociated
    # 84 - only one object is left (e.g. due to a merger or because the companion has been disrupted)
    # 85 - nothing left (both components are massless remnants)
    # 89 - other: a terminating condition different from any previous one
    
    # COMPAS
    # Simulation completed = 1     
    # Evolution stopped because an error occurred = 2
    # Allowed time exceeded = 3
    # Allowed timesteps exceeded = 4
    # SSE error for one of the constituent stars = 5
    # Error evolving binary = 6
    # Time exceeded DCO merger time = 7
    # Stars touching = 8
    # Stars merged = 9
    # Stars merged at birth = 10
    # DCO formed = 11
    # Double White Dwarf formed = 12
    # Massless Remnant formed = 13
    # Unbound binary = 14
    
    def getUCBEventForStartAndEndConditions(self):
        
        SP = self.Data["BSE_System_Parameters"]
        
        ###########################################################################
        ### 
        ### Do End Condition first, it's more similar to the above routines
        ### 
        ###########################################################################

        # Direct output
        uid = SP["SEED"][()]
        time = SP["Time"][()]
        semiMajorAxis = SP["SemiMajorAxis"][()]
        eccentricity = SP["Eccentricity"][()]
        mass1 = SP["Mass(1)"][()]
        mass2 = SP["Mass(2)"][()]
        radius1 = SP["Radius(1)"][()]
        radius2 = SP["Radius(2)"][()]
        teff1 = SP["Teff(1)"][()]
        teff2 = SP["Teff(2)"][()]
        massHeCore1 = SP["Mass_He_Core(1)"][()]
        massHeCore2 = SP["Mass_He_Core(2)"][()]
        stellarType1 = self.verifyAndConvertCompasDataToUcbUsingDict(SP["Stellar_Type(1)"][()], self.compasStellarTypeToUCBdict)
        stellarType2 = self.verifyAndConvertCompasDataToUcbUsingDict(SP["Stellar_Type(2)"][()], self.compasStellarTypeToUCBdict)

        # Indirect output
        evolStatus = self.verifyAndConvertCompasDataToUcbUsingDict(SP["Evolution_Status"][()], self.compasOutcomeToUCBdict)
        assert np.all(np.in1d(evolStatus, np.array([1, 2, 3, 4, 5, 9, -1]))) 
        scrapSeeds = evolStatus == -1 # -1 means I don't understand the compas outcome
        if self.testing:
            if np.any(scrapSeeds):
                print("There were {} strange evolutionary outcomes".format(np.sum(scrapSeeds)))
                print("Their seeds were:")
                print("[" + ", ".join(uid[scrapSeeds].astype(str)) + "]")
        event = 8*10 + evolStatus
        
        self.addEvents(  uid=uid, time=time, event=event, semiMajor=semiMajorAxis, eccentricity=eccentricity, 
                         stellarType1=stellarType1, mass1=mass1, radius1=radius1, teff1=teff1, massHeCore1=massHeCore1, 
                         stellarType2=stellarType2, mass2=mass2, radius2=radius2, teff2=teff2, massHeCore2=massHeCore2,
                         scrapSeeds=scrapSeeds)

        ###########################################################################
        ### 
        ### Do First timestep next
        ### 
        ###########################################################################

        # Direct output
        uid = SP["SEED"][()]
        time = np.zeros_like(uid)
        semiMajorAxis = SP["SemiMajorAxis@ZAMS"][()]*215.032 # Rsol
        eccentricity = SP["Eccentricity@ZAMS"][()]
        mass1 = SP["Mass@ZAMS(1)"][()]
        mass2 = SP["Mass@ZAMS(2)"][()]
        radius1 = SP["Radius@ZAMS(1)"][()] 
        radius2 = SP["Radius@ZAMS(2)"][()]
        teff1 = SP["Teff@ZAMS(1)"][()] 
        teff2 = SP["Teff@ZAMS(2)"][()]
        massHeCore1 = np.zeros_like(uid) 
        massHeCore2 = np.zeros_like(uid) 
        stellarType1 = self.verifyAndConvertCompasDataToUcbUsingDict(SP["Stellar_Type@ZAMS(1)"][()], self.compasStellarTypeToUCBdict)
        stellarType2 = self.verifyAndConvertCompasDataToUcbUsingDict(SP["Stellar_Type@ZAMS(2)"][()], self.compasStellarTypeToUCBdict)

        # Indirect output
        scrapSeeds = uid < 0 # dummy value, should be all False
        event = 13*np.ones_like(uid)  # Both stars change stellar type - kind of true
        
        self.addEvents(  uid=uid, time=time, event=event, semiMajor=semiMajorAxis, eccentricity=eccentricity, 
                         stellarType1=stellarType1, mass1=mass1, radius1=radius1, teff1=teff1, massHeCore1=massHeCore1, 
                         stellarType2=stellarType2, mass2=mass2, radius2=radius2, teff2=teff2, massHeCore2=massHeCore2,
                         scrapSeeds=scrapSeeds)


if __name__ == "__main__":

    # First argument is an integer, specifying which code base # this should be cleaned up later
    # Second argument is the filepath to the dataoutput that needs to be converted
    # Third (optional) argument is the filepath to the new T0-formatted data file
    which = int(sys.argv[1])
    ifilepath = sys.argv[2]
    try:
        ofilepath = sys.argv[3]
    except:
        ofilepath = None

    print(ifilepath)
    if not os.path.isfile(ifilepath):
        raise Exception("Not a valid input file path")
        
    metallicity = 0.02
    if which == 0:
        print("trying cosmic")
        convert_COSMIC_data_to_T0(ifilepath, metallicity, ofilepath)
    elif which == 1:
        print("trying seba")
        convert_SeBa_data_to_T0(ifilepath, metallicity, ofilepath)
    elif which == 2:
        print("trying bse")
        convert_BSE_data_to_T0(ifilepath, metallicity, ofilepath)
    elif which == 3:
        print("trying compas")
        convert_COMPAS_data_to_T0(ifilepath, ofilepath)
    else:
        print("Not set yet")
