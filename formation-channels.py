import pandas as pd
import numpy as np
import h5py as h5
import rapid_code_load_T0 as load


def select_ZAMS_WDMS_DWD(d):
    '''Selects the WDMS and DWD populations at the formation of the first and second white dwarfs

    Params
    ------
    d : `pandas.DataFrame`
        contains T0 data for binaries as specified by BinCodex

    Returns
    -------
    ZAMS : `pandas.DataFrame`
        T0 columns for Zero Age Main Sequence binaries

    WDMS : `pandas.DataFrame`
        T0 columns for WDMS binaries at the formation of the 1st WD

    DWD : `pandas.DataFrame`
        T0 columns for DWD binaries at the formation of the 2nd WD
    '''

    ZAMS = d.groupby('ID', as_index=False).first()

    WDMS1 = d.loc[((d.type1.isin([21,22,23]) & (d.type2 == 121))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    WDMS2 = d.loc[((d.type2.isin([21,22,23]) & (d.type1 == 121))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    WDMS = pd.concat([WDMS1, WDMS2])
    DWD = d.loc[(d.type1.isin([21,22,23])) & (d.type2.isin([21,22,23])) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    return ZAMS, WDMS, DWD