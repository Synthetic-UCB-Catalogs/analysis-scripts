import pandas as pd
import numpy as np
import h5py as h5
import rapid_code_load_T0 as load


def select_ZAMS_WDMS_DWD(d):
    '''Selects the WDMS and DWD populations at the formation of the first and second white dwarfs

    Parameters
    ----------
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


def select_channels_simple(d):
    '''Selects out the simple channels with the primary and secondary
    undergoing combinations of stable mass transfer and common envelope
    in addition to systems which don't interact and 'other' systems
    which interact but through different channels

    Parameters
    ----------
    d : `pandas.DataFrame`
        T0 data with all events listed

    Returns
    -------
    No_RLO : `numpy.array`
        an array containing the IDs of systems which do not go through RLO, 
    
    SMT1_SMT2 : `numpy.array`
        an array containing the IDs of systems which experience two 
        stable mass transfer events
        
    SMT1_CE2 : `numpy.array`
        an array containing the IDs of systems which experience 
        stable mass transfer from the primary and a common envelope 
        from the secondary
        
    CE1_SMT2 : `numpy.array`
        an array containing the IDs of systems which experience 
        a common envelope from the primary and stable mass transfer
        from the secondary
    
    CE1_CE2 : `numpy.array`
        an array containing the IDs of systems which experience 
        two common envelopes
        
    other : `numpy.array`
        an array containing the IDs of systems which interact 
        don't experience any of the above channels
    '''

    RLO_all = d.loc[d.event.isin([31,32,511,512,513,53])]
    No_RLO = d.loc[~d.ID.isin(RLO_all.ID)].ID.unique()

    RLO_2 = RLO_all.loc[RLO_all.ID.value_counts() == 2]
    
    SMT1_CE1 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event.isin([511, 513, 53])))].ID.unique()
    SMT2_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 32) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event.isin([512, 513, 53])))].ID.unique()

    other = np.append(SMT1_CE1, SMT2_CE2)
    
    # filter out the 
    RLO_2 = RLO_2.loc[~(RLO_2.ID.isin(other))]
    
    SMT1_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    SMT1_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()
    CE1_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    CE1_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()
    
    return No_RLO, SMT1_SMT2, SMT1_CE2, CE1_SMT2, CE1_CE2, other