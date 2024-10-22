import pandas as pd
import numpy as np
import h5py as h5
import rapid_code_load_T0 as load


def select_evolutionary_states(d):
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


def first_interaction_channels(d):
    '''Split out the different types of channels that could occur
    in the first interaction

    Parameters
    ----------
    d : `pandas.DataFrame`
        contains T0 data for binaries as specified by BinCodex

    Returns
    -------
    
    '''

    RLO_1 = d.loc[d.event.isin([31, 32, 511, 512, 513, 52, 53])].groupby('ID', as_index=False).first()

    SMT_1 = RLO_1.loc[RLO_1.event == 31].ID
    SMT_2 = RLO_1.loc[RLO_1.event == 32].ID
    CE_1 = RLO_1.loc[RLO_1.event == 511].ID
    CE_2 = RLO_1.loc[RLO_1.event == 512].ID
    DCCE = RLO_1.loc[RLO_1.event == 513].ID
    # need to check on whether systems go 511 --> 52. 
    merger = RLO_1.loc[RLO_1.event.isin([52, 53])].ID

    nonRLO = d.loc[~d.ID.isin(RLO_1.ID)].ID.unique()

    first_RLO = {'SMT_1': SMT_1,
                 'SMT_2': SMT_2,
                 'CE_1': CE_1,
                 'CE_2': CE_2,
                 'DCCE': DCCE,
                 'merger': merger,
                 'nonRLO': nonRLO,}
    return first_RLO

    

def select_final_state_ids(d):
    '''Sets the final state based on the evolution

    Parameters
    ----------
    d : `pandas.DataFrame`
        T0 data with all events listed

    Returns
    -------

    '''


def select_channels(d):
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
    channels : `Dictionary`
        A dictionary with the following keys and items:
        
        SMT1_SMT1 : `numpy.array`
            an array containing the IDs of systems which undergo 2 stable 
            mass transfer events with both initiated by the primary
        
        SMT2_SMT2 : `numpy.array`
            an array containing the IDs of systems which undergo 2 stable 
            mass transfer events with both initiated by the secondary
    
        CE1_CE1 : `numpy.array`
            an array containing the IDs of systems which undergo 2 common 
            envelope events with both initiated by the primary
        
        CE2_CE2 : `numpy.array`
            an array containing the IDs of systems which undergo 2 common 
            envelope events with both initiated by the secondary
        
        CE1 : `numpy.array`
            an array containing the IDs of systems which undergo 1 common 
            envelope event initiated by the primary

        CE2 : `numpy.array`
            an array containing the IDs of systems which undergo 1 common 
            envelope event initiated by the secondary

        CE3 : `numpy.array`
            an array containing the IDs of systems which undergo 1 common 
            envelope event initiated by the primary & secondary at the 
            same time

        SMT1 : `numpy.array`
            an array containing the IDs of systems which undergo 1 stable 
            mass transfer event initiated by the primary

        SMT2 : `numpy.array`
            an array containing the IDs of systems which undergo 1 stable 
            mass transfer event initiated by the secondary

        RLO_3_or_more_other : `numpy.array`
            an array containing the IDs of systems which undergo 3 or more
            RLO events
        
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
    d = d.loc[~((d.type1.isin([21,22,23])) & ((d.type2.isin([21,22,23]))))]
    RLO_all = d.loc[(d.event.isin([31,32,511,512,513,53]))]
    No_RLO = d.loc[~d.ID.isin(RLO_all.ID)].ID.unique()

    #RLO_2 = RLO_all.loc[RLO_all.ID.value_counts() == 2]
    ind_RLO_2 = np.where(RLO_all.ID.value_counts() == 2, RLO_all.ID.value_counts().index, -1)
    ind_RLO_2 = ind_RLO_2[ind_RLO_2 >= 0]
    RLO_2 = RLO_all.loc[RLO_all.ID.isin(ind_RLO_2)]

    # select systems with SMT that shifts to CE based on donor evolution
    SMT1_CE1 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event.isin([511, 513, 53])))].ID.unique()
    SMT2_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 32) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event.isin([512, 513, 53])))].ID.unique()

    # select systems with SMT that shifts to CE based on donor evolution
    CE1_SMT1 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event.isin([511, 513, 53])) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 31))].ID.unique()
    CE2_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event.isin([512, 513, 53])) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()

    # select systems with two SMTs from the either the primary or secondary
    SMT1_SMT1 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 31))].ID.unique()
    SMT2_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 32) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    
    # select systems with two CEs from the either the primary or secondary
    CE1_CE1 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 511))].ID.unique()
    CE2_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 512) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()
    
    
    # select out single mass transfers (these are likely to be CE becuase
    # the single mass transfer only occurs for mergers or secondaries that
    # don't evolve beyond the MS due to low masses)
    ind_RLO_1 = np.where(RLO_all.ID.value_counts() == 1, RLO_all.ID.value_counts().index, -1)
    ind_RLO_1 = ind_RLO_1[ind_RLO_1 >= 0]
    RLO_1 = RLO_all.loc[RLO_all.ID.isin(ind_RLO_1)]
    
    CE1 = d.loc[(d.ID.isin(RLO_1.ID)) & (d.event == 511)].ID.unique()
    CE1_merge = d.loc[(d.ID.isin(CE1)) & (d.event == 52)].ID.unique()
    CE1_survive = np.setxor1d(CE1, CE1_merge)
    
    CE2 = d.loc[(d.ID.isin(RLO_1.ID)) & (d.event == 512)].ID.unique()
    CE3 = d.loc[(d.ID.isin(RLO_1.ID)) & (d.event == 513)].ID.unique()

    SMT1 = d.loc[(d.ID.isin(RLO_1.ID)) & (d.event == 31)].ID.unique()
    SMT2 = d.loc[(d.ID.isin(RLO_1.ID)) & (d.event == 32)].ID.unique()


    # select the systems which go through SMT which evolves to CE based on Stellar evolution
    ind_RLO_3 = np.where(RLO_all.ID.value_counts() == 3, RLO_all.ID.value_counts().index, -1)
    ind_RLO_3 = ind_RLO_3[ind_RLO_3 >= 0]
    RLO_3 = RLO_all.loc[RLO_all.ID.isin(ind_RLO_3)]

    evCE1_SMT2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                              (RLO_3.groupby('ID', as_index=False).nth(1).event.isin([511, 513, 53])) & 
                              (RLO_3.groupby('ID', as_index=False).nth(2).event == 32))].ID.unique()
    evCE1_CE2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                              (RLO_3.groupby('ID', as_index=False).nth(1).event.isin([511])) & 
                              (RLO_3.groupby('ID', as_index=False).nth(2).event.isin([512])))].ID.unique()
    SMT1_evCE2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                               (RLO_3.groupby('ID', as_index=False).nth(1).event == 32) & 
                               (RLO_3.groupby('ID', as_index=False).nth(2).event.isin([512, 513, 53])))].ID.unique()
    CE1_evCE2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event.isin([511, 513, 53])) & 
                              (RLO_3.groupby('ID', as_index=False).nth(1).event == 32) & 
                              (RLO_3.groupby('ID', as_index=False).nth(2).event.isin([512, 513, 53])))].ID.unique()
    SMT1_CE2_SMT2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                              (RLO_3.groupby('ID', as_index=False).nth(1).event.isin([512, 513, 53])) & 
                              (RLO_3.groupby('ID', as_index=False).nth(2).event == 32))].ID.unique()
    SMT1_CE2_CE2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                              (RLO_3.groupby('ID', as_index=False).nth(1).event.isin([512, 513, 53])) & 
                              (RLO_3.groupby('ID', as_index=False).nth(2).event.isin([512, 513, 53])))].ID.unique()

    SMT1_SMT1_CE2 = RLO_3.loc[((RLO_3.groupby('ID', as_index=False).nth(0).event == 31) & 
                              (RLO_3.groupby('ID', as_index=False).nth(1).event == 31) & 
                              (RLO_3.groupby('ID', as_index=False).nth(2).event.isin([512, 513, 53])))].ID.unique()
    
    bID3 = []
    for d in [evCE1_SMT2, evCE1_CE2, SMT1_evCE2, CE1_evCE2, SMT1_CE2_SMT2, SMT1_CE2_CE2, SMT1_SMT1_CE2]:
        bID3.extend(d)
    RLO_3_other = RLO_3.loc[~RLO_3.ID.isin(bID3)].ID.unique()

    
    ind_RLO_4 = np.where(RLO_all.ID.value_counts() >= 4, RLO_all.ID.value_counts().index, -1)
    ind_RLO_4 = ind_RLO_4[ind_RLO_4 >= 0]
    RLO_4_or_more_other = RLO_all.loc[RLO_all.ID.isin(ind_RLO_4)]
    #RLO_4_or_more_other = RLO_all.loc[RLO_all.ID.value_counts() > 3].ID.unique()
    # select systems which merge due to failed CE
    #failed_CE = d.loc[d.event == 52].ID.unique()
    
    
    other = {'SMT1_SMT1' : SMT1_SMT1,
             'SMT2_SMT2' : SMT2_SMT2,
             'CE1_CE1' : CE1_CE1,
             'CE2_CE2' : CE2_CE2,
             'SMT1_CE1' : SMT1_CE1,
             'SMT2_CE2' : SMT2_CE2,
             'CE1_SMT1' : CE1_SMT1,
             'CE2_SMT2' : CE2_SMT2,
             'CE1_merge' : CE1_merge,
             'CE1_survive' : CE1_survive,
             'CE2' : CE2,
             'CE3' : CE3,
             'SMT1' : SMT1,
             'SMT2' : SMT2,
             'evCE1_SMT2' : evCE1_SMT2,
             'evCE1_CE2' : evCE1_CE2,
             'SMT1_evCE2' : SMT1_evCE2,
             'CE1_evCE2' : CE1_evCE2,
             'SMT1_CE2_SMT2' : SMT1_CE2_SMT2,
             'SMT1_CE2_CE2' : SMT1_CE2_CE2,
             'SMT1_SMT1_CE2' : SMT1_SMT1_CE2,
             'RLO_3_other' : RLO_3_other,
             'RLO_4_or_more_other' : RLO_4_or_more_other}#,
    #         #'failed_CE' : failed_CE}

    other_IDs = []
    for key in other.keys():
        other_IDs.extend(other[key])
    other_IDs = np.unique(other_IDs)
   
    # filter out the other IDs     
    RLO_2 = RLO_2.loc[~(RLO_2.ID.isin(other_IDs))]
    
    SMT1_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    SMT1_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 31) & 
                           (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()
    CE1_SMT2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 32))].ID.unique()
    CE1_CE2 = RLO_2.loc[((RLO_2.groupby('ID', as_index=False).nth(0).event == 511) & 
                          (RLO_2.groupby('ID', as_index=False).nth(1).event == 512))].ID.unique()

    channels = other
    channels['SMT1_SMT2'] = SMT1_SMT2
    channels['SMT1_CE2'] = SMT1_CE2
    channels['CE1_SMT2'] = CE1_SMT2
    channels['CE1_CE2'] = CE1_CE2
    channels['No_RLO'] = No_RLO
    return channels






