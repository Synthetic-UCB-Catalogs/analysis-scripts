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

    # the extra 12 handles BPASS
    WDMS1 = d.loc[((d.type1.isin([21,22,23]) & (d.type2.isin([12, 121])))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    WDMS2 = d.loc[((d.type2.isin([21,22,23]) & (d.type1.isin([12, 121])))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    WDMS = pd.concat([WDMS1, WDMS2])
    DWD = d.loc[(d.type1.isin([21,22,23])) & (d.type2.isin([21,22,23])) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    
    # this handles ComBinE
    if len(DWD) == 0:
        WDMS1 = d.loc[((d.type1 == 2) & (d.type2 == 121)) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
        WDMS2 = d.loc[((d.type2 == 2) & (d.type1 == 121)) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    
        WDMS = pd.concat([WDMS1, WDMS2])
        DWD = d.loc[(d.type1 == 2) & (d.type2 == 2) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    return ZAMS, WDMS, DWD


def first_interaction_channels(d, verbose=False):
    '''Split out the different types of channels that could occur
    in the first interaction

    Parameters
    ----------
    d : `pandas.DataFrame`
        contains T0 data for binaries as specified by BinCodex

    verbose : `bool`
        Sets whether to print out detailed results

    Returns
    -------
    first_RLO : `Dictionary`
        contains a list of IDs for different interaction channels
    '''

    # calculate the time that the star goes back into its Roche lobe the first time
    t_rlo_end_1 = d.loc[d.event.isin([4, 41])].groupby('ID', as_index=False).time.first()
    t_rlo_end_1.columns = ['ID', 't_rlo_end_1']
    
    d = d.merge(t_rlo_end_1, on='ID', how='left')
    d['t_rlo_end_1'] = d['t_rlo_end_1'].fillna(13700.0)
    n_ID = len(d.ID.unique())
    
    
    # filter out everything that happens after first RLO finishes (don't cut on no RLO or mergers)
    d = d.loc[~(d.time > d.t_rlo_end_1)]
    n_ID_2 = len(d.ID.unique())

    if verbose:
        print("check that we have t_rlo_end_1 columns for all IDs")
        print(n_ID_2, n_ID)
    
    # first look for RLO and non-RLO
    RLO_IDs = d.loc[d.event.isin([31, 32, 511, 512, 513, 52, 53])].ID.unique()
    RLO = d.loc[d.ID.isin(RLO_IDs)]
    nonRLO = d.loc[~d.ID.isin(RLO.ID.unique())].ID.unique()

    if verbose:
        print("Total IDs:", len(RLO.ID.unique())+len(nonRLO))
        print("RLO IDs: ", len(RLO.ID.unique()), "No RLO IDs: ",len(nonRLO))
    
    # Select out things that merge by finding mergers 
    RLO_merger_IDs = RLO.loc[(RLO.t_rlo_end_1 == 13700) & (RLO.event == 52) & (RLO.type1.isin([121, 122, 123, 124, 125, 1251, 1252, -1]))].ID.unique()
    RLO_mergers = d.loc[d.ID.isin(RLO_merger_IDs)]
    # remove the mergers from RLO
    RLO = RLO.loc[~(RLO.ID.isin(RLO_merger_IDs))]

    if verbose:
        print("RLO, RLO mergers")
        print(len(RLO.ID.unique()), len(RLO_mergers.ID.unique()))

    # First consider the mergers that occur due to failed CE
    # Could add in evolutionary transition to this but will keep simple for now
    CE_merger = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift() == 511)].ID.unique()
    # remove them from the merger frame
    RLO_mergers = RLO_mergers.loc[~RLO_mergers.ID.isin(CE_merger)]

    if verbose:
        print("RLO, RLO mergers, CE mergers")
        print(len(RLO.ID.unique()), len(RLO_mergers.ID.unique()), len(CE_merger))
        print("Sum of all three")
        print(len(RLO.ID.unique()) + len(RLO_mergers.ID.unique()) + len(CE_merger))        

    # Next count mergers that jump from RLO to merger
    contact_merger = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift() == 31)].ID.unique()
    contact_merger_BSE = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift() == 3)].ID.unique()
    contact_merger_BSE_2 = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift(2) == 3)].ID.unique()
    contact_merger_contact = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift() == 53)].ID.unique()
    contact_merger_contact_2 = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift(3) == 53)].ID.unique()
    contact_merger_kchange_1 = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift(2) == 31)].ID.unique()
    contact_merger_kchange_2 = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift(3) == 31)].ID.unique()
    contact_merger = np.append(contact_merger, contact_merger_BSE)
    contact_merger = np.append(contact_merger, contact_merger_BSE_2)
    contact_merger = np.append(contact_merger, contact_merger_contact)
    contact_merger = np.append(contact_merger, contact_merger_contact_2)
    contact_merger = np.append(contact_merger, contact_merger_kchange_1)
    contact_merger = np.append(contact_merger, contact_merger_kchange_2)
    contact_merger = np.unique(contact_merger)
    # remove them from the merger frame
    RLO_mergers = RLO_mergers.loc[~RLO_mergers.ID.isin(contact_merger)]

    if verbose:
        print("RLO, RLO mergers, CE mergers, contact merger")
        print(len(RLO.ID.unique()), len(RLO_mergers.ID.unique()), len(CE_merger), len(contact_merger))
        print("sum of all four")
        print(len(RLO.ID.unique()) + len(RLO_mergers.ID.unique()) + len(CE_merger) + len(contact_merger))

    # Next count mergers from double core common envelope
    # Could add in evolutionary transition to this but will keep simple for now
    DCCE_merger = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.event.shift() == 513)].ID.unique()
    # remove them from the merger frame
    RLO_mergers = RLO_mergers.loc[~RLO_mergers.ID.isin(DCCE_merger)]


    # Next count MS + MS mergers:
    MS_merger = RLO_mergers.loc[(RLO_mergers.event == 52) & (RLO_mergers.type1 == 121.0) & (RLO_mergers.type2 == 121.0)].ID.unique()
    # remove them from the merger frame
    RLO_mergers = RLO_mergers.loc[~RLO_mergers.ID.isin(MS_merger)]

    if verbose:
        print("RLO, RLO mergers, CE mergers, contact merger, DCCE merger, MS merger")
        print(len(RLO.ID.unique()), len(RLO_mergers.ID.unique()), len(CE_merger), len(contact_merger), len(DCCE_merger), len(MS_merger))
        print("sum of everything")
        print(len(RLO.ID.unique()) + len(RLO_mergers.ID.unique()) + len(CE_merger) + len(contact_merger) +len(DCCE_merger) + len(MS_merger))
    
    # print leftovers
    if verbose:
        print(f'Number of other RLO mergers: {len(RLO_mergers.ID.unique())}')
        #for ii in RLO_mergers.ID.unique()[:10]:
        #    print(d.loc[d.ID == ii][['time', 'event', 'type1', 'type2', 'mass1', 'mass2', 'semiMajor']])

    # Next look at successful RLO events
    # First find DCCE
    DCCE = RLO.loc[(RLO.event == 513)].ID.unique()
    # remove them
    RLO = RLO.loc[~RLO.ID.isin(DCCE)]

    # Next find CE from primary
    CE_1 = RLO.loc[(RLO.event == 511) & (RLO.event.shift() == 31)].ID.unique()
    CE_1_BSE = RLO.loc[(RLO.event == 511) & (RLO.event.shift() == 3)].ID.unique()
    CE_1_BPASS = RLO.loc[(RLO.event == 511) & (RLO.event.shift() == 13)].ID.unique()
    CE_1 = np.append(CE_1, CE_1_BSE, CE_1_BPASS)
    
    # remove them
    RLO = RLO.loc[~RLO.ID.isin(CE_1)]


    #Next find stable mass transfer from primary
    SMT_1 = RLO.loc[RLO.event == 31].ID.unique()
    # remove them
    RLO = RLO.loc[~RLO.ID.isin(SMT_1)]
    
    # Next find CE from secondary [this really shouldn't happen often]
    CE_2 = RLO.loc[(RLO.event == 512) & (RLO.event.shift() == 32)].ID.unique()
    CE_2_BSE = RLO.loc[(RLO.event == 512) & (RLO.event.shift() == 3)].ID.unique()
    CE_2 = np.append(CE_2, CE_2_BSE)
    # remove them
    RLO = RLO.loc[~RLO.ID.isin(CE_2)]
    
    # Next find stable mass transfer from secondary [this really shouldn't happen often]
    SMT_2 = RLO.loc[RLO.event == 32].ID
    # remove them
    RLO = RLO.loc[~RLO.ID.isin(SMT_2)]
    
    # print leftovers
    if verbose:
        print(f'Number of other RLO events: {len(RLO.ID.unique())}')
        print(RLO.ID.unique())
        print()
    
    first_RLO = {'SMT_1': SMT_1,
                 'SMT_2': SMT_2.values,
                 'CE_1': CE_1,
                 'CE_2': CE_2,
                 'DCCE': DCCE,
                 'failed_CE_merger': CE_merger,
                 'contact_merger': contact_merger,
                 'DCCE_merger': DCCE_merger,
                 'other_merger': RLO_mergers.ID.unique(),
                 'MS_merger': MS_merger,
                 'nonRLO': nonRLO,
                 'leftovers': RLO.ID.unique()}
    if verbose:
        tot = 0
        for k in first_RLO.keys():
            tot += len(first_RLO[k])
    
        print('channel tags: ', tot)
        print('total IDs: ', len(d.ID.unique()))
        # Combine all IDs with their keys
        print('find duplicates')
    
        # IDs that exist in the dataframe
        all_ids = set(d.ID.unique())
        
        # Flatten all IDs in the dictionary
        dict_ids = {k: set(v) for k, v in first_RLO.items()}
        all_dict_ids = set().union(*dict_ids.values())
        
        # Find IDs that are in first_RLO but not in d
        extra_ids = all_dict_ids - all_ids
        
        # Map those extras back to their dictionary keys
        extra_id_sources = {k: list(v & extra_ids) for k, v in dict_ids.items() if len(v & extra_ids) > 0}
        
        print("Extra IDs not found in dataframe:")
        print(extra_id_sources)
    
        from collections import defaultdict
    
        # build a mapping from ID → list of channels it appears in
        id_locations = defaultdict(list)
        for key, arr in first_RLO.items():
            for x in np.unique(arr):  # unique within each channel
                id_locations[x].append(key)
        
        # find IDs that appear in >1 channel
        duplicates = {k: v for k, v in id_locations.items() if len(v) > 1}
        
        print(f"Number of duplicated IDs: {len(duplicates)}")
        
        for k, v in list(duplicates.items())[:10]:
            print(k, "→", v)
    
    
        within_channel_dupes = {}
        
        for key, arr in first_RLO.items():
            vals, counts = np.unique(arr, return_counts=True)
            dupes = vals[counts > 1]
            if len(dupes) > 0:
                within_channel_dupes[key] = dupes
        
        print("IDs duplicated within individual channels:")
        print(within_channel_dupes)
        print("")
        print("\n")    
    return first_RLO

    
def first_interaction_channels_simple(d, verbose=False):
    '''Split out the different types of channels that could occur
    in the first interaction with only SMT, CE, SMT->CE, other

    Parameters
    ----------
    d : `pandas.DataFrame`
        contains T0 data for binaries as specified by BinCodex

    verbose : `bool`
        Sets whether to print out detailed results

    Returns
    -------
    first_RLO : `Dictionary`
        contains a list of IDs for different interaction channels
    '''

# def select_final_state_ids(d):
#    '''Sets the final state based on the evolution
#
#    Parameters
#    ----------
#    d : `pandas.DataFrame`
#        T0 data with all events listed
#
#    Returns
#    -------
#
#    '''


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






