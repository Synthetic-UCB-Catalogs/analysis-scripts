import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

def load_channel_ids(filename, code, variation):
    """
    Loads the binary evolution channel IDs from an HDF5 file.

    Parameters
    ----------
    filename: str
        Path to the HDF5 file.
    code: str
        The formaton channel you want binary IDs for.
    variation: str
        The variation you want formaton channel IDs for.

    Returns
    -------
    dict
        Dictionary containing the stored data.
    """
    key = f'{code}_{variation}'
    with h5py.File(filename, "r") as f:
        if key not in f:
            raise KeyError(f"No group named '{key}' in {filename}")
        grp = f[key]
        return {k: grp[k][()] for k in grp}

def dwd_count_single_code(code_name, var_type, var_name, rclone_flag=True,
                          channel=None):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code/variation. If rclone_flag is True, filepaths assume you have set up
    rclone for the project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_name: str
        Name of the initial condition/mass transfer variation (e.g.
        "fiducial").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
    channel: str or None
        Specify the name of a specific formation channel to count. If None,
        counts DWDs from all channels.
        
    Returns
    -------
    dwd_count: float
        Number of LISA DWDs predicted in the Galaxy for that code/variation.
    """
    
    if var_type == 'icv' or var_type == 'initial_condition_variations':
        var_type_string = 'initial_condition_variations/'
        var_string = var_name
    elif var_type == 'mtv' or var_type == 'mass_transfer_variations':
        var_type_string = 'mass_transfer_variations/'
        #select appropriate subfolder in mass_transfer_variations
        if var_name == 'fiducial':
            var_string = var_name
        elif var_name == 'alpha_lambda_1' or var_name == 'alpha_lambda_2' or \
            var_name == 'alpha_lambda_02' or var_name == 'alpha_lambda_05' or \
            var_name == 'alpha_gamma_2':
                var_string = 'common_envelope/' + var_name
        elif var_name == 'qcrit_claeys_14' or var_name == 'qcrit_hurley_02' \
            or var_name == 'qcrit_hurley_webbink' or var_name == 'qcrit_zetas':
                var_string = 'stability_of_mass_transfer/' + var_name
        elif var_name == 'accretion_0' or var_name == 'accretion_1' or \
            var_name == 'accretion_05':
                var_string = 'stable_accretion_efficiency/' + var_name
        else:
            raise ValueError('Invalid mass transfer variation specified.')
    else:
        raise ValueError('Please specify either initial condition or mass ' +
                        'transfer variations.')
    
    if rclone_flag == True:
        drive_filepath = '/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
        initial_string = os.environ['UCB_GOOGLE_DRIVE_DIR'] + drive_filepath
    else:
        initial_string = 'data_products/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
    lisa_dwd_filepath = initial_string + var_string + '/' + code_name + \
        '_Galaxy_LISA_DWDs.csv'
    
    lisa_dwd_array = pd.read_csv(lisa_dwd_filepath)
    
    if channel is None:                     #count all channels
        dwd_count = len(lisa_dwd_array)
    else:                                   #count one channel
        if rclone_flag == True:
            path_to_channel_file = os.environ['UCB_GOOGLE_DRIVE_DIR'] + \
                '/simulated_binary_populations/monte_carlo_comparisons/' + \
                'channel_ids.h5'
        else:
            path_to_channel_file = 'data_products' + \
                '/simulated_binary_populations/monte_carlo_comparisons/' + \
                'channel_ids.h5'
        channel_ids = load_channel_ids(filename=path_to_channel_file, 
            code=code_name, variation=var_name)
        mask = lisa_dwd_array.ID.isin(channel_ids[channel])
        dwd_count = len(lisa_dwd_array[mask])
    
    return dwd_count

def dwd_count_icv_average(code_name, rclone_flag=True):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code, averaged over each initial condition variation.
    If rclone_flag is True, filepaths assume you have set up rclone for the
    project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    mean_dwd_count: float
        Number of LISA DWDs predicted in the Galaxy for that code, averaged
        over all initial condition variations.
    """
    
    icv_names = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01', \
                 'thermal_ecc', 'uniform_ecc']
    var_count = np.empty((len(icv_names))) #holds counts from each IC variation
    
    for i in range(len(icv_names)):
        var_count[i] = dwd_count_single_code(code_name, 'icv', icv_names[i], \
                                             rclone_flag)
    
    mean_dwd_count = np.mean(var_count) #average counts over IC variations
    
    return mean_dwd_count

def dwd_count_icv_min_max(code_name, rclone_flag=True):
    """
    Calculates the number of LISA DWDs predicted in the Galaxy for a single
    code, and returns the minimum and maximum values across the different
    initial condition variations.
    If rclone_flag is True, filepaths assume you have set up rclone for the
    project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    min_dwd_count: float
        Minimum number of LISA DWDs predicted in the Galaxy for that code over
        all initial condition variations.
    max_dwd_count: float
        Minimum number of LISA DWDs predicted in the Galaxy for that code over
        all initial condition variations.
    """
    
    icv_names = ['fiducial', 'm2_min_05', 'porb_log_uniform', 'qmin_01', \
                 'thermal_ecc', 'uniform_ecc']
    var_count = np.empty((len(icv_names))) #holds counts from each IC variation
    
    for i in range(len(icv_names)):
        var_count[i] = dwd_count_single_code(code_name, 'icv', icv_names[i], \
                                             rclone_flag)
    
    min_dwd_count = np.min(var_count)
    max_dwd_count = np.min(var_count)
    
    return min_dwd_count, max_dwd_count

def all_dwd_single_code(code_name, var_type, var_name, rclone_flag=True):
    """
    Calculates the total number of DWDs in the Galaxy (not just the LISA-
    detectable ones) for a single code/variation.
    If rclone_flag is True, filepaths assume you have set up rclone for the
    project's Google Drive as per Reinhold's tutorial:
    https://docs.google.com/document/d/1v0dEQWhxzqQoJm877m7fWWhHSTwcOgIvAS87idheNnA
    If rclone_flag is False, filepaths assume you have the top-level directory
    in the project's Google Drive as working directory.
    
    Parameters
    ----------
    code_name: str
        Name of the code (e.g. "ComBinE", "SEVN").
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_name: str
        Name of the initial condition/mass transfer variation (e.g.
        "fiducial").
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
        
    Returns
    -------
    total_dwd_count: float
        Total number of DWDs predicted in the Galaxy for that code/variation.
    """
    
    if var_type == 'icv' or var_type == 'initial_condition_variations':
        var_type_string = 'initial_condition_variations/'
        var_string = var_name
    elif var_type == 'mtv' or var_type == 'mass_transfer_variations':
        var_type_string = 'mass_transfer_variations/'
        #select appropriate subfolder in mass_transfer_variations
        if var_name == 'fiducial':
            var_string = var_name
        elif var_name == 'alpha_lambda_1' or var_name == 'alpha_lambda_2' or \
            var_name == 'alpha_lambda_02' or var_name == 'alpha_lambda_05' or \
            var_name == 'alpha_gamma_2':
                var_string = 'common_envelope/' + var_name
        elif var_name == 'qcrit_claeys_14' or var_name == 'qcrit_hurley_02' \
            or var_name == 'qcrit_hurley_webbink' or var_name == 'qcrit_zetas':
                var_string = 'stability_of_mass_transfer/' + var_name
        elif var_name == 'accretion_0' or var_name == 'accretion_1' or \
            var_name == 'accretion_05':
                var_string = 'stable_accretion_efficiency/' + var_name
        else:
            raise ValueError('Invalid mass transfer variation specified.')
    else:
        raise ValueError('Please specify either initial condition or mass ' +
                        'transfer variations.')
    
    if rclone_flag == True:
        drive_filepath = '/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
        initial_string = os.environ['UCB_GOOGLE_DRIVE_DIR'] + drive_filepath
    else:
        initial_string = 'data_products/simulated_galaxy_populations/' + \
            'monte_carlo_comparisons/' + var_type_string
    bin_data_filepath = initial_string + var_string + '/' + code_name + \
        '_Galaxy_LISA_Candidates_Bin_Data.csv'
    
    bin_data_array = pd.read_csv(bin_data_filepath)
    total_dwd_count = sum(bin_data_array['SubBinNDWDsReal'])
    
    return total_dwd_count

def lisa_dwd_count_plotter(code_list, var_type, var_list, cmap='rainbow', \
                           rclone_flag=True, channel=None):
    """
    Plots the number of LISA DWDs in the Galaxy for specified codes/variations.
    
    Parameters
    ----------
    code_list: list of strs
        List of the names of the codes you want to plot.
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_list: list of strs
        List of the names of the variations you want to plot.
    cmap: str
        Pyplot colormap to use for the bar plot. Defaults to 'rainbow', but we
        recommend 'gist_rainbow' if you are comparing many (5+) variations.
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
    channel: str or None
        Specify the name of a specific formation channel to count. If None,
        counts DWDs from all channels.
    """

    fig, ax = plt.subplots()
    width = 0.7/len(var_list) #make bars narrower if plotting more variations

    plot_colormap = plt.get_cmap(cmap)
    plot_colors = plot_colormap(np.linspace(0,1,len(var_list)))

    for i in range(len(code_list)):
        for j in range(len(var_list)):
            try: ax.bar(i+j*width, dwd_count_single_code(code_list[i], \
                 var_type, var_list[j], rclone_flag), width, \
                 color=plot_colors[j])
            except FileNotFoundError: ax.bar(i+j*width, np.nan, width, \
                 color=plot_colors[j]) #handles missing codes/variations
    ax.set_xticks(np.linspace((len(var_list)/2 - 0.5)*width, len(code_list) - \
              1 + (len(var_list)/2 - 0.5)*width, len(code_list)), code_list)
    #centers ticks for each group of bars
    ax.legend(var_list)
    
    return fig, ax
    
def total_dwd_count_plotter(code_list, var_type, var_list, cmap='rainbow', \
                           rclone_flag=True):
    """
    Plots the total number of DWDs in the Galaxy (not just the LISA-detectable
    ones) for specified codes/variations.
    
    Parameters
    ----------
    code_list: list of strs
        List of the names of the codes you want to plot.
    var_type: str
        Whether you want to use the initial condition variations or the mass
        transfer variations.
    var_list: list of strs
        List of the names of the variations you want to plot.
    cmap: str
        Pyplot colormap to use for the bar plot. Defaults to 'rainbow', but we
        recommend 'gist_rainbow' if you are comparing many (5+) variations.
    rclone_flag: bool
        Whether you have set up rclone for the filepaths in the Google Drive or
        not.
    """

    fig, ax = plt.subplots()
    width = 0.7/len(var_list) #make bars narrower if plotting more variations

    plot_colormap = plt.get_cmap(cmap)
    plot_colors = plot_colormap(np.linspace(0,1,len(var_list)))

    for i in range(len(code_list)):
        for j in range(len(var_list)):
            try: ax.bar(i+j*width, all_dwd_single_code(code_list[i], \
                 var_type, var_list[j], rclone_flag), width, \
                 color=plot_colors[j])
            except FileNotFoundError: ax.bar(i+j*width, np.nan, width, \
                 color=plot_colors[j]) #handles missing codes/variations
    ax.set_xticks(np.linspace((len(var_list)/2 - 0.5)*width, len(code_list) - \
              1 + (len(var_list)/2 - 0.5)*width, len(code_list)), code_list)
    #centers ticks for each group of bars
    ax.legend(var_list)
    
    return fig, ax
