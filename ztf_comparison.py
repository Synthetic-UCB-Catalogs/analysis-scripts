import numpy as np
import matplotlib.pyplot as plt
import os

ztf_array = np.genfromtxt('ZTF frequency distance.dat')
#the ZTF data used by the plots

def wd_radius_PPE(m):
    """
    Calculates radii of WDs based on mass. Called by other functions, no need
    to call this directly. Based on Verbunt & Rappaport (1988) with
    modifications from van Zeist et al. (2025).
    """
    
    # Eggleton 1986 fit to Nauenberg for high m and
    # ZS for low m. From Verbunt & Rappaport (1988)
    # 3/2 multiplier from van Zeist et al. (2025)
    
    if m > 1.44:
        raise ValueError('WD with mass above 1.44 Msun encountered during ' +
                         'radius calculation.')
    
    
    fac1 = (m/1.44)**(2./3.)
    fac2 = 0.00057/m
    a = 3.5
    b = 1.

    r = (3/2)*0.0114*np.sqrt(1./fac1-fac1)*(1.+a*(fac2)**(2./3.)+b*fac2)**(-2./3.)
    #3/2 multiplier based on comparisons to ZTF masses and radii

    return r

def get_f_gw_from_semimajor(m1, m2, a):
    """
    Compute orbital frequency in Hz.

    Parameters
    ----------
    a : float or array
        Semi-major axis in solar radii.
    m1, m2 : float or array
        Masses in solar masses.

    Returns
    -------
    f_orb : float or array
        Orbital frequency in Hz.
    """
    G = 6.67430e-11          # m^3 kg^-1 s^-2
    M_sun = 1.98847e30       # kg
    R_sun = 6.957e8          # m.
    a_m = a * R_sun
    m_total = (m1 + m2) * M_sun

    f_orb = (1 / (2 * np.pi)) * np.sqrt(G * m_total / a_m**3)
    return 2*f_orb

def frequency_distance_bins(code_name, var_type, var_name, rclone_flag=True,
                            ecl_weight=True, recalc_rad=True):
    """
    Sorts the galaxy data into frequency/distance bins for the plotting code
    to use.
    
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
    ecl_weight: bool
        If True, weight systems by their eclipse probability when binning them.
    recalc_rad: bool
        If True, calculate radii based on the wd_radius_PPE() formula. If
        False, use whatever radii are provided in the galaxy file.
    
    Returns
    -------
    amount_per_bin: array
        Numbers of binaries per frequency-distance bin.
    log_freq_bin_bounds: array/list
        Edges of the bins in log(frequency/Hz).
    dist_bin_bounds: array/list
        Edges of the bins in distance (pc).
    """
    
    """ Initialisation """
    
    log_freq_bin_bounds = np.linspace(-5,0,num=51)
    #every 0.1 dex in log-space: -5, -4.9, -4.8 etc.
    
    dist_upper_bound = 2000 #pc
    dist_upper_bound_kpc = dist_upper_bound/1000
    dist_bin_bounds = np.linspace(0,dist_upper_bound,num=41) #every 50 pc, linearly
    
    amount_per_bin = np.zeros((50,40)) #zeros, not empty
    #50,40 instead of 51,41 because each bin is *between* the bounds from the lists above
    
    
    
    """ Fetching the right AllDWDs file """
    
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
    all_dwd_filepath = initial_string + var_string + '/' + code_name + \
        '_Galaxy_AllDWDs.csv'
    
    
    
    """ Running through the galaxy file """
    
    galaxy_file = open(all_dwd_filepath,'r')
    
    iteration_no = 0 #counter to keep track of progress
    iteration_kept = 0 #only those iterations kept within the distance limit
    header_row_flag = True #handle header row separately
    
    for line in galaxy_file:
        if header_row_flag == True: #skip first row of file (headers)
            #using the header row to identify columns
            line_as_list = list(line.split(','))
            try: dist_index = line_as_list.index('dist')
            except ValueError: dist_index = line_as_list.index('RRelkpc')
            m1_index = line_as_list.index('mass1')
            m2_index = line_as_list.index('mass2')
            a_index = line_as_list.index('semiMajor')
            r1_index = line_as_list.index('radius1')
            r2_index = line_as_list.index('radius2')
            
            header_row_flag = False
            continue
        
        line_as_list = list(line.split(','))
        dist = float(line_as_list[dist_index]) #kpc
        
        if dist < dist_upper_bound_kpc:
            dist_bin = np.floor(dist/0.05) #bins are 50 pc (0.05 kpc) wide and start at 0 pc
        else: #skip systems at larger distances
            iteration_no += 1
            if (iteration_no + 1) % 1000000 == 0: print(str(iteration_no + 1) + ' systems done')
            continue
        
        m1 = float(line_as_list[m1_index]) #Msun
        m2 = float(line_as_list[m2_index]) #Msun
        a = float(line_as_list[a_index]) #Rsun
        
        freq = get_f_gw_from_semimajor(m1,m2,a) #Hz
        freq_bin = np.floor(10*np.log10(freq)) + 50
        #rounds to nearest 0.1, then adds 50 to map -5.0 (-50) to index 0
        
        if ecl_weight == True:
            if recalc_rad == True:
                r1 = wd_radius_PPE(m1) #Rsun
                r2 = wd_radius_PPE(m2) #Rsun
            else:
                r1 = float(line_as_list[r1_index]) #Rsun
                r2 = float(line_as_list[r2_index]) #Rsun
            
            system_weight = (r1 + r2)/a #eclipse probability; a and r in Rsun
        else:
            system_weight = 1
        
        amount_per_bin[int(freq_bin),int(dist_bin)] += system_weight
        #add the (weighted) system to the total for the appropriate freq/dist bin
        
        iteration_no += 1
        iteration_kept += 1
        if (iteration_no + 1) % 1000000 == 0: print(str(iteration_no + 1) + ' systems done') #+1 because of 0-indexing
        
    galaxy_file.close()
    
    print('Total DWDs in galaxy file: ' + str(iteration_no))
    print('DWDs within distance limit: ' + str(iteration_kept))
    print('DWDs kept with eclipse prob. (should be same as above if ' +
          'ecl_weight=False): ' + str(int(sum(sum(amount_per_bin)))))
    
    return amount_per_bin, log_freq_bin_bounds, dist_bin_bounds

def plot_ztf_comparison(code_name, var_type, var_name, rclone_flag=True,
                        ecl_weight=True, recalc_rad=True, model_array=None,
                        panel_option='one'):
    """
    Makes plots comparing a given model galaxy to the ZTF sample. Has the
    option to plot either one panel from 0 to 2000 pc, or three panels with
    0-500, 500-1000 and 1000-2000 pc respectively.
    Note: Currently can plot only one galaxy model at a time.
    
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
    ecl_weight: bool
        If True, weight systems by their eclipse probability when binning them.
    recalc_rad: bool
        If True, calculate radii based on the wd_radius_PPE() formula. If
        False, use whatever radii are provided in the galaxy file.
    model_array: array
        Numbers of binaries per frequency-distance bin, as output by
        frequency_distance_bins(). Overrides the code/variation parameters if
        specified.
    panel_option: str
        Specify "one" or "merged" to have one panel with distances from 0 to
        2000 pc. Specify "three" or "split" to have separate panels for 0-500,
        500-1000 and 1000-2000 pc.
    """
    
    if model_array is None:
        model_array, log_freq_bin_bounds, dist_bin_bounds = \
            frequency_distance_bins(code_name, var_type, var_name,
            rclone_flag=rclone_flag, ecl_weight=ecl_weight, recalc_rad=recalc_rad)
    
    log_freq_bin_centres = np.linspace(-4.95,-0.05,num=50)
    #dist_bin_centres = np.linspace(25,1975,num=40)
    #would be good to couple these to the bin bounds from frequency_distance_bins()
    
    ztf_log_freq = ztf_array[:,1]
    ztf_dist_pc = ztf_array[:,2]
    
    ztf_near = np.zeros((50)) #zeros, not empty
    ztf_mid = np.zeros((50))
    ztf_far = np.zeros((50))
    
    for i in range(len(ztf_log_freq)):
        freq_bin = np.floor(10*ztf_log_freq[i]) + 50 #rounds to nearest 0.1,
        #then adds 50 to map -5.0 (-50) to index 0
        
        if ztf_dist_pc[i] <= 500:
            ztf_near[int(freq_bin)] += 1
            #add the system to the total for the appropriate frequency and distance bin
        elif ztf_dist_pc[i] <= 1000:
            ztf_mid[int(freq_bin)] += 1
        elif ztf_dist_pc[i] <= 2000:
            ztf_far[int(freq_bin)] += 1
    
    freqs_reversed = list(reversed(log_freq_bin_centres[10:30]))
    
    if panel_option == 'three' or panel_option == 'split':
        model_near = [sum(model_array[i,0:10]) for i in range(model_array.shape[0])] #0-500 pc
        model_mid = [sum(model_array[i,10:20]) for i in range(model_array.shape[0])] #500-1000 pc
        model_far = [sum(model_array[i,20:40]) for i in range(model_array.shape[0])] #1000-2000 pc
        
        model_near_cu = np.cumsum(list(reversed(model_near[10:30])))
        model_mid_cu = np.cumsum(list(reversed(model_mid[10:30])))
        model_far_cu = np.cumsum(list(reversed(model_far[10:30])))
        
        ztf_near_cu_uncorr = np.cumsum(list(reversed(ztf_near[10:30])))
        ztf_mid_cu_uncorr = np.cumsum(list(reversed(ztf_mid[10:30])))
        ztf_far_cu_uncorr = np.cumsum(list(reversed(ztf_far[10:30])))
        
        plt.figure(1)
        plt.step(freqs_reversed,0.6*model_near_cu,'C0-',marker=11,linewidth=2,where='mid',label='Model')
        plt.step(freqs_reversed,ztf_near_cu_uncorr,'k--',linewidth=2,where='mid',label='ZTF obs.')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('log(Frequency/Hz)')
        plt.ylabel('Cumulative amount of systems')
        plt.title('(A) ' + code_name + ', ' + var_name + ', distance = 0-500 pc')
        
        plt.figure(2)
        plt.step(freqs_reversed,0.6*model_mid_cu,'C0-',marker=11,linewidth=2,where='mid',label='Model')
        plt.step(freqs_reversed,ztf_mid_cu_uncorr,'k--',linewidth=2,where='mid',label='ZTF obs.')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('log(Frequency/Hz)')
        plt.ylabel('Cumulative amount of systems')
        plt.title('(B) ' + code_name + ', ' + var_name + ', distance = 500-1000 pc')
        
        plt.figure(3)
        plt.step(freqs_reversed,0.6*model_far_cu,'C0-',marker=11,linewidth=2,where='mid',label='Model')
        plt.step(freqs_reversed,ztf_far_cu_uncorr,'k--',linewidth=2,where='mid',label='ZTF obs.')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('log(Frequency/Hz)')
        plt.ylabel('Cumulative amount of systems')
        plt.title('(C) ' + code_name + ', ' + var_name + ', distance = 1000-2000 pc')
        
    elif panel_option == 'one' or panel_option == 'merged':
        model_all = [sum(model_array[i,0:40]) for i in range(model_array.shape[0])] #0-2000 pc
        model_all_cu = np.cumsum(list(reversed(model_all[10:30])))
        ztf_all = [sum(x) for x in zip(ztf_near,ztf_mid,ztf_far)]
        ztf_all_cu_uncorr = np.cumsum(list(reversed(ztf_all[10:30])))
        
        plt.figure(1)
        plt.step(freqs_reversed,0.6*model_all_cu,'C0-',marker=11,linewidth=2,where='mid',label='Model')
        plt.step(freqs_reversed,ztf_all_cu_uncorr,'k--',linewidth=2,where='mid',label='ZTF obs.')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('log(Frequency/Hz)')
        plt.ylabel('Cumulative amount of systems')
        plt.title(code_name + ', ' + var_name + ', distance = 0-2000 pc')
        
    else:
        raise ValueError('Invalid panel_option specified.')
