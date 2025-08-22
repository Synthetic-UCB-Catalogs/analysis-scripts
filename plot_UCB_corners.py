import sys, getopt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import rapid_code_load_T0 as load
import formation_channels as fc
import basicplot as bp


## arguments
code = 'COSMIC'
home = "/Users/gijsnelemans/Compute/Untitled Folder/" # put your local folder here
variation = 'porb_log_uniform'
bin_dir = home+'data_products/simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/'+variation
gal_dir = home+'data_products/simulated_galaxy_populations/monte_carlo_comparisons/initial_condition_variations/'+variation
bin_file = code+'_T0.hdf5'
gal_file = code+'_Galaxy_AllDWDs.csv'
LISA_file = code+'_Galaxy_LISA_DWDs.csv'
weight_file = code+'_Galaxy_LISA_Candidates_Bin_Data.csv'
limits = None

opts, args = getopt.getopt(sys.argv[1:],"c:d:f:g:G:L:v:l:",["code=','bin_dir=","bin_file=","gal_dir=","gal_file=","LISA_file=","variation=","limits="])

for opt, arg in opts:
    if opt in ("-c", "--code"):
        code = arg
        bin_file = code+'_T0.hdf5'
        gal_file = code+'_Galaxy_AllDWDs.csv'
        LISA_file = code+'_Galaxy_LISA_DWDs.csv'
        weight_file = code+'_Galaxy_LISA_Candidates_Bin_Data.csv'
    elif opt in ("-d", "--bin_dir"):
        bin_dir = arg
    elif opt in ("-f", "--bin_file"):
        bin_file = arg
    elif opt in ("-g", "--gal_dir"):
        gal_dit = arg
    elif opt in ("-G", "--gal_file"):
        gal_file = arg
    elif opt in ("-L", "--LISA_file"):
        LISA_file = arg
    elif opt in ("-l", "--limits"):
        lim = arg
        limits = tuple(float(i) for i in lim.split(","))
    elif opt in ("-w", "--weight_file"):
        weight_file = arg
    elif opt in ("-v", "--variation"):
        variation = arg
        bin_dir = home+'data_products/simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/'+variation
        gal_dir = home+'data_products/simulated_galaxy_populations/monte_carlo_comparisons/initial_condition_variations/'+variation



print(code,variation,bin_dir,bin_file)

## Settings
bins = 'log' # 'log' for log density, None for linear density, integer value i for number of density bins
colmap = 'Blues'
Ngrid = 30

##

def logP_from_a(m1, m2, a):
    Mtot = m1 + m2
    return np.log10(0.116*np.sqrt(a**3/(Mtot)))


## load binary evolution data

bin_path=bin_dir+'/'+bin_file
print(bin_path)
data_T0, data_header = load.load_T0_data(bin_path)
ZAMS, WDMS, DWD = fc.select_evolutionary_states(d=data_T0)

## load galaxy population data

gal_path=gal_dir+'/'+gal_file
LISA_path=gal_dir+'/'+LISA_file
print(gal_path)
Galaxy = pd.read_csv(gal_path,usecols=['ID','time','semiMajor','mass1','mass2'])
LISA_Gal = pd.read_csv(LISA_path,usecols=['ID','time','semiMajor','mass1','mass2'])
LISA_IDs=Galaxy.drop_duplicates(subset=['ID'])['ID']

weight_path=gal_dir+'/'+weight_file
LISA_Gal_bin = pd.read_csv(weight_path)
N_candidates = LISA_Gal_bin.SubBinNDWDsReal.sum()
N_LISA = N_candidates*len(LISA_Gal)/len(Galaxy)
print(N_candidates, N_LISA)

## plot ZAMS hexbin
log_P = logP_from_a(ZAMS.mass1,ZAMS.mass1,ZAMS.semiMajor)
log_tau = np.zeros(len(log_P))
bp.hexbin_plot(ZAMS.mass1,ZAMS.mass2,log_P,log_tau,code+' @ZAMS',bins,colmap,Ngrid,limits)

## plot ZAMS points
mask=ZAMS.ID.isin(LISA_IDs)
bp.point_plot(ZAMS.mass1,ZAMS.mass2,log_P,log_tau,mask,code+' @ZAMS points',limits)

## plot WDMS hexbin

log_P = logP_from_a(WDMS.mass1,WDMS.mass1,WDMS.semiMajor)
log_tau = np.log10(WDMS.time)
bp.hexbin_plot(WDMS.mass1,WDMS.mass2,log_P,log_tau,code+' @WDMS formation',bins,colmap,Ngrid,limits)

## plot WDMS points
mask=WDMS.ID.isin(LISA_IDs)
bp.point_plot(WDMS.mass1,WDMS.mass2,log_P,log_tau,mask,code+' @WDMS points',limits)

## plot DWD hexbin

log_P = logP_from_a(DWD.mass1,DWD.mass1,DWD.semiMajor)
log_tau = np.log10(DWD.time)

bp.hexbin_plot(DWD.mass1,DWD.mass2,log_P,log_tau,code+' @DWD formation',bins,colmap,Ngrid,limits)

## plot LISA full Galaxy
log_P = logP_from_a(Galaxy.mass1,Galaxy.mass1,Galaxy.semiMajor)
log_tau = np.log10(Galaxy.time)

bp.hexbin_plot(Galaxy.mass1,Galaxy.mass2,log_P,log_tau,code+' Galaxy {:,} sources'.format(int(N_candidates)),bins,colmap,Ngrid,limits)

##bp. plot LISA detected sources
log_P = logP_from_a(LISA_Gal.mass1,LISA_Gal.mass1,LISA_Gal.semiMajor)
log_tau = np.log10(LISA_Gal.time)

bp.hexbin_plot(LISA_Gal.mass1,LISA_Gal.mass2,log_P,log_tau,code+' Galaxy only {:,} LISA sources'.format(int(N_LISA)),bins,colmap,Ngrid,limits)


