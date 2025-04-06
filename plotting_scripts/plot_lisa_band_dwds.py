# %matplotlib inline

# +
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.environ["PWD"])))
from combine_popsynth_and_galaxy_data import matchDwdsToGalacticPositions
from utils import chirp_mass

SIM_DIR = os.environ['UCB_GOOGLE_DRIVE_DIR']
# -

masked_dwds, masks = matchDwdsToGalacticPositions( 
                        pathToPopSynthData= os.path.join(SIM_DIR, "simulated_binary_populations/monte_carlo_comparisons/initial_condition_variations/fiducial/COSMIC_T0.hdf5"),
                        #pathToGalacticSamples="galaxy_models/SampledGalacticLocations_Besancon_0.0142.h5",
                        useLegworkMask=False, 
                        applyInitialLisaBandFilter=False)

# +

m1 = masked_dwds[0,:]
m2 = masked_dwds[1,:]
a = masked_dwds[2,:]
fGW = masked_dwds[3,:]
d_gal = masked_dwds[6,:]

mask, mask_lisa_band = masks

fs_lgnd = 12
fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
ax = axes[0]
ax.plot(m1+m2, a[mask], 'o', label="All")
ax.plot(m1[mask_lisa_band]+m2[mask_lisa_band], a[mask][mask_lisa_band], 'o', label="LISA band")
ax.set_xlabel("$M_{total} [M_\odot]$")
ax.set_ylabel("$a_{today} [R_\odot]$")
ax.legend(fontsize=fs_lgnd)
ax.set_yscale('log')

ax = axes[1]
ax.plot(m1+m2, fGW, 'o', label="All")
ax.plot(m1[mask_lisa_band]+m2[mask_lisa_band], fGW[mask_lisa_band], 'o', label="LISA band")
ax.set_xlabel("$M_{total} [M_\odot]$")
ax.set_ylabel("$f_{GW,today}$ [Hz]")
ax.legend(fontsize=fs_lgnd)
ax.set_yscale('log')

ax = axes[2]
bins = np.linspace(-20, 0, 100)
ax.hist(np.log10(fGW), bins=bins, weights=chirp_mass(m1, m2)/d_gal, label="All")
ax.hist(np.log10(fGW[mask_lisa_band]), bins=bins, weights=chirp_mass(m1, m2)[mask_lisa_band]/d_gal[mask_lisa_band], label="LISA band")
ax.set_xlabel("$f_{GW,today}$ [Hz]")
ax.legend(fontsize=fs_lgnd)

fig.tight_layout()
plt.show()
# -



