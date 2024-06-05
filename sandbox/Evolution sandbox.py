# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evolution times
#
# Here we look at two items:
#
# - using the approximant for GW merger time proposed in [arXiv:2110.09254](https://arxiv.org/abs/2110.09254) (Mandel, 2021);
# - calculating time to interaction for double white dwarfs (DWDs) which will be determined by when a Roche lobe overflow starts;

# %% [markdown]
# ## Preamble

# %%
from legwork import evol, utils

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.visualization import quantity_support
quantity_support()
import matplotlib.pyplot as plt

MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rcdefaults()

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %matplotlib inline

# %% [markdown]
# ## Approximations
#
# Here we compare the GW merger times calculated in LEGWORK the small and large eccentricity approximations from [Peters (1964)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.B1224) with the approximant proposed in [Mandel (2021)](https://arxiv.org/abs/2110.09254). For the Peters approximations the eccentricity cutoffs are the default ones used by `legwork.evol.get_t_merge_ecc(...)`, and the Mandel approximant reads
# \begin{eqnarray}
# T &\approx& T_{\rm c}\left(1 + 0.27 e_0^{10} + 0.33 e_0^{20} + 0.2 e_0^{1000}\right)\left(1-e_0^2\right)^{7/2}\,, \\
# T_{\rm c} &=& \frac{a_0^4}{4\beta}\,, \qquad \beta \equiv \frac{64}{5}\frac{G^3 m_1 m_2(m_1 + m_2)}{c^5} = \frac{64\eta}{5} c\left(\frac{GM}{c^2}\right)^3\,,
# \end{eqnarray}
# where
# - $T_{\rm c}$: merger time for a circular binary;
# - $a_0$ and $e_0$: inital semimajor axis and eccentricity, respectively;
# - $m_1$, $m_2$, and $M\equiv m_1+m2$: masses of the binary's components and the total mass;
# - $\beta$: a combination introduced in Peters (1964) and used in LEGWORK;
# - $\eta=m_1 m_2/M^2$: the symmetric mass often used in the GW science literature. For an equal-mass binary, $\eta = 1/4$.
#
# In the comparison below, $m_1=m_2=M_\odot$ and a range of eccentricities corresponding $(1-e_0)\in[10^{-4},1]$.
#
# *Conclusion*. The plot shows that the Mandel approximant works better at small eccentricities.

# %%
# create random binaries
e_range = np.linspace(0.0, 0.9999, 10000)
m_1 = np.repeat(1, len(e_range)) * u.Msun
m_2 = np.repeat(1, len(e_range)) * u.Msun
f_orb_i = np.repeat(1e-5, len(e_range)) * u.Hz

# calculate merger time using defaults
t_merge = evol.get_t_merge_ecc(ecc_i=e_range, m_1=m_1, m_2=m_2, f_orb_i=f_orb_i)

# calculate merger time using an approximant from arXiv:2110.09254 (Mandel, 2021)
Tc = evol.get_t_merge_circ(m_1=m_1, m_2=m_2, f_orb_i=f_orb_i)
ecc10 = e_range**10
t_merge_Mandel = Tc*(1 + 0.27*ecc10 + 0.33*ecc10**2 + 0.2*ecc10**100) * (1 - e_range**2)**(7./2)

# create a figure
fig, ax = plt.subplots(figsize=(15, 8))

# plot the default as an area
ax.fill_between(e_range, np.zeros_like(t_merge), t_merge, label="Default", alpha=0.2)

ax.plot(
    e_range, t_merge_Mandel, 
    label='Mandel (2021)'
)

ax.set_xlabel("Eccentricity")
ax.set_ylabel("Merger Time [{0:latex}]".format(t_merge.unit))

ax.legend()

plt.show()

# %% [markdown]
# ## Time to interaction

# %% [markdown] raw_mimetype="text/restructuredtext"
# Here we look at how the time to merger differs from the time to Roche lobe overflow for DWDs. For the radius of Roche sphere $r_1$, we will use the formula proposed by [Eggleton (1983)](https://ui.adsabs.harvard.edu/abs/1983ApJ...268..368E/abstract):
# $$
# \frac{r_1}{a} = \frac{0.49 q^{2/3}}{0.6 q^{2/3} + \ln{(1+q^{1/3})}}\,, \qquad q=\frac{m_1}{m_2}\,.
# $$
#
# The radius of a WD is a function of its mass and effective temperature, $R=R(m,T_{\rm eff})$. We will use an implementation of that function based on a grid of WD evolution models: [MR-relation code](https://github.com/mahollands/MR_relation/tree/master).
#
# A DWD starts interacting when, for either of the components, its Roche sphere radius falls below its radius. It looks like it will first happen for the secondary: (i) its radius is larger due to the smaller mass, (ii) its Roche lobe is smaller.

# %%
from MR_relation import R_from_Teff_M
thickness='thin'

def RocheRadius(q):

    return 0.49*q**(2./3) / (0.6*q**(2./3) + np.log(1 + q**(1./3)))



# %% [markdown]
# ### Some visualizations of the functions for the Roche sphere radius and $M$-$R$ relation for WDs

# %%
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

ax1,ax2 = axes

q = np.logspace(-2,2,1000)
facRoche = RocheRadius(q)


masses = np.logspace(-3, np.log10(1.2), 100)


ax1.semilogx(q, facRoche)

ax1.set_ylim(0,1)
ax1.set_xlabel('$q$')
ax1.set_ylabel('$r_1/a$')

for temp in np.linspace(2000,22000,5):
    radii = R_from_Teff_M(temp, masses, thickness)
    ax2.plot(masses, radii, label='$T_{{\\rm eff}}={:.0f}$ K'.format(temp))

ax2.set_xlim(0, 1.25)
ax2.set_ylim(0, 0.04)
ax2.set_xticks(np.linspace(0,1.5,6))
ax2.set_yticks(np.linspace(0,0.04,5))
ax2.set_xlabel('WD mass ($M_\odot$)')
ax2.set_ylabel('WD radius ($R_\odot$)')
ax2.legend()




for ax in axes:
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)



# %% [markdown]
# For a specific example, consider $m_1=m_2=0.6M_\odot$ at $T_{\rm eff}=4000$ K.

# %%
m1 = m2 = 0.6   # Msun
Teff = 4000   # Kelvin
q = np.minimum(m1,m2)/np.maximum(m1,m2)

R1 = R_from_Teff_M(Teff, m1, thickness)
R1 *= u.Rsun
R2 = R_from_Teff_M(Teff, m2, thickness)
R2 *= u.Rsun

m1 *= u.Msun
m2 *= u.Msun


a0 = R1/RocheRadius(q)
f_orb0 = utils.get_f_orb_from_a(a=a0, m_1=m1, m_2=m2)

evol.get_t_merge_circ(m_1=m1, m_2=m2, f_orb_i=f_orb0).to(u.year)

# %%
