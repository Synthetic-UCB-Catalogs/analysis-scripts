# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from jax import config
import PositionsSeparable_np as np_sampler
from time import time
from config import Params
from jax.scipy import optimize
from jax import random
from jax import grad, jit, vmap, jacfwd, jacrev
import jax.numpy as jnp
from scipy.optimize import root_scalar
import numpy as np
import scipy
import matplotlib.pyplot as plt

MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rcdefaults()

# plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text', usetex=True)

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# %matplotlib inline

# %%


config.update("jax_enable_x64", True)

seed = 1701
key = random.key(seed)

# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.plot([],[])

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)


# ax.set_xlabel('time (s)')
# ax.set_ylabel('amplitude')

# ax.legend();

# %% [markdown]
# ### Separable

# %%
def zCDFInvVec(Xiz):
    zCoord = -jnp.log(1-Xiz)
    return zCoord


def RCDF_residual(rr, Xir):
    Res = (1-jnp.exp(-rr))-rr*jnp.exp(-rr)-Xir
    return jnp.sum(Res**2)


def RCDFInv(Xir):
    x0 = jnp.array([1.0])
    args = (Xir,)
    return optimize.minimize(RCDF_residual, x0, args, method="BFGS")


RCDFInvVec = vmap(RCDFInv)


def SamplePop(Hr, Hz, size, key=key, max_age=12.):

    key, subkey = random.split(key)
    RRandSet = random.uniform(key, shape=(size,))

    key, subkey = random.split(key)
    ZRandSet = random.uniform(key, shape=(size,))

    key, subkey = random.split(key)
    ZSignSet = jnp.sign(2*random.uniform(key, shape=(size,)) - 1)

    RSet_result = RCDFInvVec(RRandSet)
    if jnp.all(RSet_result.success):
        RSet = Hr*RSet_result.x.reshape(size,)
    else:
        print('The radial solution did not converge')
        return
    ZSet = Hz*ZSignSet*zCDFInvVec(ZRandSet)

    key, subkey = random.split(key)
    ThSet = 2*jnp.pi*random.uniform(key, shape=(size,))
    XSet = RSet*jnp.cos(ThSet)
    YSet = RSet*jnp.sin(ThSet)

    key, subkey = random.split(key)
    AgeSet = max_age*random.uniform(key, shape=(size,))

    return jnp.array([XSet, YSet, ZSet])


# Sol  = root_scalar(RCD,bracket=(0.0001*Hr,20*Hr))
# if Sol.converged:
#     R      = Sol.root
# else:
#     print('The radial solution did not converge')
#     return np.nan
# return R

# %%

rng = np.random.default_rng()

Hr = Params.HR   # kpc
Hz = Params.HZ  # kpc

sizes = np.logspace(10, 20, 11, base=2)
times_jax = []
times_np = []

for num in sizes:

    num = int(num)

    begin = time()

    seed = rng.integers(10000)
    key = random.key(seed)
    data = SamplePop(Hr, Hz, num, key)
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)

    times_jax.append(time()-begin)

    print('{:d} took {:.2f} s.'.format(num, times_jax[-1]))

times_jax = np.array(times_jax)


# %%

for num in sizes:

    num = int(num)

    begin = time()
    X, Y, Z, _ = np_sampler.SamplePop(num, Hr, Hz)
    times_np.append(time()-begin)

    print('{:d} took {:.2f} s.'.format(num, times_np[-1]))

times_np = np.array(times_np)

# %%
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4))

ax.loglog(sizes, times_jax, label='JAX')
ax.loglog(sizes, times_np, label='NumPy')

ax.grid(True, linestyle=':', linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both', length=3, width=0.5,
               which='both', direction='in', pad=10)


ax.set_xlabel('sample size')
ax.set_ylabel('time (s)')

ax.legend()

# %%
coords = ['X', 'Y', 'Z']

for i in range(3):
    for j in range(i+1, 3):

        fig, ax = plt.subplots(ncols=1, nrows=1)

        ax.hist2d(data[i], data[j], bins=100)

        # ax.grid(True,linestyle=':',linewidth='1.')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params('both', length=3, width=0.5,
                       which='both', direction='in', pad=10)

        ax.set_xlabel(coords[i])
        ax.set_ylabel(coords[j])

        ax.set_aspect('equal')

# ax.legend();

# %%
filename = 'PositionsSeparable_data.npz'

X, Y, Z = data

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)


np.savez(filename, XSet=X, YSet=Y, ZSet=Z)

# %%
