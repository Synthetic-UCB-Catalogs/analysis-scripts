from functools import partial
import numpy as np
from config import Params

import jax.numpy as jnp
from jax import vmap, random, jit
from jax.scipy import optimize


@jit
def zCDFInvVec(Xiz):
    zCoord = -jnp.log(1-Xiz)
    return zCoord


@jit
def RCDF_residual(rr, Xir):
    Res = (1-jnp.exp(-rr))-rr*jnp.exp(-rr)-Xir
    return jnp.sum(Res**2)


@jit
def RCDFInv(Xir):
    x0 = jnp.array([1.0])
    args = (Xir,)
    return optimize.minimize(RCDF_residual, x0, args, method="BFGS")


RCDFInvVec = vmap(RCDFInv)


def SamplePop(Hr, Hz, size, key, max_age=12.):

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

    return XSet, YSet, ZSet, AgeSet


if __name__ == "__main__":

    filename = Params.FILENAME
    filename += '.npz'

    Hr = Params.HR   # kpc
    Hz = Params.HZ  # kpc
    num = Params.SIZE

    seed = Params.SEED
    key = random.key(seed)

    X, Y, Z, Age = SamplePop(Hr, Hz, num, key)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    Age = np.array(Age)

    np.savez(filename, XSet=X, YSet=Y, ZSet=Z, AgeSet=Age)
