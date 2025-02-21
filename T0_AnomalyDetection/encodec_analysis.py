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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import os
import shutil

MEDIUM_SIZE = 14
BIGGER_SIZE = 18

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

# %%
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

#fig.tight_layout()
#fig.savefig('test.pdf')

# %%
from models import *
from config import Params

dataloader = DataLoader(Params.FILENAME)

num_features = len(dataloader.features)
seq_len = Params.SEQ_LEN
units = Params.UNITS


# %%
weights_file = os.path.join(Params.MODELS_FOLDER,'best_model.weights.h5')

init_gen = DataGenerator(dataloader, seq_len, batch_size=2, steps_per_epoch=1)

#test_gen.on_epoch_end()
X,_ = init_gen[0]

model = EncoDecLSTM(units,num_features)
_ = model(X)
model.load_weights(weights_file)

for layer in model.layers:
    print(f"Layer: {layer.name}")
    for var in layer.get_weights():
        print(f"\tWeights shape: {var.shape}")


# %%
batch_size = 2048

test_gen = DataGenerator(dataloader, seq_len, batch_size=batch_size, steps_per_epoch=1)
test_loss = keras.losses.mse

X,_ = test_gen[0]

raw_errors = model.predict(X) - X
errors = test_loss(X, model.predict(X))
batch_errors = np.array(
    tf.math.reduce_mean(errors, axis=-1)
)


# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.set_xscale('log')
ax.set_yscale('log')

bins = np.logspace(np.log10(batch_errors.min()), np.log10(batch_errors.max()), 100)
#bins=100
ax.hist(batch_errors, bins=bins)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('loss')
ax.set_ylabel('counts')

# %%
import corner

errors_per_feature = tf.math.reduce_mean(raw_errors, axis=1)
mean_per_feature = tf.math.reduce_mean(errors_per_feature, axis=0, keepdims=True)
std_per_feature = tf.math.reduce_std(errors_per_feature, axis=0, keepdims=True)    
errors_per_feature = (errors_per_feature - mean_per_feature)/std_per_feature

errors_per_feature = np.array(errors_per_feature)

num_toplot = 5
ranges = zip(-5.*np.ones(num_toplot), 5.*np.ones(num_toplot))

fig = corner.corner(
    errors_per_feature[:,:5],
    range=ranges
)

# %%
anomaly_ids = test_gen.unique_IDs[:batch_size][batch_errors>1]

anomalies = [test_gen.df_orig.get_group(group) for group in anomaly_ids]

len(anomalies)

# %%
aux = anomalies[0]

cols_to_include = ['time', 'mass1', 'radius1', 'Teff1', 'type1']

evol_sequence = pd.DataFrame()

for col in cols_to_include:
    cond = (aux[cols_to_include[:-1]]>=0).all(axis=1)
    evol_sequence[col] = aux[cond][col]


Tsun = 5780.  # Teff for the Sun
MVsun = 4.83    # absolute mag for the Sun (V-band)

evol_sequence['lum1'] = evol_sequence['radius1']**2 * (evol_sequence['Teff1']/Tsun)**4
evol_sequence['MV'] = MVsun - 2.5*np.log10(evol_sequence['lum1'])

cols_for_plot = ['time', 'lum1', 'MV', 'Teff1', 'type1']

plot_data = sorted(
    zip(*[evol_sequence[col] for col in cols_for_plot]),
    key=lambda x:x[0]
)

evol_sequence

# %%

_,lum,MV,Teff,_ = [el for el in zip(*plot_data)]
classes = dict(
    zip(
        list('OBAFGKM'),
        [30000., 9700., 7200., 5700., 4900., 3400., 2100.]
    )
)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,4))

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.invert_xaxis()
ax.invert_yaxis()

ax_twinx = ax.twinx()
ax_twiny = ax.twiny()
ax_twinx.set_yscale('log')
ax_twiny.set_xscale('log')
ax.set_xscale('log')


ax_twiny.scatter(Teff, MV)
ax_twinx.scatter(Teff, lum)

for i, txt in enumerate(range(len(Teff))):
    ax.annotate(txt, (Teff[i], MV[i]))

ymin,ymax = -10,15

ax_twiny.set_xlim(3e+5, 3e+3)
ax.set_xlim(3e+5, 3e+3)
ax.set_ylim(ymax, ymin)
ax_twinx.set_ylim(
    10**((MVsun-ymax)/2.5), 10**((MVsun-ymin)/2.5)
)

ax.set_xticks(list(classes.values()))
ax.set_xticklabels(list(classes.keys()))

ax_twiny.set_xlabel('effective temperature (K)')
ax.set_ylabel('$M_V$')
ax_twinx.set_ylabel('luminosity $(L_\\odot)$')

# %%
class_temps = np.array(list(classes.values()))

class_temps[1:]/class_temps[:-1]

# %%
class_temps

# %%
class_temps = np.array(
    [25000, 10000, 7500, 6000, 5000, 3500]
)

class_temps[1:]/class_temps[:-1]

# %%
