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
#plt.rc('font',**{'family':'serif','serif':['Times']})
#plt.rc('text', usetex=True)


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
epochs = Params.EPOCHS
steps_per_epoch = Params.STEPS_PER_EPOCH

train_gen = DataGenerator(dataloader, seq_len, batch_size=64, steps_per_epoch=steps_per_epoch)
val_gen = DataGenerator(dataloader, seq_len, batch_size=64, steps_per_epoch=1)

# %%

model = EncoDecLSTM(units, num_features)

if (os.path.exists(Params.MODELS_FOLDER)):
    shutil.rmtree(Params.MODELS_FOLDER)
os.makedirs(Params.MODELS_FOLDER, exist_ok=True)

cp_callback = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(Params.MODELS_FOLDER,'best_model.weights.h5'),
                save_weights_only=True,
                save_best_only=True,
                verbose=1
        )


# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss=keras.losses.MeanSquaredError(),
)


history = model.fit(
    x=train_gen, validation_data=val_gen,
    epochs=epochs, verbose=1,
    callbacks=[cp_callback]
)

# %%
# for layer in model.layers:
#     print(f"Layer: {layer.name}")
#     for var in layer.get_weights():
#         print(f"\tWeights shape: {var.shape}")

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

for key,val in history.history.items():

    ax.semilogy(np.arange(epochs) + 1, val, label=key)

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('epoch')
ax.set_ylabel('loss')

fig.tight_layout()
#fig.savefig('test.pdf')

# %%
