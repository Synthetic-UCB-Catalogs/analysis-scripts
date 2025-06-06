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
weights_file = os.path.join(Params.MODELS_FOLDER,'baseline_best_model.weights.h5')

init_gen = DataGenerator(dataloader, seq_len, batch_size=2, steps_per_epoch=1)

#test_gen.on_epoch_end()
X,_ = init_gen[0]


model = EncoDecLSTM(units,num_features)
_ = model(X)
model.load_weights(weights_file)

print(model.summary())

for layer in model.layers:
    print(f"Layer: {layer.name}")
    for var in layer.get_weights():
        print(f"\tWeights shape: {var.shape}")


# %%
batch_size = 2048*8

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
ax.hist(batch_errors, bins=bins, density=True)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('mean squared error (MSE)')
ax.set_ylabel('PDF')

fig.tight_layout()
fig.savefig('PDF_errors.png')

# %%
import corner

errors_per_feature = tf.math.reduce_mean(raw_errors, axis=1)
mean_per_feature = tf.math.reduce_mean(errors_per_feature, axis=0, keepdims=True)
std_per_feature = tf.math.reduce_std(errors_per_feature, axis=0, keepdims=True)    
errors_per_feature = (errors_per_feature - mean_per_feature)/std_per_feature

errors_per_feature = np.array(errors_per_feature)

num_toplot = 3
ranges = zip(-5.*np.ones(num_toplot), 5.*np.ones(num_toplot))

fig = corner.corner(
    errors_per_feature[:,:3],
    range=ranges
)

fig.tight_layout()
fig.savefig('corner_features.png')

# %%
anomaly_ids = test_gen.unique_IDs[:batch_size][batch_errors>1]

anomalies = [test_gen.df_orig.get_group(group) for group in anomaly_ids]

len(anomalies)

# %%
from config import *

cols_to_view = ['time', 'event', 'mass1', 'radius1', 'Teff1', 'type1', 'mass1', 'radius2', 'Teff2', 'type2']

# --- Loop through DataFrames and describe ---
for i, df in enumerate(anomalies):
    print(f"\n{'='*15} DataFrame {i+1} {'='*15}")
    # For display in environments like Jupyter:
    try:
        from IPython.display import display
        display(df[cols_to_view])
    except ImportError:
        print(df[cols_to_view].to_string())

    print(f"\n--- Descriptions for DataFrame {i+1} ---")
    for index, row in df.iterrows():
        event_code_val = row['event']
        type1_val = row['type1']
        type2_val = row['type2']

        event_desc = get_event_description(event_code_val)
        type1_desc = get_component_description(type1_val)
        type2_desc = get_component_description(type2_val)

        print(f"\nTime: {row['time']}:")
        print(f"  Event ({event_code_val}): {event_desc}")
        print(f"  Component 1 Type ({type1_val}): {type1_desc}")
        print(f"  Component 2 Type ({type2_val}): {type2_desc}")
    print(f"\n{'='*40}")

# %%
for i,df in enumerate(anomalies):
    plot_hr_diagram_from_df(
        df[cols_to_view],
        title="HR Diagram {:d}".format(i+1),
        mv_tick_step=2.,
        legend_pos='top_center_horizontal',
        plot_width=850, 
        font_size_general=16
    )

# %%
