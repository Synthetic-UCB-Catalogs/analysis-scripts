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
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class DataLoader:
    def __init__(self, filename):
        
        df = pd.read_csv(filename, skiprows=2, index_col=False)

        # reduce the ranges of "time" and "Teff1"/"Teff2": convert to Gyr and kiloK, respectively;
        conversion_facs = {
            'time': 1000.,
        }
        for col,fac in conversion_facs.items():
            df[col] /= fac
        
        
        # (-1) is NaN and (-2) is missing which should be encoded as empty strings ''
        df.replace('^\s*$', -2., inplace=True, regex=True)
        df.fillna(-1., inplace=True)

        # 1. one-hot encoding for the categorical variables [event,type1,type2]
        # 2. convert time to Gyr
        columns = ['event', 'type1', 'type2']
        for col in columns:
            df[col] = df[col].astype(str)
        
        self.df_onehot = pd.get_dummies(df, columns=columns, dtype=float)

        log_columns = ['mass1', 'mass2', 'radius1', 'radius2', 'semiMajor', 'Teff1', 'Teff2', 'massHecore1', 'massHecore2']
        log_columns_Re = [f'log_{col}_Re' for col in log_columns]
        log_columns_Im = [f'log_{col}_Im' for col in log_columns]
        eps = 1e-16
        for col,im_col in zip(log_columns,log_columns_Im):
            complex_log = np.log10(eps + self.df_onehot[col] + 0*1j)
            self.df_onehot[im_col] = np.imag(complex_log)
            self.df_onehot[col] = np.real(complex_log)
        self.df_onehot.rename(columns=dict(zip(log_columns,log_columns_Re)), inplace=True)
        
        self.features = self.df_onehot.columns.values[2:]
        self.unique_IDs = self.df_onehot.ID.unique()
        self.df_grouped = self.df_onehot.sort_values(by=['time']).groupby('ID')
        self.id_counts = self.df_onehot['ID'].value_counts()
    

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataloader, seq_len, batch_size, steps_per_epoch, seed=None, shuffle=True, **kwargs):

        super().__init__(**kwargs)
        self.shuffle = shuffle
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed=seed)
        self.seq_len = seq_len

        self.unique_IDs = dataloader.unique_IDs
        self.id_counts = dataloader.id_counts
        self.features = dataloader.features
        self.df_grouped = dataloader.df_grouped
        

    def __len__(self):
        return self.steps_per_epoch
        

    def __getitem__(self, index):
        
        sys_indices = self.unique_IDs[index*self.batch_size:(index+1)*self.batch_size]

        start_seq = self.rng.integers(0, self.id_counts[sys_indices] - self.seq_len + 1)
        end_seq = start_seq + self.seq_len
        
        batches = [
            self.df_grouped.get_group(group)[self.features].iloc[start:end].to_numpy()\
                for group,start,end in zip(sys_indices,start_seq,end_seq)
        ]

        
        return tf.convert_to_tensor(batches),tf.convert_to_tensor(batches)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.unique_IDs)

class EncoDecLSTM(keras.Model):
    def __init__(self, units, features, **kwargs):
        super(EncoDecLSTM, self).__init__(**kwargs)
        self.units = units
        self.features = features
        self.encoder = layers.LSTM(self.units, return_state=True, name='encoder') 
        self.lstm_cell = layers.LSTMCell(self.units, name='decoder_cell')
        self.dense1 = layers.Dense(self.units, activation='relu', name='projector1')
        self.dense2 = layers.Dense(self.features, activation=None, name='projector2')
        
    def build(self, input_dim):

        self.input_dim = input_dim
        
    def call(self, input_tensor, training=False):

        output, state_h, state_c = self.encoder(input_tensor, training=training)

        sequence = []
        for i in range(self.input_dim[1]):
            output,(state_h,state_c) = self.lstm_cell(output, [state_h, state_c], training=training)
            sequence.append(output)

        sequence = tf.transpose(tf.convert_to_tensor(sequence), perm=[1,0,2])
        
        return self.dense2(self.dense1(sequence))



# %%
filename = './SEVN/MIST/setA/Z0.02/sevn_mist'

dataloader = DataLoader(filename)

# %%
seq_len = 5
num_features = len(dataloader.features)

units = 128
epochs = 20
steps_per_epoch = 100


train_gen = DataGenerator(dataloader, seq_len, batch_size=64, steps_per_epoch=steps_per_epoch)
val_gen = DataGenerator(dataloader, seq_len, batch_size=64, steps_per_epoch=1)

model = EncoDecLSTM(units, num_features)

MODELS_FOLDER = './chkpoints_units={:02d}_epochs={:02d}'.format(units, epochs)
LOGS_FOLDER = './logs'

if (os.path.exists(MODELS_FOLDER)):
    shutil.rmtree(MODELS_FOLDER)
os.mkdir(MODELS_FOLDER)

if (os.path.exists(LOGS_FOLDER)):
    shutil.rmtree(LOGS_FOLDER)
os.mkdir(LOGS_FOLDER)

cp_callback = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODELS_FOLDER,'best_model.weights.h5'),
                save_weights_only=True,
                save_best_only=True,
                verbose=1
        )


# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.), 
    loss=keras.losses.MeanSquaredError(),
)


history = model.fit(
    x=train_gen, validation_data=val_gen,
    epochs=epochs, verbose=1,
    callbacks=[cp_callback]
)

# %%
for layer in model.layers:
    print(f"Layer: {layer.name}")
    for var in layer.get_weights():
        print(f"\tWeights shape: {var.shape}")

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

for key,val in history.history.items():

    ax.plot(np.arange(epochs) + 1, val, label=key)

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
weights_file = os.path.join(MODELS_FOLDER,'best_model.weights.h5')

test_gen = DataGenerator(dataloader, seq_len, batch_size=2048, steps_per_epoch=1)
test_loss = keras.losses.mse

test_gen.on_epoch_end()
X,_ = test_gen[0]

model_reconstructed = EncoDecLSTM(units,num_features)
#_ = model_reconstructed(layers.Input(shape=(seq_len,num_features)))
_ = model_reconstructed(X)
model_reconstructed.load_weights(weights_file)

for layer in model_reconstructed.layers:
    print(f"Layer: {layer.name}")
    for var in layer.get_weights():
        print(f"\tWeights shape: {var.shape}")



# %%

raw_errors = model_reconstructed.predict(X) - X
errors = test_loss(X, model_reconstructed.predict(X))
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
