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
filename = './SEVN/MIST/setA/Z0.02/sevn_mist'

with open(filename, 'r') as f:
    for k,line in enumerate(f):
        if k>10:
            break
        print(line)

# %%
df = pd.read_csv(filename, skiprows=2, index_col=False)

# reduce the ranges of "time" and "Teff1"/"Teff2": convert to Gyr and kiloK, respectively;
conversion_facs = {
    'time': 1000.,
    'Teff1': 1000.,
    'Teff2': 1000.
}

for col,fac in conversion_facs.items():
    df[col] /= fac

print(df.min())
print(df.max())

# (-1) is NaN and (-2) is missing which should be encoded as empty strings ''
df.replace('^\s*$', -2., inplace=True, regex=True)
df.fillna(-1., inplace=True)

df.head()

# %%
# 1. one-hot encoding for the categorical variables [event,type1,type2]
# 2. convert time to Gyr
columns = ['event', 'type1', 'type2']

for col in columns:
    df[col] = df[col].astype(str)

df_onehot = pd.get_dummies(df, columns=columns, dtype=float)

df_onehot.head()

# %%
unique_IDs = df_onehot.ID.unique()
unique_UIDs = df_onehot.UID.unique()

print(
    '# IDs: {:d}\n# unique IDs: {:d}'.format(
        len(unique_IDs), len(unique_UIDs)
    )
)

# %%
k = 0
systems = {}

for unique_id in unique_UIDs:
    if k > 9:
        break
    ids = df_onehot[df_onehot.UID == unique_id].ID.unique()
    if len(ids) == 1:
        continue
    systems[str(unique_id)] = ids
    k += 1

print('The first {:d} systems with 2+ IDs for one UID\n'.format(k))
print('UID:\t\t IDs')
for uid,ids in systems.items():
    print('{}\t'.format(uid), *ids)

# %%
seq_len = 5

features = df_onehot.columns.values[2:]
print('There is {:d} features:'.format(len(features)), ', '.join(features))

rng = np.random.default_rng()

random_id = rng.choice(unique_IDs)

random_sys = (df_onehot[df_onehot.ID == random_id]).sort_values('time')[features]

random_sys

# %%
len_sys = len(random_sys)

random_shift = -(seq_len-1) + rng.integers(2*seq_len-1)
shifted_sys = random_sys.shift(random_shift, fill_value=-2.)

random_start = rng.integers(len_sys - seq_len + 1)
shifted_sys.iloc[random_start:(random_start+seq_len)]

# %%
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class DataGenerator(keras.utils.Sequence):
    def __init__(self, filename, batch_size, steps_per_epoch, seq_len=5, seed=None, shuffle=True, **kwargs):

        super().__init__(**kwargs)
        self.shuffle = shuffle
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed=seed)
        self.seq_len = seq_len
        
        df = pd.read_csv(filename, skiprows=2, index_col=False)

        # reduce the ranges of "time" and "Teff1"/"Teff2": convert to Gyr and kiloK, respectively;
        conversion_facs = {
            'time': 1000.,
            'Teff1': 1000.,
            'Teff2': 1000.
        }
        log_columns = ['semiMajor', 'radius1', 'radius1']
        for col in log_columns:
            df[col] = np.log10(df[col])
        
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
        self.features = df_onehot.columns.values[2:]
        self.unique_IDs = df_onehot.ID.unique()

        if self.shuffle:
            rng.shuffle(self.unique_IDs)
        

    def __len__(self):
        return self.steps_per_epoch
        

    def __getitem__(self, index):
        
        sys_indices = self.unique_IDs[index*self.batch_size:(index+1)*self.batch_size]

        batches = []
        for random_id in sys_indices:
            random_sys = (self.df_onehot[self.df_onehot.ID == random_id]).sort_values('time')[self.features]
            len_sys = len(random_sys)
            random_shift = -(self.seq_len-1) + self.rng.integers(2*self.seq_len-1)
            shifted_sys = random_sys.shift(random_shift, fill_value=-2.)
            
            random_start = self.rng.integers(len_sys - self.seq_len + 1)
            batches.append(
                shifted_sys.iloc[random_start:(random_start+self.seq_len)].to_numpy(dtype=np.float32),
            )
            
        
        return tf.convert_to_tensor(batches),tf.convert_to_tensor(batches)

    def on_epoch_end(self):
        if self.shuffle:
            rng.shuffle(self.unique_IDs)


class EncoDecLSTM(keras.Model):
    def __init__(self, units):
        super(EncoDecLSTM, self).__init__()
        self.encoder = layers.LSTM(units, return_state=True)
        self.lstm_cell = layers.LSTMCell(units)
        self.dense1 = layers.Dense(units, activation='relu') 

    def build(self, input_dim):
        
        self.input_shape = input_dim
        self.dense2 = layers.Dense(input_dim[-1], activation=None)
        #self.decoder = layers.LSTM(input_dim[-1], activation=None, return_sequences=True, return_state=False)
        
    def call(self, input_tensor):

        output, state_h, state_c = self.encoder(input_tensor)

        sequence = []
        for i in range(self.input_shape[1]):
            output,(state_h,state_c) = self.lstm_cell(input_tensor[:,i,:], [state_h, state_c])
            # output = self.dense1(output)
            # output = self.dense2(output)
            sequence.append(self.dense2(self.dense1(output)))

        sequence = tf.transpose(tf.convert_to_tensor(sequence), perm=[1,0,2])
        
        # return self.decoder(sequence)
        return sequence
    
    
    def model(self):
        x = layers.Input(shape=(None,1))
        return keras.Model(inputs=[x], outputs=self.call(x))

# %%
train_gen = DataGenerator(filename, batch_size=64, steps_per_epoch=100)
val_gen = DataGenerator(filename, batch_size=256, steps_per_epoch=1)

model = EncoDecLSTM(128)


# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.), 
    loss=keras.losses.MeanSquaredError(),
)


history = model.fit(
    x=train_gen, validation_data=val_gen,
    epochs=5, verbose=1
)

# %%
