import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class DataLoader:
    def __init__(self, filename):
        
        df = pd.read_csv(filename, skiprows=2, index_col=False)
        self.df_orig = df

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
        self.id_counts = self.df_onehot['ID'].value_counts()
        
        self.df_grouped = self.df_onehot.sort_values(by=['time']).groupby('ID')
        self.df_orig = self.df_orig.sort_values(by=['time']).groupby('ID')

    
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
        self.df_orig = dataloader.df_orig

        if self.shuffle:
            self.rng.shuffle(self.unique_IDs)
        
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


class EncoDecBottleNeck(keras.Model):
    def __init__(self, units, features, low_dim_list, **kwargs):
        super(EncoDecLSTM, self).__init__(**kwargs)
        self.units = units
        self.features = features
        self.low_dim_list = low_dim_list

        self.encoder = layers.LSTM(self.units, return_state=True, name='encoder') 
        self.lstm_cell = layers.LSTMCell(self.units, name='decoder_cell')
        self.dense1 = layers.Dense(self.units, activation='relu', name='projector1')
        self.dense2 = layers.Dense(self.features, activation=None, name='projector2')

        self.low_dim_block = [layers.Dense(hidden_units, activation='relu') for hidden_units in self.low_dim_list]
        self.low_dim_block.append(layers.Dense(self.units, activation='relu'))
        
    def build(self, input_dim):

        self.input_dim = input_dim
        
    def call(self, input_tensor, training=False):

        output, state_h, state_c = self.encoder(input_tensor, training=training)
        for _layer in self.low_dim_block:
            output = _layer(output)

        sequence = []
        for i in range(self.input_dim[1]):
            output,(state_h,state_c) = self.lstm_cell(output, [state_h, state_c], training=training)
            sequence.append(output)

        sequence = tf.transpose(tf.convert_to_tensor(sequence), perm=[1,0,2])
        
        return self.dense2(self.dense1(sequence))

