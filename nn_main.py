##!/usr/bin/env python

#import sys
#sys.path.append('/home/oscar/analyse/nn_trade')


import matplotlib.pyplot as plt
import tensorflow as tf
import inspect
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats.stats import pearsonr
#from nn_data import *
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
import ast
import matplotlib._color_data as mcd
import seaborn as sns


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path

from nn_tools import *
from nn_datacreate import *
from nn_datagen import *
from NNDCube import *


class NN_Network():

    def __init__(self, ndef, data_t, data_v):

        self.ndef = ndef
        self.data_t = data_t
        self.data_v = data_v
        self.exp = ndef.exp

        self.define()

    def define(self):

        # First Parallel Network
        print (self.ndef.network1A[0], self.data_t.A_nvars_i, self.data_t.A_lookback)
        input_tensor = self._input(self.ndef.network1A[0], self.data_t.A_nvars_i, self.data_t.A_lookback)
        xxx = input_tensor
        for net in self.ndef.network1A:
            xxx = self._layer(net,xxx)

        # Second Parallel Network
        if self.ndef.n_nets > 1:

            input_tensorB = self._input(self.ndef.network1B[0], self.data_t.B_nvars_i, self.data_t.B_lookback)
            xxxB = input_tensorB
            for net in self.ndef.network1B:
                xxxB = self._layer(net, xxxB)
            xxx = layers.concatenate([xxx,xxxB], axis=-1)

            if self.ndef.n_nets >2:

                input_tensorC = self._input(self.ndef.network1C[0], self.data_t.C_nvars_i, self.data_t.C_lookback)
                xxxC = input_tensorC
                for net in self.ndef.network1C:
                    xxxC = self._layer(net, xxxC)
                xxx = layers.concatenate([xxx,xxxC], axis=-1)

                for net in self.ndef.network2:
                    xxx = self._layer(net, xxx)
                self.model = Model([input_tensor,input_tensorB,input_sensorC], xxx)

            else:

                for net in self.ndef.network2:
                    xxx = self._layer(net, xxx)
                self.model = Model([input_tensor,input_tensorB], xxx)

        else:
            self.model = Model(input_tensor, xxx)

        self.model.summary()

    def run(self, save=False, ename=None):

        print(self.model.output_shape)
        print ('******  Using Optimizer', self.ndef.optimizer)


        if self.ndef.optimizer == 'rmsprop':
            if self.ndef.rmsprop>0 :
                self.ndef.optimizer = tf.keras.optimizers.RMSprop(self.ndef.rmsprop)
            else:
                self.ndef.optimizer = tf.keras.optimizers.RMSprop()

        print ('Using Optimizer', self.ndef.optimizer)

        self.model.compile(loss='mean_squared_error',
                  optimizer=self.ndef.optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

        self.train_steps = (self.data_t.nvals_i)//self.data_t.batchsize
        self.val_steps = (self.data_v.nvals_i)//self.data_v.batchsize

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=self.ndef.patience)
        home = str(Path.home())
        if ename != None:
            dir = os.path.join(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/',ename)
            if not os.path.exists(dir): os.makedirs(dir)
        else:
            dir = os.path.join(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set)

        mc = ModelCheckpoint(dir+ '/'+'best_model.h5',
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True)

#       dir=home+'/analyse/nn_exps/' + self.name
#       tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=dir,histogram_freq=1)
        self.history = self.model.fit_generator(self.data_t.gen, steps_per_epoch=self.train_steps, epochs=self.ndef.epochs,
                                  validation_data = self.data_v.gen,
                                  validation_steps = self.val_steps,
                                  callbacks=[es,mc], verbose=0)
#                                  callbacks=[es,mc,tensorboard_cb])
                                  #callbacks=[PrintDot()])
        if save: self.save(ename=ename)


    def save(self, ename=None):

        # save nn output
        home = str(Path.home())
        dir = self.ndef.basedir+'/' + self.ndef.exp + '/' + self.ndef.set
        if ename != None:
            dir = os.path.join(dir,ename)
            if not os.path.exists(dir): os.makedirs(dir)



        jsonfile = os.path.join(dir, "model.json")
        h5file = os.path.join(dir, "end_model.h5")
        histfile = os.path.join(dir, "history.npy")
        model_json = self.model.to_json()
        with open(jsonfile, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(h5file)
        history_dict = self.history.history
        np.save(histfile, history_dict)

    def _layer(self, net, xxx):
        if net[0] == 'Dense':
            if len(net) < 3:
                xxx = layers.Dense(net[1])(xxx)
            else:
                xxx = layers.Dense(net[1], activation=net[2])(xxx)
        if net[0] == 'Dropout': xxx = layers.Dropout(net[1])(xxx)
        if net[0] == 'Flatten': xxx = layers.Flatten()(xxx)
        if net[0] == 'LSTM': xxx = layers.LSTM(net[1], return_sequences=net[2]['return_sequences'])(xxx)
        if net[0] == 'GRU': xxx = layers.GRU(net[1], return_sequences=net[2]['return_sequences'])(xxx)
        #           if net[0] == 'TimeDistributed' and net[1] == 'Dense': xxx =
        #                    layers.TimeDistributed(layers.Dense(net[3], activation=net[2]))(xxx)
        return xxx

    def _input(self, net, nvars, lookback):
        if net[0] == 'Dense':
            input_tensor = Input(shape=nvars)
        if net[0] == 'Flatten':
            input_tensor = Input(shape=(lookback, nvars))
        if net[0] == 'LSTM':
            input_tensor = Input(shape=(lookback, nvars))
        if net[0] == 'GRU':
            input_tensor = Input(shape=(lookback, nvars))
        return input_tensor

def nn_main(ensemble,ndef, update_data=False, run_nn=True):

    if ensemble==0 or update_data == True:
        xx = NNDCube(ndef.set, ndef.dates_train, ndef.lookbackA, 't', 'A', ndef.varsiA, ndef.varso, ndef.exp)
        xx = NNDCube(ndef.set, ndef.dates_val, ndef.lookbackA, 'v', 'A', ndef.varsiA, ndef.varso, ndef.exp)
        xx = NNDCube(ndef.set, ndef.dates_ind, ndef.lookbackA, 'i', 'A', ndef.varsiA, ndef.varso, ndef.exp)
        if ndef.n_nets == 2: 
            xx = NNDCube(ndef.set, ndef.dates_train, ndef.lookbackB, 't', 'B', ndef.varsiB, None, ndef.exp)
            xx = NNDCube(ndef.set, ndef.dates_val, ndef.lookbackB, 'v', 'B', ndef.varsiB, None, ndef.exp)
            xx = NNDCube(ndef.set, ndef.dates_ind, ndef.lookbackB, 'i', 'B', ndef.varsiB, None, ndef.exp)
        if ndef.n_nets == 3: 
            xx = NNDCube(ndef.set, ndef.dates_train, ndef.lookbackC, 't', 'C', ndef.varsiB, None, ndef.exp)
            xx = NNDCube(ndef.set, ndef.dates_val, ndef.lookbackC, 'v', 'C', ndef.varsiB, None, ndef.exp)
            xx = NNDCube(ndef.set, ndef.dates_ind, ndef.lookbackC, 'i', 'C', ndef.varsiB, None, ndef.exp)
 
    if run_nn:
        data_gen_t = NN_DataGen(ndef, 't', shuffle = True, batchsize=ndef.batchsize)
        data_gen_v = NN_DataGen(ndef, 'v', shuffle = False, batchsize=ndef.batchsize)

        model = NN_Network(ndef, data_gen_t, data_gen_v)
        model.run(save=True,ename='en'+str(ensemble))
