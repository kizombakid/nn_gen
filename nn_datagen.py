
import inspect
import ast

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path

from nn_tools import *
from nn_datacreate import *

class NN_DataGen():

    def __init__(self, ndef, type, shuffle = False, batchsize=0):

        self.ndef = ndef
        home = str(Path.home())
        dir = ndef.basedir+'/' + ndef.exp +'/'+ndef.set
        self.A_vali = np.load(os.path.join(dir, 'dataA_vali_'+type+'.npy'))
        self.valo = np.load(os.path.join(dir, 'data_valo_'+type + '.npy'))
        si = np.shape(self.A_vali)
        self.nvals_i = si[0]
        self.A_lookback = si[1]
        self.A_nvars_i = si[2]
        si = np.shape(self.valo)
        self.nvars_o = si[1]

        if ndef.n_nets >1:
            self.B_vali = np.load(os.path.join(dir, 'dataB_vali_'+type+'.npy'))
            si = np.shape(self.B_vali)
            self.B_lookback = si[1]
            self.B_nvars_i = si[2]

        if ndef.n_nets >2:
            self.C_vali = np.load(os.path.join(dir, 'dataC_vali_'+type+'.npy'))
            si = np.shape(self.C_vali)
            self.C_lookback = si[1]
            self.C_nvars_i = si[2]

        if batchsize > 0:
            self.batchsize = batchsize
        else:
            self.batchsize = self.nvals_i

        if self.batchsize > self.nvals_i: self.batchsize = self.nvals_i

        self.gen = self._generator(shuffle = shuffle)

    def _generator(self, shuffle =  False):
        i=0
        while 1:
            if shuffle:
                rows = np.random.randint(0,self.nvals_i,size=self.batchsize)
            else:
                if i+self.batchsize >= self.nvals_i:
                    i = 0
                rows = np.arange(i,i+self.batchsize)
                i += len(rows)

            samplesA = np.zeros((len(rows),self.A_lookback,self.A_nvars_i))
            targets = np.zeros((len(rows),self.nvars_o))
            for j, row in enumerate(rows):
                samplesA[j,:,:] = self.A_vali[row,:,:]
                targets[j] = self.valo[row,:]

            if self.ndef.n_nets > 1:
                samplesB = np.zeros((len(rows),self.B_lookback,self.B_nvars_i))
                for j, row in enumerate(rows):
                    samplesB[j,:,:] = self.B_vali[row,:,:]

            if self.ndef.n_nets > 2:
                samplesC = np.zeros((len(rows),self.C_lookback,self.C_nvars_i))
                for j, row in enumerate(rows):
                    samplesC[j,:,:] = self.C_vali[row,:,:]

            if self.ndef.n_nets == 1: yield samplesA, targets
            if self.ndef.n_nets == 2: yield [samplesA,samplesB], targets
            if self.ndef.n_nets == 3: yield [samplesA,samplesB,samplesC], targets
