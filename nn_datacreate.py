#!/usr/bin/env python
#from __future__ import absolute_import, division, print_function

import numpy as np
from nn_tools import *
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path

class NN_DataCreate():

    def __init__(self, dname_in, dname_out, ndef, varsi, lookback, codes, output=False, varso=None):

        for nn, code in enumerate(codes):
            self._readcode(dname_in, code, varsi, ndef.dates_train, ndef.dates_val, ndef.dates_ind, lookback,varso=varso)
            if nn == 0:
                vali_t = self.vali_t
                vali_v = self.vali_v
                vali_i = self.vali_i
                dates_t = self.dates_t
                dates_v = self.dates_v
                dates_i = self.dates_i
                if varso != None:
                    valo_t = self.valo_t
                    valo_v = self.valo_v
                    valo_i = self.valo_i
            else:
                vali_t = np.append(vali_t, self.vali_t, axis=0)
                vali_v = np.append(vali_v, self.vali_v, axis=0)
                vali_i = np.append(vali_i, self.vali_v, axis=0)
                dates_t = np.apped(dates_t, self,dates_t, axis=0)
                dates_v = np.apped(dates_v, self,dates_v, axis=0)
                dates_i = np.apped(dates_i, self,dates_i, axis=0)

                if varso != None :
                    valo_t = np.append(valo_t, self.valo_t, axis=0)
                    valo_v = np.append(valo_v, self.valo_v, axis=0)
                    valo_i = np.append(valo_v, self.valo_i, axis=0)

        home = str(Path.home())
        dir = ndef.basedir + '/' + ndef.exp+'/'+ndef.set
        if not os.path.exists(dir): os.makedirs(dir)
        np.save(os.path.join(dir, 'data'+dname_out+'_vali_t.npy'), np.float32(vali_t))
        np.save(os.path.join(dir, 'data'+dname_out+'_vali_v.npy'), np.float32(vali_v))
        np.save(os.path.join(dir, 'data'+dname_out+'_vali_i.npy'), np.float32(vali_i))

        np.save(os.path.join(dir, 'data'+dname_out+'_dates_t.npy'), dates_t)
        np.save(os.path.join(dir, 'data'+dname_out+'_dates_v.npy'), dates_v)
        np.save(os.path.join(dir, 'data'+dname_out+'_dates_i.npy'), dates_i)


        if varso != None:
            np.save(os.path.join(dir, 'data_valo_t.npy'), np.float32(valo_t))
            np.save(os.path.join(dir, 'data_valo_v.npy'), np.float32(valo_v))
            np.save(os.path.join(dir, 'data_valo_i.npy'), np.float32(valo_i))

        print ('Data ', dname_out, 'Created')

    def _readcode (self, dname_in, code, varsi, dates_train, dates_val, dates_ind, lookback, varso=None):

        home = str(Path.home())
        dir = home + '/analyse/nn_data/' + dname_in

        dates = np.load(os.path.join(dir, code + '_dates.npy'))
        if dates[0] > dates_train[0]: ipt1 = 0
        if dates[-1] < dates_ind[1]: ipi2 = len(dates)-1
        print ('dates ',dates[0],dates[-1])
        print (dates_train)
        for n in range(1,len(dates)):
            if dates[n] >= dates_train[0] and dates[n-1] < dates_train[0]: ipt1 = n
            if dates[n] >= dates_train[1] and dates[n-1] < dates_train[1]: ipt2 = n
            if dates[n] >= dates_val[0] and dates[n-1] < dates_val[0]: ipv1 = n
            if dates[n] >= dates_val[1] and dates[n-1] < dates_val[1]: ipv2 = n
            if dates[n] >= dates_ind[0] and dates[n-1] < dates_ind[0]: ipi1 = n
            if dates[n] >= dates_ind[1] and dates[n-1] < dates_ind[1]: ipi2 = n
        # Input values
        for nn,var in enumerate(varsi):
            vv = np.load(os.path.join(dir, code + '_i_' + var+ '.npy'))
            if nn == 0:
                si=np.shape(vv)
                val = np.zeros((si[0], lookback, len(varsi)))
            val[:,:,nn] = vv[:,0:lookback]
        self.vali_t = val[ipt1:ipt2, :, :]
        self.vali_v = val[ipv1:ipv2, :, :]
        self.vali_i = val[ipi1:ipi2, :, :]
        self.dates_t = dates[ipt1:ipt2]
        self.dates_v = dates[ipv1:ipv2]
        self.dates_i = dates[ipi1:ipi2]

        if varso != None:
            for nn,var in enumerate(varso):
                vv = np.load(os.path.join(dir, code + '_o_' + var+ '.npy'))
                if nn == 0:
                    si=np.shape(vv)
                    val = np.zeros((si[0], len(varso)))
                val[:,nn] = vv[:]
            self.valo_t = val[ipt1:ipt2, :]
            self.valo_v = val[ipv1:ipv2, :]
            self.valo_i = val[ipi1:ipi2, :]



# Test
#var = np.load('/home/oscar/analyse/nn_batch/exps/e_lead1/CBA/dataA_vali_v.npy')
#plt.figure()
#plt.plot(var[0:50,0,0])
#plt.show()
