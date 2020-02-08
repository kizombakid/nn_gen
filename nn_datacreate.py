#!/usr/bin/env python
#from __future__ import absolute_import, division, print_function

import numpy as np
from nn_tools import *
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path

class NN_DataCreate():

    def __init__(self, dname, dname_out, ndef, varsi, varso, lookback, codes):

        print ('****', codes)

        for nn, code in enumerate(codes):
            print ('****', code)
            vali = self.readcode(dname, code, varsi, varso, ndef.dates_train, ndef.dates_val, lookback)
            if nn == 0:
                vali_t = self.vali_t
                vali_v = self.vali_v
                valo_t = self.valo_t
                valo_v = self.valo_v
            else:
                vali_t = np.append(vali_t, self.vali_t, axis=0)
                vali_v = np.append(vali_v, self.vali_v, axis=0)
                valo_t = np.append(valo_t, self.valo_t, axis=0)
                valo_v = np.append(valo_v, self.valo_v, axis=0)

        home = str(Path.home())
        dir = home + '/analyse/nn_exps/' + ndef.exp+'/'+ndef.code
        if not os.path.exists(dir): os.makedirs(dir)
        np.save(os.path.join(dir, dname_out + '_vali_t.npy'), np.float32(vali_t))
        np.save(os.path.join(dir, dname_out + '_vali_v.npy'), np.float32(vali_v))
        np.save(os.path.join(dir, dname_out + '_valo_t.npy'), np.float32(valo_t))
        np.save(os.path.join(dir, dname_out + '_valo_v.npy'), np.float32(valo_v))
        print ('Data ', dname_out, 'Created')

    def readcode (self, dname, code, varsi, varso, dates_train, dates_val, lookback):

        home = str(Path.home())
        dir = home + '/analyse/nn_data/' + dname

        dates = np.load(os.path.join(dir, code + '_dates.npy'))
        if dates[0] > dates_train[0]: ipt1 = 0
        if dates[-1] < dates_val[1]: ipv2 = len(dates)-1
        for n in range(1,len(dates)):
            if dates[n] >= dates_train[0] and dates[n-1] < dates_train[0]: ipt1 = n
            if dates[n] >= dates_train[1] and dates[n-1] < dates_train[1]: ipt2 = n
            if dates[n] >= dates_val[0] and dates[n-1] < dates_val[0]: ipv1 = n
            if dates[n] >= dates_val[1] and dates[n-1] < dates_val[1]: ipv2 = n
        # Input values
        for nn,var in enumerate(varsi):
            vv = np.load(os.path.join(dir, code + '_i_' + var+ '.npy'))
            if nn == 0:
                si=np.shape(vv)
                val = np.zeros((si[0], lookback, len(varsi)))
            val[:,:,nn] = vv[:,0:lookback]
        self.vali_t = val[ipt1:ipt2, :, :]
        self.vali_v = val[ipv1:ipv2, :, :]

        # Output Values
        for nn,var in enumerate(varso):
            vv = np.load(os.path.join(dir, code + '_o_' + var+ '.npy'))
            if nn == 0:
                si=np.shape(vv)
                val = np.zeros((si[0], len(varso)))
            val[:,nn] = vv[:]
        self.valo_t = val[ipt1:ipt2, :]
        self.valo_v = val[ipv1:ipv2, :]


# Test
#var = np.load('/Users/oalves/analyse/nn_data/data1/CBA_i_rsi1.npy')
#plt.figure()
#plt.plot(var[:,0])
#plt.show()
