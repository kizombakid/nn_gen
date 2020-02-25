#!/usr/bin/env python
#from __future__ import absolute_import, division, print_function

import numpy as np
from nn_tools import *
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path

class NN_Awap():

    def __init__(self,set,var):

        self.dname = 'awap1'

        self.date2 = 20191231
        self.date = 19110101
        self.lead_max = 0
        self.lookback = 365
        self.set = set

        home = str(Path.home())
        dir = home +'/analyse/data/awap_series'
        self.aph = np.load(dir+'/'+set+'_'+var+'_aph.npy')
        self.apl_c = np.load(dir+'/'+set+'_'+var+'_apl_c.npy')
        self.apl_n = np.load(dir+'/'+set+'_'+var+'_apl_n.npy')
        self.apl_s = np.load(dir+'/'+set+'_'+var+'_apl_s.npy')
        self.apl_e = np.load(dir+'/'+set+'_'+var+'_apl_e.npy')
        self.apl_w = np.load(dir+'/'+set+'_'+var+'_apl_w.npy')

        self.vph = np.load(dir+'/'+set+'_'+var+'_vph.npy')
        self.vpl_c = np.load(dir+'/'+set+'_'+var+'_vpl_c.npy')
        self.vpl_n = np.load(dir+'/'+set+'_'+var+'_vpl_n.npy')
        self.vpl_s = np.load(dir+'/'+set+'_'+var+'_vpl_s.npy')
        self.vpl_e = np.load(dir+'/'+set+'_'+var+'_vpl_e.npy')
        self.vpl_w = np.load(dir+'/'+set+'_'+var+'_vpl_w.npy')

        self.dates = np.load(dir+'/'+set+'_'+var+'_date.npy')

        print ('Cor full ', var , pearsonr(self.vph, self.vpl_c))
        print ('Cor anom ', var , pearsonr(self.aph, self.apl_c))

#        plt.figure()
#        plt.plot(self.vph[0:365])
#        plt.plot(self.vpl_c[0:365])
#        plt.show()
#        plt.figure()
#        plt.plot(self.aph[0:365])
#        plt.plot(self.apl_c[0:365])
#        plt.show()


    def _input(self, fname, formula, fmean, fstd, fdates=False):
        vali = []
        dates = []

        print (fname)

        buff1 = self.lookback
        buff2 = self.lead_max

        for n in range(buff1,len(self.vph)-buff2):

 #          if ema1[n-3] > ema2[n-3] and ema1[n-4] < ema2[n-4]:
            if True:

                dates.append(int(self.dates[n]))
                vi=[]
                for k in range(0,self.lookback):
                    exec('vi.append(' + formula + ')')
                vali.append(vi)

        rstd = np.float(1.0)
        rmean = np.float(0.0)
        if fmean: rmean = np.mean(vali)
        vali = np.array(vali) - rmean
        if fstd: rstd = np.std(vali)
        vali = vali/rstd

        home = str(Path.home())
        dir = home + '/analyse/nn_data/' + self.dname
        if not os.path.exists(dir): os.makedirs(dir)
        np.save(os.path.join(dir, self.set + '_i_' + fname + '.npy'), np.float32(vali))
        np.save(os.path.join(dir, self.set + '_i_' + fname + '_meandstd.npy'), np.array([rmean,rstd]))

        if fdates:
            np.save(os.path.join(dir, self.set + '_dates.npy'), np.int32(dates))

    def _output(self, fname, formula, fmean, fstd):

        print (fname)

        buff1 = self.lookback
        buff2 = self.lead_max

        vali = []
        for n in range(buff1, len(self.vph) - buff2):

            #          if ema1[n-3] > ema2[n-3] and ema1[n-4] < ema2[n-4]:
            if True:
                exec('vali.append(' + formula + ')')

        rstd = np.float(1.0)
        rmean = np.float(0.0)
        if fmean: rmean = np.mean(vali)
        vali = np.array(vali) -rmean
        if fstd: rstd = np.std(vali)
        vali = vali / rstd

        home = str(Path.home())
        dir = home + '/analyse/nn_data/' + self.dname
        if not os.path.exists(dir): os.makedirs(dir)
        np.save(os.path.join(dir, self.set + '_o_' + fname + '.npy'), np.float32(vali))
        np.save(os.path.join(dir, self.set + '_o_' + fname + '_meandstd.npy'), np.array([rmean,rstd]))

    def nn_create_awap1(self, dname, lookback):

        self.dname = dname
        self.lookback = lookback

        home = str(Path.home())
        dir= home + '/analyse/nn_data/'+dname
        if not os.path.exists(dir): os.makedirs(dir)

        self._input('vpl_c','self.vpl_c[n]', True, True, fdates=True)
        self._input('vpl_n','self.vpl_n[n]', True, True, fdates=False)
        self._input('vpl_s','self.vpl_s[n]', True, True, fdates=False)
        self._input('vpl_e','self.vpl_e[n]', True, True, fdates=False)
        self._input('vpl_w','self.vpl_w[n]', True, True, fdates=False)

        self._input('apl_c','self.apl_c[n]', True, True, fdates=False)
        self._input('apl_n','self.apl_n[n]', True, True, fdates=False)
        self._input('apl_s','self.apl_s[n]', True, True, fdates=False)
        self._input('apl_e','self.apl_e[n]', True, True, fdates=False)
        self._input('apl_w','self.apl_w[n]', True, True, fdates=False)

        self._output('full','self.vph[n]', True, True)
        self._output('anom','self.aph[n]', False, True)



sets= ['penrith_m5']
var='tmax'

for set in sets:
    print (set)
    xx = NN_Awap(set,var)
    xx.nn_create_awap1('awap1',365)
print ('Done')
'''



# Test
#var = np.load('/Users/oalves/analyse/nn_data/data1/CBA_i_rsi1.npy')
#plt.figure()
#plt.plot(var[:,5])
#plt.show()
'''
