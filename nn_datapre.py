#!/usr/bin/env python
#from __future__ import absolute_import, division, print_function

import numpy as np
from nn_tools import *
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from pathlib import Path

class NN_data1():

    def __init__(self,code, lookback, dname):

        self.dname = 'data1'

        self.date2 = 20200101
        self.date = 19900101
        self.lead_max = 30
        self.lookback = lookback
        self.code = code

        self.price = SPriceEod(self.code, self.date, date2=self.date2)
        print (self.price.date[0],self.price.date[-1])

        self.ema1 = fts_ema(self.price.close,12,fillna=True)
        self.ema2 = fts_ema(self.price.close,30,fillna=True)
        self.ema3 = fts_ema(self.price.close, 75, fillna=True)
        self.emad1= (self.ema1-self.ema2)/self.ema2
        self.emad2= (self.ema2-self.ema3)/self.ema3
        self.rsi1 = fts_rsi(self.price.close,15, fillna=True)
        self.rsi2 = fts_rsi(self.price.close,25, fillna=True)
        self.rsi3 = fts_rsi(self.price.close,50, fillna=True)

        self._nn_create_data(dname)

    def _input(self, fname, formula, fmean, fstd, fdates=False):
        vali = []
        dates = []

        print (fname)

        buff1 = self.lookback
        buff2 = self.lead_max

        for n in range(buff1,len(self.price.close)-buff2):

 #          if ema1[n-3] > ema2[n-3] and ema1[n-4] < ema2[n-4]:
            if True:

                dates.append(int(self.price.date[n]))
                vi=[]
                for k in range(0,lookback):
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
        np.save(os.path.join(dir, code + '_i_' + fname + '.npy'), np.float32(vali))
        np.save(os.path.join(dir, code + '_i_' + fname + '_meandstd.npy'), np.array([rmean,rstd]))

        if fdates:
            np.save(os.path.join(dir, code + '_dates.npy'), np.int32(dates))

    def _output(self, fname, formula, fmean, fstd):

        print (fname)

        buff1 = self.lookback
        buff2 = self.lead_max

        vali = []
        for n in range(buff1, len(self.price.close) - buff2):

            #          if ema1[n-3] > ema2[n-3] and ema1[n-4] < ema2[n-4]:
            if True:
                exec('vali.append(' + formula + ')')

        rstd = np.float(1.0)
        rmean = np.float(0.0)
        if fmean: rmean = np.mean(vali)
        vali = np.array(vali) - rmean
        if fstd: rstd = np.std(vali)
        vali = vali / rstd

        home = str(Path.home())
        dir = home + '/analyse/nn_data/' + self.dname
        if not os.path.exists(dir): os.makedirs(dir)
        np.save(os.path.join(dir, code + '_o_' + fname + '.npy'), np.float32(vali))
        np.save(os.path.join(dir, code + '_o_' + fname + '_meandstd.npy'), np.array([rmean,rstd]))

    def _nn_create_data(self, dname):

        home = str(Path.home())
        dir= home + '/analyse/nn_data/'+dname
        if not os.path.exists(dir): os.makedirs(dir)

        self._input('open','self.price.open[n-k]', True, True, fdates=True)

        self._input('high', 'self.price.high[n - k - 1]', True, True)
        self._input('low','self.price.low[n - k - 1]', True, True)
        self._input('close','self.price.open[n-k-1]', True, True)
        self._input('volume','self.price.volume[n-k-1]', True, True)
        self._input('ema1','self.ema1[n-k-1]', True, True)
        self._input('ema2','self.ema2[n-k-1]', True, True)
        self._input('ema3','self.ema3[n-k-1]', True, True)
        self._input('rsi1','self.rsi1[n - k-1]', True, True)
        self._input('rsi2','self.rsi2[n - k-1]', True, True)
        self._input('rsi3','self.rsi3[n - k-1]', True, True)
        self._input('emad1','self.emad1[n - k - 1]', False, True)
        self._input('emad2','self.emad2[n - k - 1]', False, True)
        self._input('emag1','(self.ema1[n-k-1]-self.ema1[n-k-2])/self.ema1[n-k-2]', False, True)
        self._input('emag2','(self.ema2[n-k-1]-self.ema2[n-k-2])/self.ema2[n-k-2]', False, True)
        self._input('emag3','(self.ema3[n-k-1]-self.ema3[n-k-2])/self.ema3[n-k-2]', False, True)

        self._output('lead0','(self.price.close[n]-self.price.open[n])/self.price.open[n]', False, True)
        self._output('lead1','(self.price.close[n+1]-self.price.open[n])/self.price.open[n]', False, True)
        self._output('lead5','(np.mean(self.price.close[n+3:n+5])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('lead10','(np.mean(self.price.close[n+5:n+10])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('lead15','(np.mean(self.price.close[n+7:n+15])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('lead20','(np.mean(self.price.close[n+10:n+20])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('lead30','(np.mean(self.price.close[n+15:n+30])-self.price.open[n])/self.price.open[n]', False, True)

        self._output('mean5','(np.mean(self.price.close[n:n+5])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('mean10','(np.mean(self.price.close[n:n+10])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('mean15','(np.mean(self.price.close[n:n+15])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('mean20','(np.mean(self.price.close[n:n+20])-self.price.open[n])/self.price.open[n]', False, True)
        self._output('mean30','(np.mean(self.price.close[n:n+30])-self.price.open[n])/self.price.open[n]', False, True)

codes = ['RIO','BHP','IAG','FMG','SUN','WOW','WES','WPL','IPL','BXB','AMC','NCM','TLS']

dname = 'data1'
lookback = 500
for code in codes:
    print (code)
    xx = NN_data1(code, lookback, dname)
print ('Done')

# Test
#var = np.load('/Users/oalves/analyse/nn_data/data1/CBA_i_rsi1.npy')
#plt.figure()
#plt.plot(var[:,5])
#plt.show()
