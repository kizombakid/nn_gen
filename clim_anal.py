##!/usr/bin/env python



import matplotlib.pyplot as plt
import tensorflow as tf
import inspect
from scipy.stats.stats import pearsonr
#from nn_data import *
import ast
import matplotlib._color_data as mcd
import seaborn as sns


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path

import sys
sys.path.append('/home/oscar/analyse/nn_trade')

from nn_tools import *
from nn_datagen import NN_DataGen


expdir ='/home/oscar/analyse/nn_batch/exps/e_awap1/penrith_m5'
rref = np.load(expdir+'/dataA_vali_v.npy')
ref = rref[:,0,0]

print (np.shape(ref))

obs = np.load(expdir+'/data_valo_v.npy').flatten()
fc  = np.load(expdir+'/en1/predict_v_best.npy').flatten()

print (len(ref),len(obs),len(fc))

cor_ref = pearsonr(ref,obs)
cor_fc = pearsonr(obs,fc)

print (cor_ref,cor_fc)
plt.figure()
sns.distplot(obs, hist=False, kde=True, label = 'obs')
sns.distplot(ref, hist=False, kde=True, label = 'ref')
sns.distplot(fc, hist=False, kde=True, label = 'fc')
plt.show()



'''

fc


        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

        ax1.plot(self.data_t.valo.flatten(),color='b')
        ax1.plot(predict_t_ave, color='r')

        ax2.scatter(self.data_t.valo.flatten(),predict_t_ave)
        cor = pearsonr(predict_t_ave, self.data_t.valo.flatten())[0]
        ax2.title.set_text(str(round(cor,2)))

        ax3.plot(self.data_v.valo.flatten(),color='b')
        ax3.plot(predict_v_ave, color='r')

        ax4.scatter(self.data_v.valo.flatten(),predict_v_ave)
        cor = pearsonr(predict_v_ave, self.data_v.valo.flatten())[0]
        ax4.title.set_text(str(round(cor,2)))

        fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/scat_en_epochs_'+str(nens_epoch)+'.png')
        fig.clear()'''
