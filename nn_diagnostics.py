##!/usr/bin/env python



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

import sys
sys.path.append('/home/oscar/analyse/nn_trade')

from nn_tools import *
from nn_datagen import NN_DataGen


class NN_Diagnostics():

    def __init__(self, ndef):

        self.data_t = NN_DataGen(ndef, 't', shuffle = False, batchsize=ndef.batchsize)
        self.data_v = NN_DataGen(ndef, 'v', shuffle = False, batchsize=ndef.batchsize)
        self.data_i = NN_DataGen(ndef, 'i', shuffle = False, batchsize=ndef.batchsize)
        self.ndef = ndef


    def summary(self):


        best_t = np.zeros(self.ndef.n_ensembles)
        best_v = np.zeros(self.ndef.n_ensembles)
        best_i = np.zeros(self.ndef.n_ensembles)
        end_t = np.zeros(self.ndef.n_ensembles)
        end_v = np.zeros(self.ndef.n_ensembles)
        end_i = np.zeros(self.ndef.n_ensembles)
        no_epochs = np.zeros(self.ndef.n_ensembles)

        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            jsonfile = os.path.join(dir,"model.json")
            histfile = os.path.join(dir, "history.npy")

            # load json and create model
            json_file = open(jsonfile, 'r')
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
                    # load weights into new model
            self.history =  np.load(histfile, allow_pickle=True)
            hist=ast.literal_eval(str(self.history))
            no_epochs[ens]=len(hist['loss'])

            self.train_steps = (self.data_t.nvals_i)//self.ndef.batchsize
            self.val_steps = (self.data_v.nvals_i)//self.ndef.batchsize
            self.ind_steps = (self.data_i.nvals_i)//self.ndef.batchsize

            self.model.load_weights(os.path.join(dir,  'end_model.h5'))

            if self.ndef.n_nets == 1:
                samples_t = self.data_t.A_vali
                samples_v = self.data_v.A_vali
                samples_i = self.data_i.A_vali
            if self.ndef.n_nets == 2:
                samples_t = [self.data_t.A_vali, self.data_t.B_vali]
                samples_v = [self.data_v.A_vali, self.data_v.B_vali]
                samples_i = [self.data_i.A_vali, self.data_i.B_vali]
            if self.ndef.n_nets == 3:
                samples_t = [self.data_t.A_vali, self.data_t.B_vali, self.data_t.C_vali]
                samples_v = [self.data_v.A_vali, self.data_v.B_vali, self.data_v.C_vali]
                samples_i = [self.data_i.A_vali, self.data_i.B_vali, self.data_i.C_vali]

            predict_t_end = self.model.predict(samples_t).flatten()
            predict_v_end = self.model.predict(samples_v).flatten()
            predict_i_end = self.model.predict(samples_i).flatten()
            end_t[ens] = pearsonr(predict_t_end, self.data_t.valo.flatten())[0]
            end_v[ens] = pearsonr(predict_v_end, self.data_v.valo.flatten())[0]
            end_i[ens] = pearsonr(predict_i_end, self.data_i.valo.flatten())[0]
            if ens == 0:
                predict_t_end_ave = np.array(predict_t_end)
                predict_v_end_ave = np.array(predict_v_end)
                predict_i_end_ave = np.array(predict_i_end)
            else:
                predict_t_end_ave += np.array(predict_t_end)
                predict_v_end_ave += np.array(predict_v_end)
                predict_i_end_ave += np.array(predict_i_end)

            self.model.load_weights(os.path.join(dir,  'best_model.h5'))
            predict_t_best = self.model.predict(samples_t).flatten()
            predict_v_best = self.model.predict(samples_v).flatten()
            predict_i_best = self.model.predict(samples_i).flatten()
            best_t[ens] = pearsonr(predict_t_best, self.data_t.valo.flatten())[0]
            best_v[ens] = pearsonr(predict_v_best, self.data_v.valo.flatten())[0]
            best_i[ens] = pearsonr(predict_i_best, self.data_i.valo.flatten())[0]
            if ens == 0:
                predict_t_best_ave = np.array(predict_t_best)
                predict_v_best_ave = np.array(predict_v_best)
                predict_i_best_ave = np.array(predict_i_best)
            else:
                predict_t_best_ave += np.array(predict_t_best)
                predict_v_best_ave += np.array(predict_v_best)
                predict_i_best_ave += np.array(predict_i_best)

            dir = self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/en'+str(ens)
            np.save(dir+'/predict_t_best.npy',predict_t_best)
            np.save(dir+'/predict_v_best.npy',predict_v_best)
            np.save(dir+'/predict_i_best.npy',predict_i_best)
            np.save(dir+'/predict_t_end.npy',predict_t_end)
            np.save(dir+'/predict_v_end.npy',predict_v_end)
            np.save(dir+'/predict_i_end.npy',predict_i_end)

        predict_t_end_ave = predict_t_end_ave/float(self.ndef.n_ensembles)
        predict_v_end_ave = predict_v_end_ave/float(self.ndef.n_ensembles)
        predict_i_end_ave = predict_i_end_ave/float(self.ndef.n_ensembles)
        predict_t_best_ave = predict_t_best_ave/float(self.ndef.n_ensembles)
        predict_v_best_ave = predict_v_best_ave/float(self.ndef.n_ensembles)
        predict_i_best_ave = predict_i_best_ave/float(self.ndef.n_ensembles)

        cor_t_end = pearsonr(predict_t_end_ave, self.data_t.valo.flatten())[0]
        cor_v_end = pearsonr(predict_v_end_ave, self.data_v.valo.flatten())[0]
        cor_i_end = pearsonr(predict_i_end_ave, self.data_i.valo.flatten())[0]
        cor_t_best = pearsonr(predict_t_best_ave, self.data_t.valo.flatten())[0]
        cor_v_best = pearsonr(predict_v_best_ave, self.data_v.valo.flatten())[0]
        cor_i_best = pearsonr(predict_i_best_ave, self.data_i.valo.flatten())[0]
        np.save(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/cor_t_end.npy',cor_t_end)
        np.save(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/cor_v_end.npy',cor_v_end)
        np.save(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/cor_i_end.npy',cor_v_end)
        np.save(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/cor_t_best.npy',cor_t_best)
        np.save(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/cor_v_best.npy',cor_v_best)
        np.save(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/cor_i_best.npy',cor_v_best)

        f = open(self.ndef.basedir+'/'+self.ndef.exp+'/' + self.ndef.set+'/summary.txt', 'w')
        f.write('BEST \n')
        for ens in range(0,self.ndef.n_ensembles):
            f.write(str(round(best_t[ens],3))+'    '+str(round(best_v[ens],3))+'    '+str(round(best_i[ens],3))+'    '+str(no_epochs[ens])+'\n')
        f.write ('Mean of Ensembles \n')
        f.write (str(round(best_t.mean(),3))+'    '+str(round(best_v.mean(),3))+'    '+str(round(best_i.mean(),3))+'\n')
        f.write ('Cor of ensemble mean \n')
        f.write (str(round(cor_t_best,3))+'    '+str(round(cor_v_best,3))+'    '+str(round(cor_i_best,3))+'\n')

        f.write('\n\nEND \n')
        for ens in range(0,self.ndef.n_ensembles):
            f.write(str(round(end_t[ens],3))+'    '+str(round(end_v[ens],3))+'    '+str(round(end_i[ens],3))+'    '+str(no_epochs[ens])+'\n')
        f.write ('Mean of Ensembles \n')
        f.write (str(round(end_t.mean(),3))+'    '+str(round(end_v.mean(),3))+'    '+str(round(end_i.mean(),3))+'\n')
        f.write ('Cor of ensemble mean \n')
        f.write (str(round(cor_t_end,3))+'    '+str(round(cor_v_end,3))+'    '+str(round(cor_i_end,3))+'\n')
        f.close()

    def plot_history(self):

        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            histfile = os.path.join(dir, "history.npy")
            history =  np.load(histfile, allow_pickle=True)

            hist=ast.literal_eval(str(history))

            clrs = sns.color_palette('husl', n_colors=self.ndef.n_ensembles)  # a list of RGB tuples


            if ens ==0:
                fig,axs = plt.subplots(2,1)
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('Mean Abs Error [MPG]')
                axs[1].set_xlabel('Epoch')
                axs[1].set_ylabel('Mean Square Error [$MPG^2$]')

            lines = axs[0].plot(range(1,len(hist['mean_absolute_error'])+1),hist['mean_absolute_error'], label='Train Error')
            lines[0].set_color(clrs[ens])
            lines = axs[0].plot(range(1,len(hist['val_mean_absolute_error'])+1),hist['val_mean_absolute_error'], label='Val Error')
            lines[0].set_color(clrs[ens])
            axs[0].legend()

            lines = axs[1].plot(range(1,len(hist['mean_squared_error'])+1),hist['mean_squared_error'], label='Train Error')
            lines[0].set_color(clrs[ens])
            lines = axs[1].plot(range(1,len(hist['val_mean_squared_error'])+1),hist['val_mean_squared_error'], label='Val Error')
            lines[0].set_color(clrs[ens])

            axs[1].legend()

        fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/loss.png')
        fig.clear()
        plt.close('all')


    def plot_scatter(self,type='best',nens_cor=None, nens_epoch=None):

        cor_t = np.zeros(self.ndef.n_ensembles)
        cor_v = np.zeros(self.ndef.n_ensembles)
        cor_i = np.zeros(self.ndef.n_ensembles)

        # Plot for each ensemble Membe
        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_'+type+'.npy')
            predict_v = np.load(dir+'/predict_v_'+type+'.npy')
            predict_i = np.load(dir+'/predict_i_'+type+'.npy')
            cor_t[ens] = pearsonr(predict_t, self.data_t.valo.flatten())[0]
            cor_v[ens] = pearsonr(predict_v, self.data_v.valo.flatten())[0]
            cor_i[ens] = pearsonr(predict_i, self.data_i.valo.flatten())[0]


            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6) ) = plt.subplots(3, 2, figsize=(12, 12))

            ax1.plot(self.data_t.valo.flatten(),color='b')
            ax1.plot(predict_t, color='r')

            ax2.scatter(self.data_t.valo.flatten(),predict_t)
            ax2.title.set_text('Train '+str(round(cor_t[ens],2)))

            ax3.plot(self.data_v.valo.flatten(),color='b')
            ax3.plot(predict_v, color='r')

            ax4.scatter(self.data_v.valo.flatten(),predict_v)
            ax4.title.set_text('Val '+str(round(cor_v[ens],2)))

            ax5.plot(self.data_i.valo.flatten(),color='b')
            ax5.plot(predict_i, color='r')

            ax6.scatter(self.data_i.valo.flatten(),predict_i)
            ax6.title.set_text('Ind '+str(round(cor_i[ens],2)))

            fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/scat_en'+str(ens)+'_'+type+'.png')
            fig.clear()


            if ens == 0:
                predict_t_ave = np.array(predict_t)
                predict_v_ave = np.array(predict_v)
                predict_i_ave = np.array(predict_i)
            else:
                predict_t_ave += np.array(predict_t)
                predict_v_ave += np.array(predict_v)
                predict_i_ave += np.array(predict_i)

        predict_t_ave = predict_t_ave/float(self.ndef.n_ensembles)
        predict_v_ave = predict_v_ave/float(self.ndef.n_ensembles)
        predict_i_ave = predict_i_ave/float(self.ndef.n_ensembles)

        # Plot for ensemble average
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

        ax1.plot(self.data_t.valo.flatten(),color='b')
        ax1.plot(predict_t_ave, color='r')

        ax2.scatter(self.data_t.valo.flatten(),predict_t_ave)
        cor = pearsonr(predict_t_ave, self.data_t.valo.flatten())[0]
        ax2.title.set_text('Train '+str(round(cor,2)))

        ax3.plot(self.data_v.valo.flatten(),color='b')
        ax3.plot(predict_v_ave, color='r')

        ax4.scatter(self.data_v.valo.flatten(),predict_v_ave)
        cor = pearsonr(predict_v_ave, self.data_v.valo.flatten())[0]
        ax4.title.set_text('Val '+str(round(cor,2)))

        ax5.plot(self.data_i.valo.flatten(),color='b')
        ax5.plot(predict_i_ave, color='r')

        ax6.scatter(self.data_i.valo.flatten(),predict_i_ave)
        cor = pearsonr(predict_i_ave, self.data_i.valo.flatten())[0]
        ax6.title.set_text('Ind '+str(round(cor,2)))

        fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/scat_en_mean_'+type+'.png')
        fig.clear()

        # Ensemble Mean of Best Correlations
        if nens_cor==None: nens_cor=max(len(cor_v)//2,1)
        index = np.argsort(cor_v)[::-1]

        for i in range(0,nens_cor):
            ens=index[i]
            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_'+type+'.npy')
            predict_v = np.load(dir+'/predict_v_'+type+'.npy')
            predict_i = np.load(dir+'/predict_i_'+type+'.npy')
            if i==0:
                predict_t_ave = np.array(predict_t)
                predict_v_ave = np.array(predict_v)
                predict_i_ave = np.array(predict_i)
            else:
                predict_t_ave += np.array(predict_t)
                predict_v_ave += np.array(predict_v)
                predict_i_ave += np.array(predict_i)

        predict_t_ave = predict_t_ave/float(nens_cor)
        predict_v_ave = predict_v_ave/float(nens_cor)
        predict_i_ave = predict_i_ave/float(nens_cor)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

        ax1.plot(self.data_t.valo.flatten(),color='b')
        ax1.plot(predict_t_ave, color='r')

        ax2.scatter(self.data_t.valo.flatten(),predict_t_ave)
        cor = pearsonr(predict_t_ave, self.data_t.valo.flatten())[0]
        ax2.title.set_text('Train '+str(round(cor,2)))

        ax3.plot(self.data_v.valo.flatten(),color='b')
        ax3.plot(predict_v_ave, color='r')

        ax4.scatter(self.data_v.valo.flatten(),predict_v_ave)
        cor = pearsonr(predict_v_ave, self.data_v.valo.flatten())[0]
        ax4.title.set_text('Val '+str(round(cor,2)))

        ax5.plot(self.data_i.valo.flatten(),color='b')
        ax5.plot(predict_i_ave, color='r')

        ax6.scatter(self.data_i.valo.flatten(),predict_i_ave)
        cor = pearsonr(predict_i_ave, self.data_i.valo.flatten())[0]
        ax6.title.set_text('Ind '+str(round(cor,2)))

        fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/scat_en_cor_'+str(nens_cor)+'_'+type+'.png')
        fig.clear()
        plt.close('all')






        # Ensemble Mean of Longest Epochs

        no_epochs = np.zeros(self.ndef.n_ensembles)

        home = str(Path.home())
        for ens in range(0,self.ndef.n_ensembles):
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            histfile = os.path.join(dir, "history.npy")
            history =  np.load(histfile, allow_pickle=True)
            hist=ast.literal_eval(str(history))
            no_epochs[ens]=len(hist['loss'])


        if nens_epoch==None: nens_epoch=len(no_epochs)//2
        index = np.argsort(no_epochs)[::-1]

        for i in range(0,nens_epoch):
            ens=index[i]
            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_'+type+'.npy')
            predict_v = np.load(dir+'/predict_v_'+type+'.npy')
            predict_i = np.load(dir+'/predict_i_'+type+'.npy')
            if i==0:
                predict_t_ave = np.array(predict_t)
                predict_v_ave = np.array(predict_v)
                predict_i_ave = np.array(predict_i)
            else:
                predict_t_ave += np.array(predict_t)
                predict_v_ave += np.array(predict_v)
                predict_i_ave += np.array(predict_i)

        predict_t_ave = predict_t_ave/float(nens_cor)
        predict_v_ave = predict_v_ave/float(nens_cor)
        predict_i_ave = predict_i_ave/float(nens_cor)

        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

        ax1.plot(self.data_t.valo.flatten(),color='b')
        ax1.plot(predict_t_ave, color='r')

        ax2.scatter(self.data_t.valo.flatten(),predict_t_ave)

        cor = pearsonr(predict_t_ave, self.data_t.valo.flatten())[0]
        ax2.title.set_text('Train '+str(round(cor,2)))

        ax3.plot(self.data_v.valo.flatten(),color='b')
        ax3.plot(predict_v_ave, color='r')

        ax4.scatter(self.data_v.valo.flatten(),predict_v_ave)
        cor = pearsonr(predict_v_ave, self.data_v.valo.flatten())[0]
        ax4.title.set_text('Val '+str(round(cor,2)))

        ax5.plot(self.data_i.valo.flatten(),color='b')
        ax5.plot(predict_i_ave, color='r')

        ax6.scatter(self.data_i.valo.flatten(),predict_i_ave)
        cor = pearsonr(predict_i_ave, self.data_i.valo.flatten())[0]
        ax6.title.set_text('Ind '+str(round(cor,2)))


        fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/scat_en_epochs_'+str(nens_epoch)+'_'+type+'.png')
        fig.clear()
        plt.close('all')



    def plot_scatter_all(self,type='best'):

        fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(18,18))

        ymin = 9999
        ymax = -99999
        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_'+type+'.npy')
            ax1.scatter(self.data_t.valo.flatten(),predict_t)
            ymin = min(ymin,min(predict_t))
            ymax = max(ymax,max(predict_t))
        ax1.plot([0.0,0.0],[ymin,ymax])

        ymin = 9999
        ymax = -99999
        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            predict_v = np.load(dir+'/predict_v_'+type+'.npy')
            ax2.scatter(self.data_v.valo.flatten(),predict_v)
            ymin = min(ymin,min(predict_v))
            ymax = max(ymax,max(predict_v))
        ax2.plot([0.0,0.0],[ymin,ymax])

        ymin = 9999
        ymax = -99999
        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set + '/en'+str(ens)
            predict_i = np.load(dir+'/predict_i_'+type+'.npy')
            ax3.scatter(self.data_i.valo.flatten(),predict_i)
            ymin = min(ymin,min(predict_i))
            ymax = max(ymax,max(predict_i))
        ax3.plot([0.0,0.0],[ymin,ymax])

        fig.savefig(self.ndef.basedir+'/'+self.ndef.exp+ '/' + self.ndef.set+'/scat_all_'+type+'.png')
        fig.clear()
        plt.close('all')
