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




class NN_DataGen():

    def __init__(self, exp, code, name_vali, name_valo, batchsize, shuffle = False):

        home = str(Path.home())
        dir = home + '/analyse/nn_exps/' + exp +'/'+code
        self.vali = np.load(os.path.join(dir, name_vali + '.npy'))
        self.valo = np.load(os.path.join(dir, name_valo + '.npy'))

        si = np.shape(self.vali)
        self.nvals_i = si[0]
        self.lookback = si[1]
        self.nvars_i = si[2]
        si = np.shape(self.valo)
        self.nvars_o = si[1]

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

            samples = np.zeros((len(rows),self.lookback,self.nvars_i))
            targets = np.zeros((len(rows),self.nvars_o))

            for j, row in enumerate(rows):
                samples[j,:,:] = self.vali[row,:,:]
                targets[j] = self.valo[row,:]

            yield samples, targets


class NN_Network():

    def __init__(self, ndef, data_t, data_v):

        self.ndef = ndef
        self.data_t = data_t
        self.data_v = data_v
        self.exp = ndef.exp
        self.lookback = data_t.lookback

        self.define()

    def define(self):

        # First Parallel Network
        print (self.ndef.network1A[0], self.data_t.nvars_i, self.lookback)
        input_tensor = self._input(self.ndef.network1A[0], self.data_t.nvars_i, self.lookback)
        xxx = input_tensor
        for net in self.ndef.network1A:
            xxx = self._layer(net,xxx)

        # Second Parallel Network
        if len(self.ndef.network1B) > 0:

            input_tensorB = self._input(self.ndef.network1B[0], self.data_t.nvars_i, self.lookback)
            xxxB = input_tensorB
            for net in self.ndef.network1B:
                xxxB = self._layer(net, xxxB)

            xxx = layers.concatenate([xxx,xxxB], axis=-1)

            for net in self.ndef.network2:
                xxx = self._layer(net, xxx)

            self.model = Model([input_tensor,input_tensorB], xxx)

        else:
            self.model = Model(input_tensor, xxx)

        self.model.summary()

    def run(self, save=False, ename=None):

        print(self.model.output_shape)

        if self.ndef.optimizer == 'rmsprop':
            if self.ndef.rmsprop>0 :
                self.ndef.optimizer = tf.keras.optimizers.RMSprop(self.ndef.rmsprop)
            else:
                self.ndef.optimizer = tf.keras.optimizers.RMSprop()

        self.model.compile(loss='mean_squared_error',
                  optimizer=self.ndef.optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

        self.train_steps = (self.data_t.nvals_i)//self.data_t.batchsize
        self.val_steps = (self.data_v.nvals_i)//self.data_v.batchsize

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=self.ndef.patience)
        home = str(Path.home())
        if ename != None:
            dir = os.path.join(home+'/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code+'/',ename)
            if not os.path.exists(dir): os.makedirs(dir)
        else:
            dir = os.path.join('home,analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code)

        mc = ModelCheckpoint(dir+ '/'+'best_model.h5',
                         monitor='val_loss', mode='min', verbose=0, save_best_only=True)

#       dir=home+'/analyse/nn_exps/' + self.name
#       tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=dir,histogram_freq=1)
        self.history = self.model.fit_generator(self.data_t.gen, steps_per_epoch=self.train_steps, epochs=self.ndef.epochs,
                                  validation_data = self.data_v.gen,
                                  validation_steps = self.val_steps,
                                  callbacks=[es,mc])
#                                  callbacks=[es,mc,tensorboard_cb])
                                  # callbacks=[PrintDot()])
        if save: self.save(ename=ename)


    def save(self, ename=None):

        # save nn output
        home = str(Path.home())
        dir = home + '/analyse/nn_exps/' + self.ndef.exp + '/' + self.ndef.code
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
        if net[0] == 'LSTM': xxx = layers.LSTM(net[1], return_sequences=net[2]['return_sequences'])(xxx)
        if net[0] == 'GRU': xxx = layers.GRU(net[1], return_sequences=net[2]['return_sequences'])(xxx)
        #           if net[0] == 'TimeDistributed' and net[1] == 'Dense': xxx =
        #                    layers.TimeDistributed(layers.Dense(net[3], activation=net[2]))(xxx)
        return xxx

    def _input(self, net, nvars, lookback):
        if net[0] == 'Dense':
            input_tensor = Input(shape=nvars)
        if net[0] == 'Flattvaen':
            input_tensor = Input(shape=(lookback, nvars))
        if net[0] == 'LSTM':
            input_tensor = Input(shape=(lookback, nvars))
        if net[0] == 'GRU':
            input_tensor = Input(shape=(lookback, nvars))
        return input_tensor

def nn_main(n,ndef, update_data=False):

    if n==0 or update_data == True:
        dataA = NN_DataCreate(ndef.dnameA, 'dataA', ndef, ndef.varsiA, ndef.varsoA, ndef.lookbackA, ndef.codesA)
       #dataB = NN_DataCreate(ndef.dnameB, 'dataA', ndef, ndef.varsiB, ndef.varsoB, ndef.lookbackB, ndef.codesB)

    dataA_gen_t = NN_DataGen(ndef.exp, ndef.code, 'dataA_vali_t', 'dataA_valo_t',
                       ndef.batchsizeT, shuffle = True)
    dataA_gen_v = NN_DataGen(ndef.exp, ndef.code, 'dataA_vali_v', 'dataA_valo_v',
                       ndef.batchsizeV, shuffle = False)

    model = NN_Network(ndef, dataA_gen_t, dataA_gen_v)
    model.run(save=True,ename='en'+str(n))

class NN_Diagnostics():

    def __init__(self, ndef):

        self.data_t = NN_DataGen(ndef.exp, ndef.code, 'dataA_vali_t', 'dataA_valo_t',
                       ndef.batchsizeT, shuffle = False)

        self.data_v = NN_DataGen(ndef.exp, ndef.code, 'dataA_vali_v', 'dataA_valo_v',
                       ndef.batchsizeV, shuffle = False)
        self.ndef = ndef


    def summary(self):


        best_t = np.zeros(self.ndef.n_ensembles)
        best_v = np.zeros(self.ndef.n_ensembles)
        end_t = np.zeros(self.ndef.n_ensembles)
        end_v = np.zeros(self.ndef.n_ensembles)
        no_epochs = np.zeros(self.ndef.n_ensembles)

        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code + '/en'+str(ens)
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


            self.train_steps = (self.data_t.nvals_i)//self.ndef.batchsizeT
            self.val_steps = (self.data_v.nvals_i)//self.ndef.batchsizeV

            self.model.load_weights(os.path.join(dir,  'end_model.h5'))
            predict_t_end = self.model.predict(self.data_t.vali).flatten()
            predict_v_end = self.model.predict(self.data_v.vali).flatten()
            end_t[ens] = pearsonr(predict_t_end, self.data_t.valo.flatten())[0]
            end_v[ens] = pearsonr(predict_v_end, self.data_v.valo.flatten())[0]
            if ens == 0:
                predict_t_end_ave = np.array(predict_t_end)
                predict_v_end_ave = np.array(predict_v_end)
            else:
                predict_t_end_ave += np.array(predict_t_end)
                predict_v_end_ave += np.array(predict_v_end)

            self.model.load_weights(os.path.join(dir,  'best_model.h5'))
            predict_t_best = self.model.predict(self.data_t.vali).flatten()
            predict_v_best = self.model.predict(self.data_v.vali).flatten()
            best_t[ens] = pearsonr(predict_t_best, self.data_t.valo.flatten())[0]
            best_v[ens] = pearsonr(predict_v_best, self.data_v.valo.flatten())[0]
            if ens == 0:
                predict_t_best_ave = np.array(predict_t_best)
                predict_v_best_ave = np.array(predict_v_best)
            else:
                predict_t_best_ave += np.array(predict_t_best)
                predict_v_best_ave += np.array(predict_v_best)

            dir = home + '/analyse/nn_exps/'+self.ndef.exp+'/' + self.ndef.code+'/en'+str(ens)
            np.save(dir+'/predict_t_best.npy',predict_t_best)
            np.save(dir+'/predict_v_best.npy',predict_v_best)

        predict_t_end_ave = predict_t_end_ave/float(self.ndef.n_ensembles)
        predict_v_end_ave = predict_v_end_ave/float(self.ndef.n_ensembles)
        predict_t_best_ave = predict_t_best_ave/float(self.ndef.n_ensembles)
        predict_v_best_ave = predict_v_best_ave/float(self.ndef.n_ensembles)

        cor_t_end = pearsonr(predict_t_end_ave, self.data_t.valo.flatten())[0]
        cor_v_end = pearsonr(predict_v_end_ave, self.data_v.valo.flatten())[0]
        cor_t_best = pearsonr(predict_t_best_ave, self.data_t.valo.flatten())[0]
        cor_v_best = pearsonr(predict_v_best_ave, self.data_v.valo.flatten())[0]
        np.save(home + '/analyse/nn_exps/'+self.ndef.exp+'/' + self.ndef.code+'/cor_t_end.npy',cor_t_end)
        np.save(home + '/analyse/nn_exps/'+self.ndef.exp+'/' + self.ndef.code+'/cor_v_end.npy',cor_v_end)
        np.save(home + '/analyse/nn_exps/'+self.ndef.exp+'/' + self.ndef.code+'/cor_t_best.npy',cor_t_best)
        np.save(home + '/analyse/nn_exps/'+self.ndef.exp+'/' + self.ndef.code+'/cor_v_best.npy',cor_v_best)

        f = open(home + '/analyse/nn_exps/'+self.ndef.exp+'/' + self.ndef.code+'/summary.txt', 'w')
        f.write('BEST \n')
        for ens in range(0,self.ndef.n_ensembles):
            f.write(str(round(best_t[ens],3))+'    '+str(round(best_v[ens],3))+'    '+str(no_epochs[ens])+'\n')
        f.write ('Mean of Ensembles \n')
        f.write (str(round(best_t.mean(),3))+'    '+str(round(best_v.mean(),3))+'\n')
        f.write ('Cor of ensemble mean \n')
        f.write (str(round(cor_t_best,3))+'    '+str(round(cor_v_best,3))+'\n')

        f.write('\n\nEND \n')
        for ens in range(0,self.ndef.n_ensembles):
            f.write(str(round(end_t[ens],3))+'    '+str(round(end_v[ens],3))+'    '+str(no_epochs[ens])+'\n')
        f.write ('Mean of Ensembles \n')
        f.write (str(round(end_t.mean(),3))+'    '+str(round(end_v.mean(),3))+'\n')
        f.write ('Cor of ensemble mean \n')
        f.write (str(round(cor_t_end,3))+'    '+str(round(cor_v_end,3))+'\n')
        f.close()
        
    def plot_history(self):
    
        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code + '/en'+str(ens)
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
            
        fig.savefig(home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code+'/loss.png')     
        fig.clear()

    def plot_scatter(self,nens_cor=None, nens_epoch=None):
    
        cor_t = np.zeros(self.ndef.n_ensembles)
        cor_v = np.zeros(self.ndef.n_ensembles)

        # Plot for each ensemble Membe    
        for ens in range(0,self.ndef.n_ensembles):

            home = str(Path.home())
            dir = home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_best.npy')
            predict_v = np.load(dir+'/predict_v_best.npy')
            cor_t[ens] = pearsonr(predict_t, self.data_t.valo.flatten())[0]
            cor_v[ens] = pearsonr(predict_v, self.data_v.valo.flatten())[0]
            
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))                              
                     
            ax1.plot(self.data_t.valo.flatten(),color='b')
            ax1.plot(predict_t, color='r')       
             
            ax2.scatter(self.data_t.valo.flatten(),predict_t)
            ax2.title.set_text(str(round(cor_t[ens],2))) 
            ax3.plot(self.data_v.valo.flatten(),color='b')
            ax3.plot(predict_v, color='r')
            
            ax4.scatter(self.data_v.valo.flatten(),predict_v)
            ax4.title.set_text(str(round(cor_v[ens],2))) 
           
            fig.savefig(home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code+'/scat_en'+str(ens)+'.png')   
            fig.clear()  
            
            
            if ens == 0:
                predict_t_ave = np.array(predict_t)
                predict_v_ave = np.array(predict_v)
            else:
                predict_t_ave += np.array(predict_t)
                predict_v_ave += np.array(predict_v)

        predict_t_ave = predict_t_ave/float(self.ndef.n_ensembles)
        predict_v_ave = predict_v_ave/float(self.ndef.n_ensembles)

        # Plot for ensemble average        
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
                   
        fig.savefig(home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code+'/scat_en_mean.png')   
        fig.clear()  
        
        # Ensemble Mean of Best Correlations
        if nens_cor==None: nens_cor=len(cor_v)//2
        index = np.argsort(cor_v)[::-1]
        print ('**************')
        print (cor_v)
        print (index)
        
        for i in range(0,nens_cor):
            ens=index[i]
            home = str(Path.home())
            dir = home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_best.npy')
            predict_v = np.load(dir+'/predict_v_best.npy')
            if i==0:
                predict_t_ave = np.array(predict_t)
                predict_v_ave = np.array(predict_v)
            else:            
                predict_t_ave += np.array(predict_t)
                predict_v_ave += np.array(predict_v)
        predict_t_ave = predict_t_ave/float(nens_cor)
        predict_v_ave = predict_v_ave/float(nens_cor)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
                     
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
                   
        fig.savefig(home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code+'/scat_en_cor_'+str(nens_cor)+'.png')     
        
                
        
        
        
        
        # Ensemble Mean of Longest Epochs

        no_epochs = np.zeros(self.ndef.n_ensembles)
        
        home = str(Path.home())
        for ens in range(0,self.ndef.n_ensembles):
            dir = home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code + '/en'+str(ens)
            histfile = os.path.join(dir, "history.npy")
            history =  np.load(histfile, allow_pickle=True)
            hist=ast.literal_eval(str(history))
            no_epochs[ens]=len(hist['loss'])


        if nens_epoch==None: nens_epoch=len(no_epochs)//2
        index = np.argsort(no_epochs)[::-1]        
        
        for i in range(0,nens_epoch):
            ens=index[i]
            home = str(Path.home())
            dir = home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code + '/en'+str(ens)
            predict_t = np.load(dir+'/predict_t_best.npy')
            predict_v = np.load(dir+'/predict_v_best.npy')
            if i==0:
                predict_t_ave = np.array(predict_t)
                predict_v_ave = np.array(predict_v)
            else:            
                predict_t_ave += np.array(predict_t)
                predict_v_ave += np.array(predict_v)
        predict_t_ave = predict_t_ave/float(nens_cor)
        predict_v_ave = predict_v_ave/float(nens_cor)
        
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
                   
        fig.savefig(home + '/analyse/nn_exps/'+self.ndef.exp+ '/' + self.ndef.code+'/scat_en_epochs_'+str(nens_epoch)+'.png')     
        fig.clear()



    

'''
def plot_preds(truth, pred):

    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.plot(truth, label='truth')
    plt.plot(pred, label='pred')
    # plt.ylim([0,5])
    plt.legend()
    plt.show()

'''
