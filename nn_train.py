#!/usr/bin/env python

import matplotlib.pyplot as plt
import tensorflow as tf
import inspect
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats.stats import pearsonr
from nn_data import *
from tensorflow.keras.optimizers import RMSprop

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




def nn_display_stats(model,history,train_pp,train_po):

    # evaluate the model - ************  need check what data should be used for this
#   scores = model.evaluate(train_pp, train_po, verbose=0)
#    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_history(history)

    # *Testing
#    loss, mae, mse = model.evaluate(train_pp, train_po, verbose=0)
#    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    return


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    # plt.ylim([0,5])
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    # plt.ylim([0,20])
    plt.legend()
    plt.show()

class NNData():
    def __init__(self,name):
        dir = '/Users/oalves/python/nn/exps/' + name
        self.vali_t = np.load(os.path.join(dir, 'vali_t.npy'))
        self.valo_t = np.load(os.path.join(dir, 'valo_t.npy'))
        self.vali_v = np.load(os.path.join(dir, 'vali_v.npy'))
        self.valo_v = np.load(os.path.join(dir, 'valo_v.npy'))

        si = np.shape(self.vali_t)
        self.nvars_i_t = si[1]
        self.nvals_i_t = si[0]

        si = np.shape(self.valo_t)
        self.nvars_o_t = si[1]
        self.nvals_o_t = si[1]

        si = np.shape(self.vali_v)
        self.nvars_i_v = si[1]
        self.nvals_i_v = si[0]

        si = np.shape(self.valo_v)
        self.nvars_o_v = si[1]
        self.nvals_o_v = si[1]



def exec_network_serial(name, network, data, batchsize, epochs, rmsprop):

    model = keras.Sequential()

    for n in range(0,len(network)):
        net = network[n]
        print (net)
        if n == 0:
            if net[0] == 'Dense' : model.add(layers.Dense(net[2],activation=net[1],input_shape=(data.nvars_i_t,)))
        elif n==len(network)-1:
            if net[0] == 'Dense': model.add(layers.Dense(net[2]))
        else:
            if net[0] == 'Dense' : model.add(layers.Dense(net[2],activation=net[1]))
            if net[0] == 'Dropout' : model.add(layers.Dropout(net[2]))

    if rmsprop>0 :
        optimizer = tf.keras.optimizers.RMSprop(rmsprop)
    else:
        optimizer = tf.keras.optimizers.RMSprop()

    #optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
                print(str(epoch), end='')
            print('.', end='')

    history = model.fit(
    data.vali_t, data.valo_t, batch_size=batchsize,
    epochs=epochs, #verbose=0,
    validation_data=(data.vali_v, data.valo_v))
    #callbacks=[PrintDot()]


    predict_t = model.predict(data.vali_t).flatten()
    predict_v = model.predict(data.vali_v).flatten()

    print (' ')
    print ('Train     predict vs output  ', pearsonr(predict_t, data.valo_t[:, 0]))
    print ('Validate  predict vs output  ', pearsonr(predict_v, data.valo_v[:, 0]))
    print (' ')

    # save nn output
    dir = '/Users/oalves/python/nn/exps/' + name
    jsonfile = os.path.join(dir, "model.json")
    h5file = os.path.join(dir, "model.h5")
    histfile = os.path.join(dir, "history.npy")
    model_json = model.to_json()
    with open(jsonfile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5file)
    print("Saved model to disk")
    history_dict = history.history
    # Save it under the form of a json file
    np.save(histfile, history_dict)

    model.summary()
    nn_display_stats(model, history, data.vali_t, data.valo_t)

    print ('All Done')



def exp_dj1():
    # use only the DJ change over last day
    name=  inspect.stack()[0][3]
    epochs = 2
    nn_create_data(name, 'djasx', 'X', [1], 20190102, 10000, 20170101, 0)
    data =  NNData(name)
    # print correlations for reference
    print ('Train      value 1 vs output: ',pearsonr(data.vali_t[:,0], data.valo_t[:, 0]))
    print ('Validate   value 1 vs output: ',pearsonr(data.vali_v[:,0], data.valo_v[:, 0]))
    network = []
    network.append(['Dense','relu',2])
    network.append(['Dense','',data.nvars_o_t])
    exec_network_serial(name, network, data , 1, epochs, 0)

def exp_dj2():

    name=  inspect.stack()[0][3]
    epochs = 2
    nn_create_data(name, 'djasx', 'X', [3,3], 20190102, 10000, 20170101, 0)
    data =  NNData(name)
    # print correlations for reference
    print ('Train      value 1 vs output: ',pearsonr(data.vali_t[:,0], data.valo_t[:, 0]))
    print ('Validate   value 1 vs output: ',pearsonr(data.vali_v[:,0], data.valo_v[:, 0]))
    network = []
    network.append(['Dense','relu',12])
    network.append(['Dense','relu',12])
    network.append(['Dense','',data.nvars_o_t])
#    exec_network_serial(name, network, data , data.nvals_i, epochs, 0)
    exec_network_serial(name, network, data , 4, epochs, 0)


def exp_dj3():
    # use absolute DJ over last two days (not the change) similar to dj1
    name =  inspect.stack()[0][3]
    epochs = 10
    nn_create_data(name, 'djasx', 'X', [0, 0 , 2], 20190102, 10000, 20170101, 0)
    data = NNData(name)
    network = []
    network.append(['Dense', 'relu', 16])
    network.append(['Dense', 'relu', 16])
    network.append(['Dense', '', data.nvars_o_t])
    #    exec_network_serial(name, network, data , data.nvals_i, epochs, 0)
    exec_network_serial(name, network, data, 16, epochs, 0)

def exp_dj4():
    # use absolute DJ and ASX over last four days (not the change) similar to dj2
    name = inspect.stack()[0][3]
    epochs = 40
    nn_create_data(name, 'djasx', 'X', [0, 0 , 4, 4], 20190102, 10000, 20170101, 0)
    data = NNData(name)
    network = []
    network.append(['Dense', 'relu', 16])
    network.append(['Dense', 'relu', 16])
    network.append(['Dense', '', data.nvars_o_t])
    #    exec_network_serial(name, network, data , data.nvals_i, epochs, 0)
    exec_network_serial(name, network, data, 4, epochs, 0)


def exp_dj5():
    # use both DJ and ASX change last day + absolute DJ and ASX over last four days (not the change)
    name=  inspect.stack()[0][3]
    epochs = 500
    nn_create_data(name, 'djasx', 'X', [1,1,4,4], 20190102, 10000, 20170101, 0)
    data =  NNData(name)
    # print correlations for reference
    print ('Train      value 1 vs output: ',pearsonr(data.vali_t[:,0], data.valo_t[:, 0]))
    print ('Validate   value 1 vs output: ',pearsonr(data.vali_v[:,0], data.valo_v[:, 0]))
    network = []
    network.append(['Dense','relu',16])
    network.append(['Dense','relu',16])
    network.append(['Dense','',data.nvars_o_t])
#    exec_network_serial(name, network, data , data.nvals_i, epochs, 0)
    exec_network_serial(name, network, data , 64, epochs, 0.00001)



def nn_generator(datai, datao, lookback, shuffle=False, batch_size=128):

    si = np.shape(datai)
    nvals_i = si[0]
    nvars_i = si[1]
    si = np.shape(datao)
    nvars_o = si[1]

    if batch_size > nvals_i: batch_size = nvals_i

    i = lookback

    while 1:
        if shuffle:
            rows = np.random.randint(lookback,nvals_i,size=batch_size)
        else:

            if i+batch_size >= nvals_i:
                i = lookback
            rows = np.arange(i,i+batch_size)
            i += len(rows)

        samples = np.zeros((len(rows),lookback,nvars_i))
        targets = np.zeros((len(rows),nvars_o))

        for j, row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j])
            samples[j] = datai[indices]
            targets[j] = datao[j]

        yield samples, targets

def exp_rn1():


    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                print('')
                print(str(epoch), end='')
            print('.', end='')



    # use both DJ and ASX change last day + absolute DJ and ASX over last four days (not the change)
    name=  inspect.stack()[0][3]
    nn_create_data(name, 'djasx', 'X', [1,1], 20190102, 10000, 20170101, 0)
    data = NNData(name)

    lookback=3
    batch_size=4
    train_gen = nn_generator(data.vali_t, data.valo_t, lookback, shuffle=True, batch_size=batch_size)
    val_gen = nn_generator(data.vali_v, data.valo_v, lookback, shuffle=False, batch_size=batch_size)
    train_gen2 = nn_generator(data.vali_t, data.valo_t, lookback, shuffle=False, batch_size=batch_size)
    val_gen2 = nn_generator(data.vali_v, data.valo_v, lookback, shuffle=False, batch_size=batch_size)

    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(lookback,data.nvars_i_t)))
    model.add(layers.Dense(12, activation = 'relu'))
    model.add(layers.Dense(12, activation = 'relu'))
    model.add(layers.Dense(data.nvars_o_t))

    model.compile(optimizer = RMSprop(),
           loss = 'mean_squared_error',
            metrics = ['mean_absolute_error', 'mean_squared_error'])

    val_steps = (data.nvals_i_v-lookback)//batch_size
    train_steps = (data.nvals_i_t-lookback)//batch_size

    history = model.fit_generator(train_gen, steps_per_epoch=train_steps,  epochs=2,  \
                                  validation_data = val_gen, \
                                  validation_steps = val_steps)
                                  #callbacks=[PrintDot()])

    # Need work out what validation_steps should be -

    predict_t = model.predict_generator(train_gen2,steps=train_steps).flatten()
    predict_v = model.predict_generator(val_gen2,steps=val_steps).flatten()

    print (len(predict_t),len(data.valo_t))
    print (len(predict_v),len(data.valo_v))

    print (' ')
    print ('Train     predict vs output  ', pearsonr(predict_t, data.valo_t[lookback:lookback+len(predict_t), 0]))
    print ('Validate  predict vs output  ', pearsonr(predict_v, data.valo_v[lookback:lookback+len(predict_v), 0]))
    print (' ')

    # save nn output
    dir = '/Users/oalves/python/nn/exps/' + name
    jsonfile = os.path.join(dir, "model.json")
    h5file = os.path.join(dir, "model.h5")
    histfile = os.path.join(dir, "history.npy")
    model_json = model.to_json()
    with open(jsonfile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5file)
    print("Saved model to disk")
    history_dict = history.history
    # Save it under the form of a json file
    np.save(histfile, history_dict)

    model.summary()
    nn_display_stats(model, history, data.vali_t, data.valo_t)

    print('All Done')

 #exp_dj1() # use only the DJ change over last day
#exp_dj2() # use DJ change and ASX change over past n days
#exp_dj3() # use absolute DJ over last two days (not the change) similar to dj1
#exp_dj4() # use absolute DJ and ASX over last four days (not the change) similar to dj2
#exp_dj5()  # use both DJ and ASX change last day + absolute DJ and ASX over last four days (not the change)
# dj5 is sensitive to rmsprop value

# These two should produce the same results - one is direct form, the second one uses a generator
#exp_dj2()
exp_rn1()