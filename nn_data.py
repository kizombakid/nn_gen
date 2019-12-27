#!/usr/bin/env python
#from __future__ import absolute_import, division, print_function

import numpy as np
from nn_tools import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def nn_data_djasx(code,date,daysback,params):

    # simple asx/dj prediction with nback and emas
    nback=params+[0,0,0,0,0,0,0,0,0,0] # add more zeros to fill out

    vali = []
    valo = []
    vdate = []

    dj = SPriceEod('dj', date, daysback=daysback+10)
    asx = SPriceEod('asx200', date, daysback=daysback)
    djm1 = fts_ema(dj.close,12,fillna=True)
    djm2 = fts_ema(dj.close,30,fillna=True)
    asxm1 = fts_ema(asx.close,12,fillna=True)
    asxm2 = fts_ema(asx.close,30,fillna=True)

    buff1 = max(nback)+1
    buff2 = 2

    for n in range(buff1,len(asx.close)-buff2):
        vi=[]
        vo=[]

        vdate.append(asx.date[n])

        # work out the index for the DJ prior to asx date
        for i in range(0,len(dj.close)):
            if asx.date[n] > dj.date[i] and asx.date[n] <= dj.date[i+1]:
                index = i

        for k in range(0,nback[0]): vi.append(dj.close[index-k] - dj.close[index - 1-k])
        for k in range(0,nback[1]): vi.append(asx.close[n - 1 - k] - asx.close[n - 2 - k])
        for k in range(0,nback[2]): vi.append(dj.close[index - k])
        for k in range(0,nback[3]): vi.append(asx.close[n-1-k])
        for k in range(0,nback[4]): vi.append(djm1[index-k]-djm2[index-k])      # difference in the two moving averages
        for k in range(0,nback[5]): vi.append(asxm1[n-1 - k] - asxm2[n-1 - k])  # difference in the two moving averages

        vali.append(vi)

        vo.append((asx.close[n]-asx.close[n-1])/asx.close[n-1])
        valo.append(vo)

    vali=np.array(vali)
    valo=np.array(valo)

    return vali, valo, vdate

def split_normalise(vali,valo,vdate,sdate,buffer):

    for i in range(0,len(vdate)):
        if int(vdate[i]) >= sdate and int(vdate[i-1]) < sdate:
            index=i

    print ('Training date range   ', vdate[0],vdate[index-1-buffer])
    print ('Validation data range ', vdate[index],vdate[-1])

    vali_t = vali[:index-buffer,:]
    vali_v = vali[index:,:]
    valo_t = valo[:index-buffer,:]
    valo_v = valo[index:,:]

    si1 = np.shape(vali_t)
    si2 = np.shape(valo_t)
    si3 = np.shape(vali_v)
    nvals_t = si1[0]
    nvals_v = si3[0]
    nvars_i = si1[1]
    nvars_o = si2[1]

    print ('No training values      ', nvals_t)
    print ('No validation values    ', nvals_v)
    print ('No input variables      ', nvars_i)
    print ('No output variables     ', nvars_o)

    # normalise input values for training and validation
    for i in range(0,nvars_i):
        vv = vali_t[:, i]
        std = np.std(vv)
        mean = np.mean(vv)
        for n in range(0,nvals_t):
            vali_t[n, i] = (vali_t[n, i] - mean)/std
        for n in range(0,nvals_v):
            vali_v[n, i] = (vali_v[n, i] - mean)/std

    # normalise output values for training and validation
    for i in range(0,nvars_o):
        vv = valo_t[:, i]
        std = np.std(vv)
        mean = np.mean(vv)
        for n in range(0,nvals_t):
            valo_t[n, i] = (valo_t[n, i] - mean)/std
        for n in range(0,nvals_v):
            valo_v[n, i] = (valo_v[n, i] - mean)/std

    return vali_t, vali_v, valo_t, valo_v


def nn_create_data(name, datatype, batches, params, date_end, daysback, date_split, buffer):

    nn = 0
    for batch in batches:
        print ('Reading batch ', batch)
        nn += 1
        func = globals()["nn_data_" +datatype]
        vali, valo, vdate = func(batch, date_end, daysback, params) #5000
        vali_t2, vali_v2, valo_t2, valo_v2 = split_normalise(vali,valo,vdate,date_split,buffer)
        if nn == 1:
            vali_t = np.copy(vali_t2)
            vali_v = np.copy(vali_v2)
            valo_t = np.copy(valo_t2)
            valo_v = np.copy(valo_v2)
        else:
            vali_t = np.append(vali_t,vali_t2, axis = 0)
            vali_v = np.append(vali_v,vali_v2, axis = 0)
            valo_t = np.append(valo_t,valo_t2, axis = 0)
            valo_v = np.append(valo_v,valo_v2, axis = 0)

    print ('vali_t',np.shape(vali_t))
    print ('valo_t',np.shape(valo_t))
    print ('vali_v',np.shape(vali_v))
    print ('valo_v',np.shape(valo_v))


    dir='/Users/oalves/python/nn/exps/'+name
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(os.path.join(dir,'vali_t.npy'),vali_t)
    np.save(os.path.join(dir,'valo_t.npy'),valo_t)
    np.save(os.path.join(dir,'vali_v.npy'),vali_v)
    np.save(os.path.join(dir,'valo_v.npy'),valo_v)

