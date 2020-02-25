
epochs = 200
n_ensembles =8
n_nets = 1

dates_train = [19130101, 19891231]
dates_val = [19900101, 20091231]
dates_ind = [20100101, 20200101]

sets = ['penrith_m5']


batchsize = 8

optimizer = 'adam'
patience = 10
                   
varso = ['anom']

dnameA = 'awap1'
codesA = []
lookbackA = 2
varsiA = ['apl_c','apl_n','apl_s','apl_e','apl_w']
network1A = ['LSTM',2,{'return_sequences':False}]
network1A = ['Dense',1]

dnameB = 'data2'
codesB = []
lookbackB = 5
varsiB = []
network1B = ['LSTM',64,{'return_sequences':True}]

network2 = ['Dense',1]

codesC = []


