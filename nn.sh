
cat > def.txt <<eof

epochs = 500
n_ensembles = 3

type = 1

dates_train = [20000101, 20151101]
dates_val = [20160101, 20190101]

code = 'CBA'

dnameA = 'data1'
codesA = ['CBA']
lookbackA = 60
varsiA = ['open', 'high','low','close', 'volume','ema1', 'ema2','ema3','rsi1','rsi2','rsi3','emad1','emad2','emag1','emag2','emag3']
varsoA = ['lead0']
network1A = ['LSTM',64,{'return_sequences':True}]
network1A = ['LSTM',64,{'return_sequences':True}]
network1A = ['LSTM',64,{'return_sequences':True}]
network1A = ['LSTM',64,{'return_sequences':False}]
network1A = ['Dense',1]

batchsizeT = 4
batchsizeV = 4

Optimiser = 'adam'
patience = 50
                   
network1B  ['LSTM',16,{'return_sequences':False}]
network1B  ['LSTM',16,{'return_sequences':False}]

Network2  ['Dense',1]

eof


python <<eof
import sys
sys.path.append('/home/oscar/analyse/nn_trade')
from nn_main import *
cd=os.getcwd()
exp=cd.split('/')[-1]

ndef = NN_Def(exp)
codes = ['CBA','ANZ','NAB','WBC','RIO','BHP','IAG','FMG','SUN','WOW','WES','WPL','IPL','BXB','AMC','NCM','TLS']
codes = ['CBA']


for cod in codes:
    ndef.code=cod
    ndef.codesA=[cod]
    for n in range(0,ndef.n_ensembles):
        nn_main(n,ndef,update_data=False)

    xx = NN_Diagnostics(ndef)
    xx.summary()
    xx.plot_history()
    xx.plot_scatter()


eof
