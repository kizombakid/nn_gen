
import sys

sys.path.append('/home/oscar/analyse/nn_trade')
from nn_main import *
from nn_diagnostics import *

exp = sys.argv[1]


ndef = NN_Def(exp,basedir='/home/oscar/analyse/nn_batch/exps')

if len(ndef.codesA) == 0: 
    fcodesA = True
else:
    fcodesA = False
if len(ndef.codesB) == 0: 
    fcodesB = True
else:
    fcodesB = False
if len(ndef.codesC) == 0: 
    fcodesC = True
else:
    fcodesC = False

for set in ndef.sets:

    ndef.set=set
    if fcodesA: ndef.codesA=[str(set)]
    if fcodesB: ndef.codesB=[str(set)]
    if fcodesC: ndef.codesC=[str(set)]
    print ('****** ',ndef.codesA)


    #for n in range(0,ndef.n_ensembles):

        #nn_main(n,ndef,update_data=False,run_nn=True)

    xx = NN_Diagnostics(ndef)
    xx.summary()
    xx.plot_history()
    xx.plot_scatter()
    xx.plot_scatter_all()
    xx.plot_scatter(type='end')
    xx.plot_scatter_all(type='end')

print ('Completed Experiment ',exp)
