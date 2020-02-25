
from pathlib import Path


class NN_Def():

    def __init__(self, name, basedir=""):

        self.exp = name

        home = str(Path.home())
        if basedir == "":
            dir = home + '/analyse/nn_exps/'
        else:
            dir = basedir


        file = open(dir + '/'+name+"/def.ctl", "r")

        self._defaults()
        self.basedir = dir

        for line in file:

            if '=' in line:

                line = line.replace(" ", "").split("=")
                if 'network' in line[0]:
                    exec('self.' + line[0] + '.append(' + str(line[1]) + ')')
                else:
                    exec('self.' + line[0] + '=' + str(line[1]))

    def _defaults(self):

        self.network1A = []
        self.network1B = []
        self.network2 = []
        self.epochs = 1
        self.lookback = 1
        self.batchsize = 1
        self.rmsprop = 0
        self.patience = 1
        self.optimizer = 'rmsprop'



