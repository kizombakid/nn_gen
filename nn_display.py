import numpy as np
import os
from tensorflow.keras.models import model_from_json
from scipy.stats.stats import pearsonr
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from PyQt5.QtGui import QMainWindow
from PyQt5.QtWidgets import QTableWidget,QTableWidgetItem



class NNData():

    def __init__(self,name):

        dir='/Users/oalves/python/nn/exps/'+name
        self.vali_t = np.load(os.path.join(dir,'vali_t.npy'))
        self.valo_t = np.load(os.path.join(dir,'valo_t.npy'))
        self.vali_v = np.load(os.path.join(dir,'vali_v.npy'))
        self.valo_v = np.load(os.path.join(dir,'valo_v.npy'))

        si = np.shape(self.vali_t)
        self.nvars_i = si[1]

        si = np.shape(self.valo_t)
        self.nvars_o = si[1]

class NNModel():
    def __init__(self,data,name):
        # save nn output
        dir = '/Users/oalves/python/nn/exps/'+name
        jsonfile = os.path.join(dir,"model.json")
        h5file = os.path.join(dir,  "model.h5")
        histfile = os.path.join(dir, "history.npy")

        # load json and create model
        json_file = open(jsonfile, 'r')
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        # load weights into new model
        self.model.load_weights(h5file)

        self.history =  np.load(histfile)

        self.predict_t = self.model.predict(data.vali_t).flatten()
        self.predict_v = self.model.predict(data.vali_v).flatten()

class DispTS(QMainWindow):

    def __init__(self, data, nn):

        super(DispTS, self).__init__()

        self.w = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.w.setLayout(self.layout)

        self.plt_series_t1 = data.valo_t[:,0]
        self.plt_series_t2 = nn.predict_t[:]

        self.plt_series_v1 = data.valo_v[:,0]
        self.plt_series_v2 = nn.predict_v[:]

        self.plt_scat_v1 = data.valo_v[:,0]
        self.plt_scat_v2 = nn.predict_v[:]

        self.plt_scat_t1 = data.valo_t[:,0]
        self.plt_scat_t2 = nn.predict_t[:]

        self.series_t = pg.plot()
        self.series_t.plot(self.plt_series_t1)
        aline = self.series_t.plot(self.plt_series_t2)
        aline.setPen(pg.mkPen('r',))
        self.series_t.addItem(aline)

        self.series_v = pg.plot()
        self.series_v.plot(self.plt_series_v1)
        aline = self.series_v.plot(self.plt_series_v2)
        aline.setPen(pg.mkPen('r',))
        self.series_v.addItem(aline)

        self.scatter_t = pg.plot()
        ascat = pg.ScatterPlotItem(self.plt_scat_t1,self.plt_scat_t2)
        self.scatter_t.addItem(ascat)

        self.scatter_v = pg.plot()
        ascat = pg.ScatterPlotItem(self.plt_scat_v1,self.plt_scat_v2)
        self.scatter_v.addItem(ascat)

        # row,column,rowspan,colspand
        self.layout.addWidget(self.series_t, 0, 0, 1, 1)  # plot goes on right side, spanning 3 rows
        self.layout.addWidget(self.series_v, 1, 0, 1, 1)  # plot goes on right side, spanning 3 rows
        self.layout.addWidget(self.scatter_t, 0, 1, 1, 1)  # plot goes on right side, spanning 3 rows
        self.layout.addWidget(self.scatter_v, 1, 1, 1, 1)  # plot goes on right side, spanning 3 rows

        ## Display the widget as a new window
        self.w.show()


name = 'dj1'
data = NNData(name)
nn = NNModel(data,name)

print (len(data.vali_t),len(data.vali_v))
print (' ')
print ('Train     input vs output    ', pearsonr(data.vali_t[:,0],data.valo_t[:,0]))
print ('Train     predict vs output  ',pearsonr(nn.predict_t,data.valo_t[:,0]))
print (' ')
print ('Validate  input vs output    ',pearsonr(data.vali_v[:,0],data.valo_v[:,0]))
print ('Validate  predict vs output  ',pearsonr(nn.predict_v,data.valo_v[:,0]))


app = QtGui.QApplication([])

win1 = DispTS(data, nn)
#win2 = DispNN()

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #QtGui.QApplication.instance().exec_()
        ## Start the Qt event loop
        app.exec_()
'''
Scatter: Predict vs actual (train + validate)
Timeseries: any values (train + validate)
input, output, predict (train + validate) timeseries and scatter
'''