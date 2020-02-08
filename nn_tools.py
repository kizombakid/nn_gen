
from fts_tools import *
from sql_tools import *
import sqlite3
import os
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from pathlib import Path


class NN_Def():

    def __init__(self, name, dir=""):

        self.exp = name

        home = str(Path.home())
        if dir == "": dir = home + '/analyse/nn_exps/' + name

        file = open(dir + "/def.txt", "r")

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



class SPriceEod():

    def __init__(self,code,date,daysback=200, date2=None, daysforward=0):
        if code == 'dj' or code == 'asx200':
            self.read_index(code, date, daysback=daysback)
        else:
            self.read_eod(code, date, daysback=daysback,daysforward=daysforward, date2=date2)

        self.get_indicators()

        return

    def read_eod(self,code,date,daysback=200,daysforward=0, date2=None):
        basedir = get_basedir()
        sqlfile = "eod_main_code.db"
        print (sqlfile)
        sqlfilei = os.path.join(basedir, 'data', sqlfile)
        conni = sqlite3.connect(sqlfilei)
        curi = conni.cursor()

        if date2 != None:
            date0 = min(date,date2)
            date1 = max(date,date2)
        else:
            date0 = date_add(date, -daysback)
            date1 = date_add(date, daysforward)

        prices = sql_get_codedates(curi, code, date0, date1, datetype='source')
        self.date = prices['date']
        self.open = prices['open']
        self.high = prices['high']
        self.low = prices['low']
        self.close = prices['close']
        self.volume = prices['volume']
        self.index = range(0, len(prices))

        return

    def read_index(self,code,date,daysback=200):

        rdate, ropen, rhigh, rlow, rclose, rvolume = sql_rd_eod_index(date, code, daysback=daysback)
        self.date = rdate
        self.open = ropen
        self.high = rhigh
        self.low = rlow
        self.close = rclose
        self.volume = rvolume
        self.index = range(0, len(rdate))

        return


    def get_indicators(self):
        self.ema_short_length = 12
        self.ema_long_length = 30
        self.ema_short = fts_ema(self.close, self.ema_short_length,fillna=True)
        self.ema_long = fts_ema(self.close, self.ema_long_length,fillna=True)
        self.ema_volume = fts_ema(self.volume, 5, fillna=True)
        self.rsi = fts_rsi(self.close,25,fillna=True)


def sql_rd_eod_index(date, code,daysback=200):
    home = str(Path.home())
    sqldir = home+'/python/nn/nn_gen'
    sqlfile = os.path.join(sqldir, code + '.db')
    print (sqlfile)

    rdate = []
    ropen = []
    rhigh = []
    rlow = []
    rclose = []
    rvolume = []

    conn = sqlite3.connect(sqlfile)
    cur = conn.cursor()
    cur.execute('SELECT * FROM price ORDER BY date')
    xx = cur.fetchall()
    conn.close()

    rdate = []
    ropen = []
    rhigh = []
    rlow = []
    rclose = []
    rvolume = []

    nd = min(daysback,len(xx))

    for row in xx:
        if int(row[0]) > date:
            return rdate[-nd:], ropen[-nd:], rhigh[-nd:], rlow[-nd:], rclose[-nd:], rvolume[-nd:]
        rdate.append(int(row[0]))
        ropen.append(float(row[1]))
        rhigh.append(float(row[2]))
        rlow.append(float(row[3]))
        rclose.append(float(row[4]))
        rvolume.append(float(row[5]))

    return rdate[-nd:], ropen[-nd:], rhigh[-nd:], rlow[-nd:], rclose[-nd:], rvolume[-nd:]


class CandlestickItem(pg.GraphicsObject):
    ## Create a subclass of GraphicsObject.
    ## The only required methods are paint() and boundingRect()
    ## (see QGraphicsItem documentation)
    def __init__(self, data,nvals=''):
        pg.GraphicsObject.__init__(self)
     #   self.data = data  ## data must have fields: time, open, close, min, max
        if nvals == '':
            nv=len(data.index)
        else:
            nv=nvals
        self.generatePicture(data,nvals=nv)

    def generatePicture(self,data,nvals=''):

        ## pre-computing a QPicture object allows paint() to run much more quickly,
        ## rather than re-drawing the shapes every time.
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('k', width=2))
        w = (data.index[1] - data.index[0]) / 3.
        if nvals == '':
            nv=len(data.index)
        else:
            nv=nvals
        for i in range(0,nv):
            t = data.index[i]
            open = data.open[i]
            vmax = data.high[i]
            vmin = data.low[i]
            close = data.close[i]


            if vmin != vmax : p.drawLine(QtCore.QPointF(t, vmin), QtCore.QPointF(t, vmax))
            if open > close:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(QtCore.QRectF(t - w, open, w * 2, close - open))

         #p.end()

        return

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        ## boundingRect _must_ indicate the entire area that will be drawn on
        ## or else we will get artifacts and possibly crashing.
        ## (in this case, QPicture does all the work of computing the bouning rect for us)
        return QtCore.QRectF(self.picture.boundingRect())

