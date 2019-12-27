
from fts_tools import *
import sqlite3
import numpy as np
import datetime
import os


class SPriceEod():

    def __init__(self,code,date,daysback=200):
        if code == 'dj' or code == 'asx200':
            self.read_index(code, date, daysback=daysback)
        else:
            self.read_eod(code, date, daysback=daysback)

        self.get_indicators()

        return

    def read_eod(self,code,date,daysback=200):
        basedir = get_basedir()
        sqlfile = "eod_main_code.db"
        sqlfilei = os.path.join(basedir, 'data', sqlfile)
        conni = sqlite3.connect(sqlfilei)
        curi = conni.cursor()
        date0 = date_add(date, -daysback)
        date2 = date

        prices = sql_get_codedates(curi, code, date0, date2, datetype='source')
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
    sqldir = '/Users/oalves/python/nn/nn_gen'
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


