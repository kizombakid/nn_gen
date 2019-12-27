import pandas as pd
import numpy as np

def fts_ma(ts,window):
    # ts numpy array or list
    # calculates exponential moving average
    tsp=pd.Series(ts)

    ma = tsp.rolling(window).mean()
    
    return np.array(ma)

def fts_ema(series, periods, fillna=False):
    sr=pd.Series(series)
    if fillna:
        return np.array(sr.ewm(span=periods, min_periods=0).mean())
    return np.array(sr.ewm(span=periods, min_periods=periods).mean())

def fts_rsi(cls, n, fillna=False):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    close=pd.Series(cls)
    
    diff = close.diff(1)
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = fts_ema(up, n)
    emadn = fts_ma(dn, n)

    rsi = pd.Series(100 * emaup / (emaup + emadn))
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return np.array(pd.Series(rsi, name='rsi'))

