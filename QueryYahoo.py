import pandas as pd
import yfinance as yf

avg_data = pd.read_pickle('SPX+MID+R2K_USD5mADV_40bp_spd.pkl')
tickers = avg_data.index.to_list() + ['SPY']
data = yf.download(
        # tickers list or string as well
        tickers = tickers,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        period = "5y",

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1d",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = False,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = False,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )

idx = pd.IndexSlice
data = data.loc[:, idx[:, 'Adj Close']]
data = data.droplevel(1, axis=1)
data.to_csv('5yrHistData_withSPY.pkl')