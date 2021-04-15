import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay

def next_expiry(dt):
    # Returns dt if dt is an expiry, otherwise the next occuring expiry
    NYSE = mcal.get_calendar('NYSE')
    nyse_bday = CustomBusinessDay(holidays=NYSE.holidays().holidays)
    first_day = pd.Timestamp(dt.year, dt.month, 1)
    exp_day = 16 + (4 - first_day.dayofweek) % 7 # Options expire on *Saturday*
    this_expiry = pd.Timestamp(dt.year, dt.month, exp_day)
    if dt < this_expiry:
        return this_expiry - nyse_bday
    else:
        first_day = dt + pd.offsets.MonthBegin(1)
        exp_day = 15 + (4 - first_day.dayofweek) % 7
        return pd.Timestamp(first_day.year, first_day.month, exp_day) - nyse_bday