import pandas as pd
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay

NYSE = mcal.get_calendar('NYSE') # Expiries sometimes fall on Good Friday
nyse_bday = CustomBusinessDay(holidays=NYSE.holidays().holidays)

def next_expiry(dt):
    # Returns dt if dt is an expiry, otherwise the next occurring expiry
    first_day = pd.Timestamp(dt.year, dt.month, 1)
    exp_day = 16 + (4 - first_day.dayofweek) % 7 # Options technically expire on *Saturday*
    this_expiry = pd.Timestamp(dt.year, dt.month, exp_day)
    if dt < this_expiry:
        return this_expiry - nyse_bday # Last trading day, which is what we care about
    else:
        first_day = dt + pd.offsets.MonthBegin(1)
        exp_day = 16 + (4 - first_day.dayofweek) % 7
        return pd.Timestamp(first_day.year, first_day.month, exp_day) - nyse_bday

def expiry_list(start_dt, end_dt):
    # Return all expiries between start_dt and end_dt double-inclusive
    expiries = [next_expiry(start_dt)]
    while (dt := next_expiry(expiries[-1] + nyse_bday)) <= end_dt:
        expiries.append(dt)
    return expiries