import pandas as pd

def next_expiry(dt):
    # Returns dt if dt is an expiry, otherwise the next occuring expiry
    first_day = pd.Timestamp(dt.year, dt.month, 1)
    exp_day = 15 + (4 - first_day.dayofweek) % 7
    this_expiry = pd.Timestamp(dt.year, dt.month, exp_day)
    if dt <= this_expiry:
        return this_expiry
    else:
        first_day = dt + pd.offsets.MonthBegin(1)
        exp_day = 15 + (4 - first_day.dayofweek) % 7
        return pd.Timestamp(first_day.year, first_day.month, exp_day)