# TO DO: Move greeks to post expiry; Implement SPY Hedge; Multi-thread

# Runs a simplified trading simulation
#
# A trading portfolio consisting of a specified number of short and long names is created
# at the close on the 3rd Friday of each month.  The portfolio is chosen from a universe with
# constraints (such as a maximum historic vol, no biotechs). This portfolio is delta-hedged initially
# and rebalanced on the close daily through expiration, when a new portfolio is created.
# Trades are sized as equal initial theta. We use historic 3-month vols as a proxy for implieds throughout.

import pandas as pd
import numpy as np
from utils import next_expiry
import random
from OptionPricer import EuroOption
import pandas_market_calendars as mcal
from pandas.tseries.offsets import CustomBusinessDay

# Key simulation behaviour variables
pf_size = 500
pct_shorts = .5
spy_hedge = False
init_theta = 25000
std_window = 63
max_vol = .7
r = 0.0011
outlier = 5000000

# Load price and ADV data, set universe
prices = pd.read_pickle('5yrHistData.pkl')
prices = prices.loc[:, prices.notna().all()]
avg_data = pd.read_pickle('SPX+MID+R2K_USD5mADV_40bp_spd.pkl').sort_values('ADV_USD', ascending=False)
universe = avg_data.index.intersection(prices.columns)[:1000]
biotechs = pd.read_csv('Biotechs.csv', index_col='Symbol')
universe = universe.difference(biotechs.index)
prices = prices[universe.append(pd.Index(['SPY']))]

# Calculate returns and rolling standard deviations of returns
returns = np.log(prices) - np.log(prices.shift())
stds = returns.rolling(std_window).std() * 252**0.5
stds = stds[stds.notna().all(axis=1)]

# Initialise result variables
sim_start = next_expiry(stds.index[0])
days = stds.index[stds.index > sim_start]
cols = ['PnL', 'LongsP', 'ShortsP', 'LongsD', 'ShortsD', 'LongsG', 'ShortsG', 'LongsT', 'ShortsT', 'LongsV', 'ShortsV']
pnls = pd.DataFrame(index=days, columns=cols)
outliers = pd.DataFrame(columns=['Ticker', 'PnL'])

# Pick portfolio and purchase at closing prices, with delta-hedge
NYSE = mcal.get_calendar('NYSE')
nyse_bday = CustomBusinessDay(holidays=NYSE.holidays().holidays)

def pick_tickers(dt):
    low_vols = stds.columns[stds.loc[dt] < max_vol].intersection(universe).to_list()
    pf = random.sample(low_vols, min(pf_size, len(low_vols)))
    shorts = random.sample(pf, int(len(pf) * pct_shorts))
    # longs = [t for t in pf if t not in shorts]
    cols = ['CurrS', 'PrevS', 'Qty', 'K', 'Exp', 'CurrP', 'PrevP', 'CurrD', 'PrevD', 'CurrT', 'CurrG', 'CurrV']
    pf = pd.DataFrame(index=pf, columns=cols)
    expiry = next_expiry(dt + nyse_bday)
    for t in pf.index:
        S = K = prices.loc[dt, t]
        sig = stds.loc[dt, t]
        p, d, g, _ = EuroOption(S, K, 'C', sig, dt, expiry, 0, 0, r)
        q = (init_theta * 252) / (50 * sig**2 * S * g) * (-1 if t in shorts else 1)
        pf.loc[t, ['CurrS', 'K', 'Exp', 'Qty', 'CurrP', 'CurrD']] = [S, K, expiry, q, p, d]
    return pf

pf = pick_tickers(sim_start)

# Calculate daily P&L through expiration
start_t = pd.Timestamp.now()
for count, dt in enumerate(days):
    if count > 0:
        now_t = pd.Timestamp.now()
        avg_t = (now_t - start_t) / count
        end_t = now_t + avg_t * (len(days) - count)
        print(f'Processing {count + 1} of {len(days)}, expected completion {end_t:%H:%M}')
    else:
        print(f'Processing 1st day of {len(days)}')
    pf['PrevS'] = pf['CurrS']
    pf['PrevP'] = pf['CurrP']
    pf['PrevD'] = pf['CurrD']
    pf['CurrS'] = prices.loc[dt, pf.index]
    for t in pf.index:
        S, K, expiry = pf.loc[t, ['CurrS', 'K', 'Exp']]
        sig = stds.loc[dt, t]
        pf.loc[t, ['CurrP', 'CurrD', 'CurrG', 'CurrV']] = EuroOption(S, K, 'C', sig, dt, expiry, 0, 0, r)
        pf.loc[t, 'CurrT'] = 50 * S * pf.loc[t, 'CurrG'] * sig**2 / 252
        pf.loc[t, 'OptPnL'] = pf.loc[t, 'Qty'] * (pf.loc[t, 'CurrP'] - pf.loc[t, 'PrevP'])
        pf.loc[t, 'HedgePnL'] = -pf.loc[t, 'Qty'] * pf.loc[t, 'PrevD'] * (pf.loc[t, 'CurrS'] - pf.loc[t, 'PrevS'])
        pf.loc[t, 'TotPnL'] = pf.loc[t, 'OptPnL'] + pf.loc[t, 'HedgePnL']
        if abs(pf.loc[t, 'TotPnL']) > outlier:
            outliers.loc[dt, ['Ticker', 'PnL']] = t, pf.loc[t, 'TotPnL']
    pnls.loc[dt, 'PnL'] = pf['TotPnL'].sum()
    pnls.loc[dt, 'LongsP'] = (pf['Qty'] * pf['CurrP']).loc[pf['Qty'] > 0].sum()
    pnls.loc[dt, 'ShortsP'] = (pf['Qty'] * pf['CurrP']).loc[pf['Qty'] < 0].sum()
    pnls.loc[dt, 'LongsD'] = (pf['Qty'] * pf['CurrS'] * pf['CurrD']).loc[pf['Qty'] > 0].sum()
    pnls.loc[dt, 'ShortsD'] = (pf['Qty'] * pf['CurrS'] * pf['CurrD']).loc[pf['Qty'] < 0].sum()
    pnls.loc[dt, 'LongsG'] = (pf['Qty'] * pf['CurrS'] * pf['CurrG']).loc[pf['Qty'] > 0].sum()
    pnls.loc[dt, 'ShortsG'] = (pf['Qty'] * pf['CurrS'] * pf['CurrG']).loc[pf['Qty'] < 0].sum()
    pnls.loc[dt, 'LongsT'] = (pf['Qty'] * pf['CurrT']).loc[pf['Qty'] > 0].sum()
    pnls.loc[dt, 'ShortsT'] = (pf['Qty'] * pf['CurrT']).loc[pf['Qty'] < 0].sum()
    pnls.loc[dt, 'LongsV'] = (pf['Qty'] * pf['CurrV']).loc[pf['Qty'] > 0].sum()
    pnls.loc[dt, 'ShortsV'] = (pf['Qty'] * pf['CurrV']).loc[pf['Qty'] < 0].sum()
    if dt == expiry:
        pf = pick_tickers(dt)
        print(f'Picked {pf.shape[0]} new tickers')

# Summarise and display results
name = f'Sz={pf_size}_Pct={pct_shorts}_SPY={spy_hedge}_Theta={init_theta}_{pd.Timestamp.now():%Y%m%d_%H%M}'
pnls.to_csv(f'./sims/{name}.csv')
outliers.to_csv(f'./sims/{name}_outs.csv')

pnls.sort_values(by='PnL', inplace=True)
print('5 Worst Days')
print('-'*23)
for i in range(5):
    print(f'{pnls.index[i]:%Y-%m-%d}: {pnls.iloc[i, 0]:,.0f}')
print()
print('5 Best Days')
print('-'*23)
for i in range(5):
    print(f'{pnls.index[-i-1]:%Y-%m-%d}: {pnls.iloc[-i-1, 0]:,.0f}')