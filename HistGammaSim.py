# Runs a simplified trading simulation
#
# A trading portfolio consisting of a specified number of short and long names is created
# at the close on the 3rd Friday of each month.  The portfolio is chosen from a universe with
# constraints (such as a maximum historic vol). This portfolio is delta-hedged initially
# and rebalanced on the close daily through expiration, when a new portfolio is created.
# We use historic 3-month vols as a proxy for implieds throughout.

import pandas as pd
import numpy as np
from utils import next_expiry
import random
from OptionPricer import EuroOption

# Key simulation behaviour variables
pf_size = 100
pct_shorts = .5
spy_hedge = False
max_gamma = 800000
std_window = 63
max_vol = .7
r = 0.0011

# Load price and ADV data, set universe
prices = pd.read_pickle('5yrHistData.pkl')
prices = prices.loc[:, prices.notna().all()]
avg_data = pd.read_pickle('SPX+MID+R2K_USD5mADV_40bp_spd.pkl').sort_values('ADV_USD', ascending=False)
universe = avg_data.index.intersection(prices.columns)[:1000]
prices = prices[universe.append(pd.Index(['SPY']))]

# Calculate returns and rolling standard deviations of returns
returns = np.log(prices) - np.log(prices.shift())
#returns.drop(returns.index[0], inplace=True)
stds = returns.rolling(std_window).std() * 252**0.5
stds = stds[stds.notna().all(axis=1)]

# Initialise result variables
sim_start = next_expiry(stds.index[0])
days = stds.index[stds.index > sim_start]
pnls = pd.Series(name='PnL', dtype='float64')

# Pick portfolio and purchase at closing prices, with delta-hedge
def pick_tickers(dt):
    low_vols = stds.columns[stds.loc[dt] < max_vol].intersection(universe).to_list()
    pf = random.sample(low_vols, pf_size)
    shorts = random.sample(pf, int(pf_size * pct_shorts))
    longs = [t for t in pf if t not in shorts]
    cols = ['CurrS', 'PrevS', 'Qty', 'K', 'Exp', 'CurrV', 'PrevV', 'CurrD', 'PrevD']
    pf = pd.DataFrame(index=pf, columns=cols)
    expiry = next_expiry(dt + pd.Timedelta('1d'))
    for t in pf.index:
        S = K = prices.loc[dt, t]
        sig = stds.loc[dt, t]
        p, d, g, _ = EuroOption(S, K, 'C', sig, dt, expiry, 0, 0, r)
        q = (max_gamma / S) / g * (-1 if t in shorts else 1)
        pf.loc[t, ['CurrS', 'K', 'Exp', 'Qty', 'CurrV', 'CurrD']] = [S, K, expiry, q, p, d]
    return pf

pf = pick_tickers(sim_start)

# Calculate daily P&L through expiration
for i, dt in enumerate(days[:200]):
    print(f'Day {i} of {len(days)}')
    pf['PrevS'] = pf['CurrS']
    pf['PrevV'] = pf['CurrV']
    pf['PrevD'] = pf['CurrD']
    pf['CurrS'] = prices.loc[dt, pf.index]
    for t in pf.index:
        S, K, expiry = pf.loc[t, ['CurrS', 'K', 'Exp']]
        sig = stds.loc[dt, t]
        pf.loc[t, 'CurrV'], pf.loc[t, 'CurrD'], _, _ = EuroOption(S, K, 'C', sig, dt, expiry, 0, 0, r)
        pf.loc[t, 'OptPnL'] = pf.loc[t, 'Qty'] * (pf.loc[t, 'CurrV'] - pf.loc[t, 'PrevV'])
        pf.loc[t, 'HedgePnL'] = -pf.loc[t, 'Qty'] * pf.loc[t, 'PrevD'] * (pf.loc[t, 'CurrS'] - pf.loc[t, 'PrevS'])
        pf.loc[t, 'TotPnL'] = pf.loc[t, 'OptPnL'] + pf.loc[t, 'HedgePnL']
    pnls[dt] = pf['TotPnL'].sum()
    if dt == expiry:
        pf = pick_tickers(dt)

# Summarise and display results
pnls.sort_values(inplace=True)
name = f'Sz={pf_size}_Pct={pct_shorts}_SPY={spy_hedge}_Gam={max_gamma}_{pd.Timestamp.now():%Y%m%d_%H%M}'
pnls.to_csv(f'./sims/{name}.csv')

print('5 Worst Days')
print('-'*23)
for i in range(5):
    print(f'{pnls.index[i]:%Y-%m-%d}: {pnls.iloc[i]:,.0f}')
print()
print('5 Best Days')
print('-'*23)
for i in range(5):
    print(f'{pnls.index[-i-1]:%Y-%m-%d}: {pnls.iloc[-i-1]:,.0f}')