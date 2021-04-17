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
from utils import next_expiry, expiry_list
import random
from OptionPricer import EuroOption

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
stds = stds.iloc[:100]

# Specify final output columns
out_cols = ['PnL', 'LongsP', 'ShortsP', 'LongsD', 'ShortsD', 'LongsG', 'ShortsG',
            'LongsT', 'ShortsT', 'LongsV', 'ShortsV']

def run_sim(start_dt, end_dt):
    # Introduce yourself
    print(f'Processing month {start_dt:%b-%Y}')

    # Initialise result variables
    pnls = pd.DataFrame(columns=out_cols)
    outliers = pd.DataFrame(columns=['Ticker', 'PnL'])

    # Pick tickers
    low_vols = stds.columns[stds.loc[start_dt] < max_vol].to_list()
    pf = random.sample(low_vols, min(pf_size, len(low_vols)))
    shorts = random.sample(pf, int(len(pf) * pct_shorts))

    # Init table for storing daily values per ticker, and pnl storage function
    pf = pd.DataFrame(index=pf)
    def write_date(dt):
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

    # Initialise first day positions and greeks
    for t in pf.index:
        S = K = prices.loc[start_dt, t]
        sig = stds.loc[start_dt, t]
        p, d, g, v = EuroOption(S, K, 'C', sig, start_dt, end_dt, 0, 0, r)
        q = (init_theta * 252) / (50 * sig**2 * S * g) * (-1 if t in shorts else 1)
        pf.loc[t, 'TotPnL'] = 0
        pf.loc[t, ['CurrS', 'K', 'Exp', 'Qty', 'CurrP', 'CurrD', 'CurrG', 'CurrV']] = [S, K, end_dt, q, p, d, g, v]
        pf.loc[t, 'CurrT'] = 50 * S * g * sig ** 2 / 252
    write_date(start_dt)

    # Calculate daily P&L through expiration
    days = [d for d in stds.index if start_dt < d <= end_dt]
    for dt in days:
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
        write_date(dt)
    return pnls, outliers

def main():
    # Split simulation into monthly chunks for parallelization
    expiries = expiry_list(stds.index[0], stds.index[-1])
    dt_ranges = [(expiries[i], expiries[i + 1]) for i in range(len(expiries) - 1)]

    # Initialise result variables
    global pnls, outliers, result_pairs
    idx = stds[expiries[0]:expiries[-1]].index
    pnls = pd.DataFrame(0, index=idx, columns=out_cols)
    outliers = pd.DataFrame()
    result_pairs = [run_sim(*t) for t in dt_ranges]

    for pair in result_pairs:
        pnl, outs = pair
        pnls.loc[pnl.index] += pnl
        outliers.append(outs)

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

if __name__ == '__main__':
    main()