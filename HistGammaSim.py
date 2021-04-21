# TO DO: Implement SPY Hedge

# Runs a simplified trading simulation: Multi-Threaded Implementation with vectorized portfolio pricing
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
import concurrent.futures

# Key simulation behaviour variables
pf_size = 500 # underlyings
pct_shorts = .5
spy_hedge = False # not implemented yet anyway
init_theta = 25000 # per name |theta| at position inception
max_pf_gamma = 100000000 # max portfolio gamma at inception
std_window = 63 # use 3 month hist vols
max_vol = .7 # don't trade anything more volatile than this
r = 0.0011 # risk free rate
outlier = 5000000 # record individual name-day P&L with abs mag bigger than this

# Load price and ADV data, set universe
prices = pd.read_pickle('5yrHistData.pkl')
prices = prices.loc[:, prices.notna().all()]
avg_data = pd.read_pickle('SPX+MID+R2K_USD5mADV_40bp_spd.pkl').sort_values('ADV_USD', ascending=False)
# avg_data contains info like ADV, spread width etc.  We're interested in ADV so we can filter the universe
universe = avg_data.index.intersection(prices.columns)[:1000] # 1,000 most traded stocks we have data for
biotechs = pd.read_csv('Biotechs.csv', index_col='Symbol')
universe = universe.difference(biotechs.index) # exclude biotechs
prices = prices[universe.append(pd.Index(['SPY']))] # for when SPY hedge implemented

# Calculate returns and rolling standard deviations of returns
returns = np.log(prices) - np.log(prices.shift())
stds = returns.rolling(std_window).std() * 252**0.5
stds = stds[stds.notna().all(axis=1)]

# Specify final output columns
out_cols = ['PnL', 'LongsP', 'ShortsP', 'LongsD', 'ShortsD', 'LongsG', 'ShortsG',
            'LongsT', 'ShortsT', 'LongsV', 'ShortsV']
# PnL, option premium value, delta, gamma, theta, vega

# Function to run the sim between two dates
def run_sim(dts):
    start_dt = dts[0]
    end_dt = dts[1]

    # Introduce yourself
    print(f'Processing month {start_dt:%b-%Y}')

    # Initialise result variables and filters
    pnls = pd.DataFrame(columns=out_cols)
    outliers = pd.DataFrame()
    low_vols = stds.columns[stds.loc[start_dt] < max_vol].to_list()

    # Try multiple portfolios until one meets max portfolio gamma constraint
    pf_gamma = max_pf_gamma + 1
    while pf_gamma > max_pf_gamma:
        # Pick tickers
        pf = random.sample(low_vols, min(pf_size, len(low_vols)))
        shorts = random.sample(pf, int(len(pf) * pct_shorts))
        pf = pd.DataFrame(index=pf)
        # Set quantities
        S = K = prices.loc[start_dt, pf.index]
        sig = stds.loc[start_dt, pf.index]
        p, d, g, v = EuroOption(S, K, False, sig, start_dt, end_dt, 0, 0, r)
        q = (init_theta * 252) / (50 * sig**2 * S * g) * (1 - 2 * pf.index.isin(shorts))
        pf['TotPnL'] = 0
        pf['CurrS'] = S
        pf['K'] = K
        pf['Qty'] = q
        pf['CurrP'] = p
        pf['CurrD'] = d
        pf['CurrG'] = g
        pf['CurrV'] = v
        pf['CurrT'] = 50 * S * g * sig ** 2 / 252
        # Check compliance
        pf_gamma = abs((pf['Qty'] * pf['CurrS'] * pf['CurrG']).sum())

    # Write initial values to pnls
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

    write_date(start_dt)

    # Calculate daily P&L through expiration
    days = [d for d in stds.index if start_dt < d <= end_dt]
    for dt in days:
        pf['PrevS'] = pf['CurrS']
        pf['PrevP'] = pf['CurrP']
        pf['PrevD'] = pf['CurrD']
        pf['CurrS'] = prices.loc[dt, pf.index]
        S, K = pf['CurrS'], pf['K']
        sig = stds.loc[dt, pf.index]
        p, d, g, v = EuroOption(S, K, False, sig, dt, end_dt, 0, 0, r)
        pf['CurrP'] = p
        pf['CurrD'] = d
        pf['CurrG'] = g
        pf['CurrV'] = v
        pf['CurrT'] = 50 * S * pf['CurrG'] * sig**2 / 252
        pf['OptPnL'] = pf['Qty'] * (pf['CurrP'] - pf['PrevP'])
        pf['HedgePnL'] = -pf['Qty'] * pf['PrevD'] * (pf['CurrS'] - pf['PrevS'])
        pf['TotPnL'] = pf['OptPnL'] + pf['HedgePnL']
        outs = [[dt, tk, pf.loc[tk, 'TotPnL']] for tk in pf[abs(pf['TotPnL']) > outlier].index]
        if len(outs) > 0:
            outliers = outliers.append(outs)
        write_date(dt)
    return pnls, outliers

def main():
    # Split simulation into monthly chunks for parallelization
    expiries = expiry_list(stds.index[0], stds.index[-1])
    dt_ranges = [(expiries[i], expiries[i + 1]) for i in range(len(expiries) - 1)]

    # Initialise result variables
    idx = stds[expiries[0]:expiries[-1]].index
    pnls = pd.DataFrame(0, index=idx, columns=out_cols)
    outliers = pd.DataFrame()

    # Run sim using multi-threading across available CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_pairs = executor.map(run_sim, dt_ranges)

    # Reassemble the chunks
    for pair in result_pairs:
        pnl, outs = pair
        pnls.loc[pnl.index] += pnl
        outliers = outliers.append(outs)

    # Save and display results
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