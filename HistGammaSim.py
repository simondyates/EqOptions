import pandas as pd
import random

data = pd.read_pickle('5yrHistData.pkl')
avg_data = pd.read_pickle('SPX+MID+R2K_USD5mADV_40bp_spd.pkl')

start_dt = pd.to_datetime('2016-04-14')
data = data[data.index >= start_dt]
complete_data = data.loc[:, data.notna().all()]
avg_data = avg_data.loc[complete_data.columns[:-1]] # Cuts of SPY in last column
universe = avg_data.sort_values('ADV_USD', ascending=False).index.tolist()

returns = complete_data / complete_data.shift(1) - 1
returns.drop(start_dt, inplace=True)
spy_rets = returns['SPY']
returns.drop('SPY', axis=1, inplace=True)

pf_size = 500
portfolio = universe[:pf_size] # random.sample(returns.columns.tolist(), pf_size)
pf_short_pct = .5
shorts = random.sample(portfolio, int(pf_size * pf_short_pct))
longs = [p for p in portfolio if p not in shorts]

gamma_limit = 800000 # $gamma i.e change in $delta for 1% move in spot
net_gamma = gamma_limit * (len(longs) - len(shorts))
pnls = 50 * gamma_limit * returns**2

avg_vol = returns[portfolio].values.std() * 252**0.5
spy_vol = spy_rets.std() * 252**0.5

short_pnl = -pnls[shorts].sum(axis=1) + 50 * (avg_vol**2 / 252) * gamma_limit * len(shorts)
long_pnl = pnls[longs].sum(axis=1) - 50 * (avg_vol**2 / 252) * gamma_limit * len(longs)
spy_pnl = -50 * net_gamma * spy_rets**2 + 50 * (spy_vol**2 / 252) * net_gamma
tot_pnl = short_pnl + long_pnl + spy_pnl
print(f'Short min {short_pnl.min():,.0f}')
print(f'Short max {short_pnl.max():,.0f}')
print(f'Long min {long_pnl.min():,.0f}')
print(f'Long max {long_pnl.max():,.0f}')
print(f'SPY min {spy_pnl.min():,.0f}')
print(f'SPY max {spy_pnl.max():,.0f}')
print(f'Tot min {tot_pnl.min():,.0f}')
print(f'Tot max {tot_pnl.max():,.0f}')
#tot_pnl.hist(bins = 30)