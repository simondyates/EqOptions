# Same as OptionPricer.py but uses vectorized numpy calculations

import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

def EuroOption(S, K, is_put, sig, today, expiry, divyld_cc, bc_cc, r):
    # is_put is Boolean (but then interpreted as int)

    t = (expiry - today) / timedelta(days=365)
    if t <= 0:
        price = (S - K).clip(0) * (1 - is_put) + (K - S).clip(0) * is_put
        delta = gamma = vega = S * 0 # To ensure we return a series or array
        return price, delta, gamma, vega

    q = divyld_cc + bc_cc  # i.e. use a +ve bc if short and -ve if long

    srt = sig * t ** 0.5
    d1 = (np.log(S / K) + t * (r - q + sig ** 2 / 2)) / srt
    d2 = d1 - srt

    nd1 = norm.cdf(d1) * (1 - is_put) + norm.cdf(-d1) * is_put
    nd2 = norm.cdf(d2) * (1 - is_put) + norm.cdf(-d2) * is_put
    price = (S * np.exp(-q * t) * nd1 - K * np.exp(-r * t) * nd2) * (1 - 2 * is_put)

    nd1 = norm.cdf(d1)
    delta = np.exp(-q * t) * (nd1 - is_put) + S * 0 # last term is to maintain a series index if present

    t1 = np.exp(-q * t) / (S * sig * t ** 0.5)
    t2 = (2 * np.pi) ** (-0.5) * np.exp(-d1 ** 2 / 2)
    t3 = S / 100  # Comment this out if you don't want d(delta)for 1% stock move
    gamma = t1 * t2 * t3

    t1 = S * np.exp(-q * t) * t ** 0.5 / 100
    t2 = np.exp(-d1 ** 2 / 2) / (2 * np.pi) ** (0.5)
    vega = t1 * t2

    return price, delta, gamma, vega
