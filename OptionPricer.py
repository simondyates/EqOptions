from math import exp, log, pi
from scipy.stats import norm
from datetime import timedelta

def EuroOption(S, K, putcall, sig, today, expiry, divyld_cc, bc_cc, r):
    putcall = putcall.lower()

    t = (expiry - today) / timedelta(days=365)
    if t <= 0:
        if putcall == 'c':
            price = max(0, S - K)
        else:
            price = max(0, K - S)
        delta = gamma = vega = 0
        return price, delta, gamma, vega

    q = divyld_cc + bc_cc  # i.e. use a +ve bc if short and -ve if long

    fwd = S * exp((r - q) * t)

    srt = sig * t ** 0.5
    d1 = (log(S / K) + t * (r - q + sig ** 2 / 2)) / srt
    d2 = d1 - srt

    if putcall == 'c':
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        price = S * exp(-q * t) * nd1 - K * exp(-r * t) * nd2
    else:
        nd1 = norm.cdf(-d1)
        nd2 = norm.cdf(-d2)
        price = -S * exp(-q * t) * nd1 + K * exp(-r * t) * nd2

    nd1 = norm.cdf(d1)
    if putcall == 'c':
        delta = exp(-q * t) * nd1
    else:
        delta = exp(-q * t) * (nd1 - 1)

    t1 = exp(-q * t) / (S * sig * t ** 0.5)
    t2 = (2 * pi) ** (-0.5) * exp(-d1 ** 2 / 2)
    t3 = S / 100  # Comment this out if you don't want d(delta)for 1% stock move
    gamma = t1 * t2 * t3

    t1 = S * exp(-q * t) * t ** 0.5 / 100
    t2 = exp(-d1 ** 2 / 2) / (2 * pi) ** (0.5)
    vega = t1 * t2

    return price, delta, gamma, vega
