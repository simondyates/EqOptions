import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

def EuroOption(S, K, is_put, sig, today, expiry, divyld_cc, bc_cc, r):
    # Inputs can be scalars or arrays
    # Returns scalars if all inputs are scalar, otherwise numpy.ndarray

    t = np.clip((expiry - today) / timedelta(days=365), 0, None)
    q = divyld_cc + bc_cc  # i.e. use a +ve bc if stock hedge is short and -ve if long

    srt = sig * t ** 0.5
    np.seterr(divide='ignore', invalid='ignore') # t might contain zeros, which we handle
    d1 = (np.log(S / K) + t * (r - q + sig ** 2 / 2)) / srt
    d2 = d1 - srt

    nd1 = norm.cdf(d1) * (1 - is_put) + norm.cdf(-d1) * is_put
    nd2 = norm.cdf(d2) * (1 - is_put) + norm.cdf(-d2) * is_put
    price = np.array((S * np.exp(-q * t) * nd1 - K * np.exp(-r * t) * nd2) * (1 - 2 * is_put))
    price = np.nan_to_num(price) # handle the case S==K and t==0

    nd1 = norm.cdf(d1)
    delta = np.array(np.exp(-q * t) * (nd1 - is_put))
    delta = np.nan_to_num(delta)

    t1 = np.exp(-q * t) / (S * sig * t ** 0.5)
    t2 = (2 * np.pi) ** (-0.5) * np.exp(-d1 ** 2 / 2)
    t3 = S / 100  # Comment this out if you don't want d(delta)for 1% stock move
    gamma = np.array(t1 * t2 * t3)
    gamma[t == 0] = 0

    t1 = S * np.exp(-q * t) * t ** 0.5 / 100
    t2 = np.exp(-d1 ** 2 / 2) / (2 * np.pi) ** 0.5
    vega = np.array(t1 * t2)
    vega[t == 0] = 0

    # Return scalars if we can
    if not(price.shape):
        price = price.item()
        delta = delta.item()
        gamma = gamma.item()
        vega = vega.item()

    return price, delta, gamma, vega
