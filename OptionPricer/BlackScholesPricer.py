#! python3
# -*- coding: utf-8 -*-
'''
@File   : BlackScholesPricer.py
@Created: 2025/03/22 06:35
@Author : DONG Yihong
'''

import numpy as np
from scipy.stats import norm

from . import Pricer

class BlackScholesPricer(Pricer):

    def __init__(self):
        super().__init__()
        return
    
    def option_price(self, S, K, T, r, sigma, q=0):
        """
        Calculate the Black-Scholes price for European call and put options.
        
        Args:
            S (float): Spot price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to maturity (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float): Dividend yield.
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate call and put prices
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return call_price, put_price

    @staticmethod
    def black_scholes_vectorized(S, K, T, r, sigma, q=0):
        """
        Vectorized Black-Scholes formula for European call and put options.
        """

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return call_price, put_price
