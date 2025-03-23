#! python3
# -*- coding: utf-8 -*-
'''
The common parent class of Pricer classes.

@File   : Pricer.py
@Created: 2025/03/23 08:50
@Author : DONG Yihong
'''

class Pricer():
    def __init__(self):
        return
    
    def option_price(self, S, K, r, sigma, T, q):
        """
        Predict the option price.
        
        Args:
            S (float): Spot price of the underlying asset.
            K (float): Strike price of the option.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            T (float): Time to maturity (in years).
            q (float): Dividend yield.
            
        Returns:
            tuple: A tuple containing the predicted call price and put price.
        """
        pass