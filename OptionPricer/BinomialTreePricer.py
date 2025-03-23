#! python3
# -*- coding: utf-8 -*-
'''
@File   : BinomialTreePricer.py
@Created: 2025/03/22 16:55
@Author : DONG Yihong
'''

import numpy as np

from . import Pricer

class BinomialTreePricer(Pricer):

    def __init__(self):
        super().__init__()
        return
    
    def option_price(self, S, K, T, r, sigma, q, n_steps=100):
        """
        Calculate the price of European call and put options using the Cox-Ross-Rubinstein binomial tree model.

        Args:
            S (float): Current stock price (spot price)
            K (float): Strike price
            T (float): Time to maturity (in years)
            r (float): Risk-free interest rate (annualized)
            sigma (float): Volatility of the stock's returns (annualized)
            q (float): Dividend yield
            n_steps (int): Number of time steps in the tree (default = 100)

        Returns:
            call_price (float): Estimated call option price
            put_price (float): Estimated put option price
        """
        # Calculate time step and up/down factors
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset prices at maturity
        S_T = S * (u ** np.arange(n_steps, -1, -1)) * (d ** np.arange(0, n_steps + 1))

        # Calculate payoffs at maturity
        call_payoff = np.maximum(S_T - K, 0)
        put_payoff = np.maximum(K - S_T, 0)

        # Work backward through the tree
        for step in range(n_steps - 1, -1, -1):
            call_payoff = np.exp(-r * dt) * (p * call_payoff[:-1] + (1 - p) * call_payoff[1:])
            put_payoff = np.exp(-r * dt) * (p * put_payoff[:-1] + (1 - p) * put_payoff[1:])

        return call_payoff[0], put_payoff[0]

    def recursive(self, S, K, T, r, sigma, q, n_steps, step=0, price=None, option_type='call'):
        """
        Recursive implementation of the Binomial Tree model for European call and put options.

        Parameters:
        S (float): Current stock price (spot price)
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the stock's returns (annualized)
        q (float): Dividend yield
        n_steps (int): Number of time steps in the tree
        step (int): Current step in the tree (default = 0)
        price (float): Current asset price (default = None)
        option_type (str): 'call' or 'put' (default = 'call')

        Returns:
        option_price (float): Estimated option price
        """
        if price is None:
            price = S  # Initialize with the current stock price

        # Base case: At maturity, calculate the payoff
        if step == n_steps:
            if option_type == 'call':
                return max(price - K, 0)  # Call option payoff
            elif option_type == 'put':
                return max(K - price, 0)   # Put option payoff
            else:
                raise ValueError("option_type must be 'call' or 'put'")

        # Calculate up and down factors
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

        # Recursively calculate option prices for up and down moves
        price_up = price * u
        price_down = price * d
        option_price_up = __class__.binomial_tree_recursive(S, K, T, r, sigma, q, n_steps, step + 1, price_up, option_type)
        option_price_down = __class__.binomial_tree_recursive(S, K, T, r, sigma, q, n_steps, step + 1, price_down, option_type)

        # Discount the expected option price
        option_price = np.exp(-r * dt) * (p * option_price_up + (1 - p) * option_price_down)
        return option_price

    def binomial_tree_vectorized(self, S, K, T, r, sigma, q, n_steps=100):
        """
        Vectorized implementation of the Binomial Tree model for European call and put options.

        Parameters:
        S (float): Current stock price (spot price)
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility of the stock's returns (annualized)
        q (float): Dividend yield
        n_steps (int): Number of time steps in the tree (default = 100)

        Returns:
        call_price (float): Estimated call option price
        put_price (float): Estimated put option price
        """
        # Calculate time step and up/down factors
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

        # Initialize asset prices at maturity
        S_T = S * (u ** np.arange(n_steps, -1, -1)) * (d ** np.arange(0, n_steps + 1))

        # Calculate payoffs at maturity
        call_payoff = np.maximum(S_T - K, 0)
        put_payoff = np.maximum(K - S_T, 0)

        # Work backward through the tree
        for step in range(n_steps - 1, -1, -1):
            call_payoff = np.exp(-r * dt) * (p * call_payoff[:-1] + (1 - p) * call_payoff[1:])
            put_payoff = np.exp(-r * dt) * (p * put_payoff[:-1] + (1 - p) * put_payoff[1:])

        return call_payoff[0], put_payoff[0]
