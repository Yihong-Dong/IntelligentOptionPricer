#! python3
# -*- coding: utf-8 -*-
'''
@File   : MonteCarloPricer.py
@Created: 2025/03/22 18:29
@Author : DONG Yihong
'''

import random
import math
from concurrent.futures import ThreadPoolExecutor
import time

from . import Pricer

# Integrated Model: Combines all features

class MonteCarloPricer(Pricer):

    def __init__(self, n_simulations: int = 1000):
        super().__init__()
        self.n_simulations = n_simulations
        return

    @staticmethod
    def generate_price_paths(S, r, q, sigma, T, n_simulations):
        """
        Lazy generation of price paths using yield.
        """
        for _ in range(n_simulations // 2):
            Z = random.gauss(0, 1)
            yield S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
            yield S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * (-Z))  # Antithetic variate

    @staticmethod
    def simulate_path(S, K, T, r, sigma, q, Z):
        """
        Simulate a single price path and calculate payoffs.
        """
        S_T = S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
        call_payoff = max(S_T - K, 0)
        put_payoff = max(K - S_T, 0)
        return call_payoff, put_payoff
    
    def option_price(self, S, K, T, r, sigma, q, n_workers=4):
        """
        Integrated Monte Carlo model with all features:
        - Lazy generation (yield)
        - Antithetic variables
        - Random matrix for Greeks
        - Parallelization
        
        Args:
            S (float): Spot price of the underlying asset.
            K (float): Strike price of the option.
            T (float): Time to maturity (in years).
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            q (float): Dividend yield.
        """
        n_simulations = self.n_simulations

        # Generate price paths and calculate payoffs
        call_payoffs = []
        put_payoffs = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                lambda z: __class__.simulate_path(S, K, T, r, sigma, q, z),
                [random.gauss(0, 1) for _ in range(n_simulations)]
            ))

        call_payoffs, put_payoffs = zip(*results)

        # Calculate option prices
        call_price = math.exp(-r * T) * (sum(call_payoffs) / n_simulations)
        put_price = math.exp(-r * T) * (sum(put_payoffs) / n_simulations)

        # Approximate Delta (finite difference method)
        delta_S = 0.01 * S
        S_T_plus = [S + delta_S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z) for z in [random.gauss(0, 1) for _ in range(n_simulations)]]
        S_T_minus = [S - delta_S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z) for z in [random.gauss(0, 1) for _ in range(n_simulations)]]

        call_payoff_plus = [max(s - K, 0) for s in S_T_plus]
        call_payoff_minus = [max(s - K, 0) for s in S_T_minus]

        call_delta = math.exp(-r * T) * (sum(call_payoff_plus) - sum(call_payoff_minus)) / (2 * delta_S * n_simulations)

        return call_price, put_price, call_delta

    # Individual Models
    @staticmethod
    def monte_carlo_basic(S, K, T, r, sigma, q, n_simulations=10000):
        """
        Basic Monte Carlo implementation without specialized libraries.
        """
        call_payoffs = []
        put_payoffs = []

        for _ in range(n_simulations):
            Z = random.gauss(0, 1)
            S_T = S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
            call_payoffs.append(max(S_T - K, 0))
            put_payoffs.append(max(K - S_T, 0))

        call_price = math.exp(-r * T) * (sum(call_payoffs) / n_simulations)
        put_price = math.exp(-r * T) * (sum(put_payoffs) / n_simulations)

        return call_price, put_price

    @staticmethod
    def generate_lazy_price_paths(S, r, q, sigma, T, n_simulations):
        """
        Lazy generation of price paths using yield.
        """
        for _ in range(n_simulations):
            Z = random.gauss(0, 1)
            yield S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)

    @staticmethod
    def monte_carlo_lazy(S, K, T, r, sigma, q, n_simulations=10000):
        """
        Monte Carlo with lazy generation.
        """

        call_payoffs = []
        put_payoffs = []

        for S_T in __class__.generate_lazy_price_paths(S, r, q, sigma, T, n_simulations):
            call_payoffs.append(max(S_T - K, 0))
            put_payoffs.append(max(K - S_T, 0))

        call_price = math.exp(-r * T) * (sum(call_payoffs) / n_simulations)
        put_price = math.exp(-r * T) * (sum(put_payoffs) / n_simulations)

        return call_price, put_price

    @staticmethod
    def monte_carlo_antithetic(S, K, T, r, sigma, q, n_simulations=10000):
        """
        Monte Carlo with antithetic variables.
        """
        call_payoffs = []
        put_payoffs = []

        for _ in range(n_simulations // 2):
            Z = random.gauss(0, 1)
            S_T1 = S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
            S_T2 = S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * (-Z))
            call_payoffs.append(max(S_T1 - K, 0))
            call_payoffs.append(max(S_T2 - K, 0))
            put_payoffs.append(max(K - S_T1, 0))
            put_payoffs.append(max(K - S_T2, 0))

        call_price = math.exp(-r * T) * (sum(call_payoffs) / n_simulations)
        put_price = math.exp(-r * T) * (sum(put_payoffs) / n_simulations)

        return call_price, put_price

    @staticmethod
    def monte_carlo_greeks(S, K, T, r, sigma, q, n_simulations=10000):
        """
        Monte Carlo with random matrix to approximate Greeks.
        """
        Z = [random.gauss(0, 1) for _ in range(n_simulations)]
        S_T = [S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z) for z in Z]

        # Calculate option prices
        call_payoffs = [max(s - K, 0) for s in S_T]
        put_payoffs = [max(K - s, 0) for s in S_T]
        call_price = math.exp(-r * T) * (sum(call_payoffs) / n_simulations)
        put_price = math.exp(-r * T) * (sum(put_payoffs) / n_simulations)

        # Approximate Delta (finite difference method)
        delta_S = 0.01 * S
        S_T_plus = [(S + delta_S) * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z) for z in Z]
        S_T_minus = [(S - delta_S) * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z) for z in Z]

        call_payoff_plus = [max(s - K, 0) for s in S_T_plus]
        call_payoff_minus = [max(s - K, 0) for s in S_T_minus]

        call_delta = math.exp(-r * T) * (sum(call_payoff_plus) - sum(call_payoff_minus)) / (2 * delta_S * n_simulations)

        return call_price, put_price, call_delta

    @staticmethod
    def monte_carlo_parallel(S, K, T, r, sigma, q, n_simulations=10000, n_workers=4):
        """
        Parallelized Monte Carlo simulation.
        """
        def simulate_path(_):
            Z = random.gauss(0, 1)
            S_T = S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
            call_payoff = max(S_T - K, 0)
            put_payoff = max(K - S_T, 0)
            return call_payoff, put_payoff

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(simulate_path, range(n_simulations)))

        call_payoffs, put_payoffs = zip(*results)

        call_price = math.exp(-r * T) * (sum(call_payoffs) / n_simulations)
        put_price = math.exp(-r * T) * (sum(put_payoffs) / n_simulations)

        return call_price, put_price
