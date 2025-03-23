#! python3
# -*- coding: utf-8 -*-
'''
@File   : Benchmarking.py
@Created: 2025/03/22 18:40
@Author : DONG Yihong
'''

import time
from OptionPricer import BlackScholesPricer, BinomialTreePricer, MonteCarloPricer, NeuralNetworkPricer
from risk_free_curve import RiskFreeCurve
from volatility_surface import get_spot_price, get_implied_volatility, get_real_option_price

def get_user_input():
    """
    Get user inputs for strike price, time to maturity, and dividend yield.
    """
    print("Welcome to the Options Pricer for S&P 500!")
    K = float(input("Enter the strike price (K): "))
    T = float(input("Enter the time to maturity (T) in years: "))
    q = 0
    return K, T, q

def generate_result_text(method: str, call_price: float, put_price: float, time_ms: float, real_call_price: float, real_put_price: float):
    """Generate the result text.
    
    Args:
        method (str): The name of the pricing method.
        call_price (float): The predicted call option price.
        put_price (float): The predicted put option price.
        time_ms (float): The timespan in millisecond the pricer take to run the pricer.
        real_call_price (float): Real market price of the call option.
        real_put_price (float): Real market price of the put option.
    """
    call_error = abs((call_price - real_call_price) / real_call_price) * 100
    put_error = abs((put_price - real_put_price) / real_put_price) * 100
    return f"| {method:<25} | {call_price:>10.2f} | {put_price:>9.2f} | {time_ms:>9.2f} | {call_error:>14.2f} | {put_error:>13.2f} |"

def compare_methods(S, K, T, r, sigma, q, real_call_price, real_put_price, running_choice: int = 2, model_choice: int = 0):
    """
    Compare the performance of all 5 pricing models.
    
    Args:
        S (float): Spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        q (float): Dividend yield.
        real_call_price (float): Real market price of the call option.
        real_put_price (float): Real market price of the put option.
        running_choice (int, optional): Choose 1 or 2: 1. Run a single pricing model; 2. Compare all models (benchmark). Default 2.
        model_choice (int, optional): Choose from 1 to 5. Default 0.
    """

    result_texts = list()
    # Black-Scholes
    if (running_choice == 2 or model_choice == 1):
        start_time = time.time()
        bs_pricer = BlackScholesPricer()
        bs_call, bs_put = bs_pricer.option_price(S, K, T, r, sigma, q)
        # bs_call, bs_put = BlackScholesPricer.option_price(S, K, T, r, sigma, q)
        bs_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        result_texts.append(generate_result_text(method="Black-Scholes", 
            call_price=bs_call, put_price=bs_put, time_ms=bs_time, 
            real_call_price=real_call_price, real_put_price=real_put_price))

    # Monte Carlo (1,000 scenarios)
    if (running_choice == 2 or model_choice == 2):
        start_time = time.time()
        mc_pricer_1k = MonteCarloPricer(n_simulations=1000)
        mc_call_1k, mc_put_1k, _ = mc_pricer_1k.option_price(S, K, T, r, sigma, q)
        # mc_call_1k, mc_put_1k, _ = MonteCarloPricer.option_price(S, K, T, r, sigma, q, n_simulations=1000)
        mc_time_1k = (time.time() - start_time) * 1000
        result_texts.append(generate_result_text(method="Monte Carlo (1,000 sim)", 
            call_price=mc_call_1k, put_price=mc_put_1k, time_ms=mc_time_1k, 
            real_call_price=real_call_price, real_put_price=real_put_price))

    # Monte Carlo (10,000 scenarios)
    if (running_choice == 2 or model_choice == 3):
        start_time = time.time()
        mc_pricer_10k = MonteCarloPricer(n_simulations=10000)
        mc_call_10k, mc_put_10k, _ = mc_pricer_10k.option_price(S, K, T, r, sigma, q)
        # mc_call_10k, mc_put_10k, _ = MonteCarloPricer.option_price(S, K, T, r, sigma, q, n_simulations=10000)
        mc_time_10k = (time.time() - start_time) * 1000
        result_texts.append(generate_result_text(method="Monte Carlo (10,000 sim)", 
            call_price=mc_call_10k, put_price=mc_put_10k, 
            time_ms=mc_time_10k, 
            real_call_price=real_call_price, real_put_price=real_put_price))

    # Binomial Tree
    if (running_choice == 2 or model_choice == 4):
        start_time = time.time()
        bt_pricer = BinomialTreePricer()
        bt_call, bt_put = bt_pricer.option_price(S, K, T, r, sigma, q)
        # bt_call, bt_put = BinomialTreePricer.option_price(S, K, T, r, sigma, q)
        bt_time = (time.time() - start_time) * 1000
        result_texts.append(generate_result_text(method="Binomial Tree", 
            call_price=bt_call, put_price=bt_put, 
            time_ms=bt_time, 
            real_call_price=real_call_price, real_put_price=real_put_price))

    # Neural Network
    if (running_choice == 2 or model_choice == 5):
        start_time = time.time()
        nn_pricer = NeuralNetworkPricer()
        nn_call, nn_put = nn_pricer.option_price(S, K, T, r, sigma, q)
        # nn_call, nn_put = NeuralNetworkPricer.neural_network_option_price(S, K, T, r, sigma, q)
        nn_time = (time.time() - start_time) * 1000
        # results.append(("Neural Network", nn_call, nn_put, nn_time))
        result_texts.append(generate_result_text(method="Neural Network", 
            call_price=nn_call, put_price=nn_put, 
            time_ms=nn_time, 
            real_call_price=real_call_price, real_put_price=real_put_price))

    print("\nReal Prices:")
    print(f"Call Price: {real_call_price}")
    print(f"Put Price: {real_put_price}")
    print("\nBenchmark Results and Error Comparison:")
    print("| Method                    | Call Price | Put Price | Time (ms) | Call Error (%) | Put Error (%) |")
    print("|---------------------------|------------|-----------|-----------|----------------|---------------|")
    _ = [print(result) for result in result_texts]
    print("|---------------------------|------------|-----------|-----------|----------------|---------------|")

if __name__ == '__main__':
    
    # Retrieve market data dynamically
    S = get_spot_price()  # Current spot price of S&P 500
    sigma = get_implied_volatility()  # Implied volatility
    r = RiskFreeCurve.get_risk_free_rate()  # Risk-free rate

    # Get user inputs
    K, T, q = get_user_input()
    real_call_price, real_put_price, expiration_date = get_real_option_price(T=T, K=K)  # Real option prices from yfinance

    # Display market data and user inputs
    print("\nMarket Data:")
    print(f"Spot Price (S): {S:.2f}")
    print(f"Implied Volatility (σ): {sigma:.4f}")
    print(f"Risk-Free Rate (r): {r:.4f}")
    print(f"Real Call Price (from yfinance): {real_call_price:.2f}")
    print(f"Real Put Price (from yfinance): {real_put_price:.2f}")
    print("\nUser Inputs:")
    print(f"Strike Price (K): {K:.2f}")
    print(f"Time to Maturity (T): {T:.2f} years")
    print(f"Dividend Yield (q): {q:.4f}")

    # Ask the user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run a single pricing model")
    print("2. Compare all models (benchmark)")
    running_choice = int(input("Enter your choice (1 or 2): "))
    model_choice = 0

    if running_choice in [1, 2]:
        if running_choice == 1:
            model_choice = int(input("Enter your choice (1, 2, 3, 4, or 5): "))
        compare_methods(S, K, T, r, sigma, q, real_call_price, real_put_price, running_choice=running_choice, model_choice=model_choice)
    else:
        print("Invalid choice. Exiting.")