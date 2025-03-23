from settings import UNDERLYING_ASSET_TICKER
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_spot_price(ticker: str = UNDERLYING_ASSET_TICKER):
    """
    Retrieve the current spot price of the S&P 500 using yfinance.

    Args:
        ticker (str): Stock ticker in yfinance.
    """
    ticker_obj = yf.Ticker(ticker)
    spot_price = ticker_obj.history(period="1d")['Close'].iloc[-1]  # Get the latest closing price
    return spot_price

def get_implied_volatility(ticker: str = UNDERLYING_ASSET_TICKER):
    """
    Retrieve the implied volatility of the S&P 500 using yfinance.

    Args:
        ticker (str): Stock ticker in yfinance.
    """
    ticker_obj = yf.Ticker(ticker)
    options = ticker_obj.options  # Get available expiration dates
    if not options:
        raise ValueError("No options data available for the given ticker")

    # Get the nearest expiration date
    nearest_expiry = options[0]

    # Get options data for the nearest expiration date
    options_data = ticker_obj.option_chain(nearest_expiry)
    implied_volatility = options_data.calls['impliedVolatility'].mean()  # Average implied volatility
    return implied_volatility

def get_real_option_price(T, K, ticker: str = UNDERLYING_ASSET_TICKER):
    """
    Get call and put option prices for a given time period T (in years) and strike price K.
    
    Parameters:
        ticker: str = UNDERLYING_ASSET_TICKER.
        T (float): Time period in years (e.g., 0.25 for 3 months).
        K (float): Strike price (e.g., 5400).
    
    Returns:
        call_price (float): Last traded price of the call option.
        put_price (float): Last traded price of the put option.
        expiration_date (str): Closest expiration date to the target date.
    """
    # Create a Ticker object
    ticker_obj = yf.Ticker(ticker)

    # Get the current date
    today = datetime.today()

    # Calculate the target date (T years from today)
    target_date = today + timedelta(days=int(T * 365))

    # Fetch the available expiration dates
    expiration_dates = ticker_obj.options

    # Find the expiration date closest to the target date
    closest_expiration = min(expiration_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_date))

    print(f"Closest expiration date to {T} years: {closest_expiration}")

    # Fetch the option chain for the closest expiration date
    option_chain = ticker_obj.option_chain(closest_expiration)

    # Filter call and put options for the strike price K
    call_options = option_chain.calls[option_chain.calls['strike'] == K]
    put_options = option_chain.puts[option_chain.puts['strike'] == K]

    # Get the call and put option prices
    call_price = call_options['lastPrice'].values[0] if not call_options.empty else None
    put_price = put_options['lastPrice'].values[0] if not put_options.empty else None

    return call_price, put_price, closest_expiration

def create_volatility_surface(ticker: str = UNDERLYING_ASSET_TICKER):
    """
    Create a volatility surface for the given ticker using implied volatility data from yfinance.
    Includes both call and put options.

    Args:
        ticker (str): Stock ticker in yfinance.
    """
    # Get the spot price
    S = get_spot_price(ticker)

    # Get available expiration dates
    ticker_obj = yf.Ticker(ticker)
    expiration_dates = ticker_obj.options

    # Initialize arrays to store results
    implied_volatilities = []
    strikes = []
    times_to_maturity = []
    option_types = []  # To distinguish between call and put options

    # Loop through expiration dates
    for expiry in expiration_dates:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        T = (expiry_date - datetime.today()).days / 365.0  # Time to maturity in years

        # Get options data for the current expiration date
        options_chain = ticker_obj.option_chain(expiry)

        # Extract call options data
        calls = options_chain.calls
        for _, row in calls.iterrows():
            K = row['strike']  # Strike price
            iv = row['impliedVolatility']  # Implied volatility from yfinance

            # Store the data
            if not np.isnan(iv):
                implied_volatilities.append(iv)
                strikes.append(K)
                times_to_maturity.append(T)
                option_types.append('call')  # Mark as call option

        # Extract put options data
        puts = options_chain.puts
        for _, row in puts.iterrows():
            K = row['strike']  # Strike price
            iv = row['impliedVolatility']  # Implied volatility from yfinance

            # Store the data
            if not np.isnan(iv):
                implied_volatilities.append(iv)
                strikes.append(K)
                times_to_maturity.append(T)
                option_types.append('put')  # Mark as put option

    # Convert lists to numpy arrays
    strikes = np.array(strikes)
    times_to_maturity = np.array(times_to_maturity)
    implied_volatilities = np.array(implied_volatilities)
    option_types = np.array(option_types)

    # Create grid data for plotting surfaces
    strike_grid = np.linspace(min(strikes), max(strikes), num=100)
    time_grid = np.linspace(min(times_to_maturity), max(times_to_maturity), num=100)
    strike_mesh, time_mesh = np.meshgrid(strike_grid, time_grid)

    # Interpolate call and put volatilities on the grid
    call_mask = option_types == 'call'
    put_mask = option_types == 'put'

    call_vol_interpolated = griddata((strikes[call_mask], times_to_maturity[call_mask]), 
                                      implied_volatilities[call_mask], 
                                      (strike_mesh, time_mesh), method='cubic')

    put_vol_interpolated = griddata((strikes[put_mask], times_to_maturity[put_mask]), 
                                    implied_volatilities[put_mask], 
                                    (strike_mesh, time_mesh), method='cubic')

    # Plot the volatility surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot call options surface
    ax.plot_surface(strike_mesh, time_mesh, call_vol_interpolated, 
                    cmap='Blues', alpha=0.7, label='Call Options')

    # Plot put options surface
    ax.plot_surface(strike_mesh, time_mesh, put_vol_interpolated, 
                    cmap='Reds', alpha=0.7, label='Put Options')

    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f'Volatility Surface for {ticker} (Call and Put Options)')
    ax.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    create_volatility_surface(ticker=UNDERLYING_ASSET_TICKER)
    # create_volatility_surface(ticker="BABA")
