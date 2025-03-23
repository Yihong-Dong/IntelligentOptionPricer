from settings import RISK_FREE_ASSET_TICKER

import yfinance as yf
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class RiskFreeCurve():

    def __init__(self):
        return
    
    @staticmethod
    def get_risk_free_rate(ticker: str = RISK_FREE_ASSET_TICKER):
        """
        Retrieve the risk-free rate using the yield on the 10-year US Treasury Bond.
        
        Args:
            ticker (str): The Yahoo Finance ticker symbol for the Treasury bond.
        
        Returns:
            float: The risk-free rate as a decimal.
        """
        treasury = yf.Ticker(ticker)
        # Fetch the latest closing price of the bond, convert it to a decimal by dividing by 100
        risk_free_rate = treasury.history(period="1d")['Close'].iloc[-1] / 100  
        return risk_free_rate

    @staticmethod
    def get_treasury_yields():
        """
        Retrieve yields for US Treasury Bonds of different maturities using yfinance.
        
        Returns:
        tuple: A tuple containing two lists - one for maturities in years and another for corresponding yields as decimals.
        """
        # Dictionary mapping maturity periods to their respective Yahoo Finance ticker symbols
        tickers = {
            "1-month": "^IRX",
            "3-month": "^IRX",
            "6-month": "^IRX",
            "1-year": "^IRX",
            "2-year": "^FVX",
            "5-year": "^FVX",
            "10-year": "^TNX",
            "30-year": "^TYX"
        }

        maturities = []
        yields = []
        for maturity, ticker in tickers.items():
            bond = yf.Ticker(ticker)
            # Fetch the latest closing price of the bond, convert it to a decimal by dividing by 100
            yield_value = bond.history(period="1d")['Close'].iloc[-1] / 100  
            # Extract the numeric part of the maturity string and convert it to a float
            if '-' in maturity:
                num_part = maturity.split('-')[0]
            else:
                num_part = maturity
            if 'month' in maturity:
                maturities.append(float(num_part) / 12)  # Convert months to years
            else:
                maturities.append(float(num_part))
            yields.append(yield_value)

        # Sort maturities and yields together based on maturities
        sorted_indices = sorted(range(len(maturities)), key=lambda k: maturities[k])
        maturities = [maturities[i] for i in sorted_indices]
        yields = [yields[i] for i in sorted_indices]

        return maturities, yields

    @staticmethod
    def interpolate_yield_curve(maturities, yields):
        """
        Interpolate the yield curve using cubic spline interpolation.
        """
        # Create a cubic spline interpolator based on the provided maturities and yields
        yield_curve = CubicSpline(maturities, yields)
        return yield_curve

    @staticmethod
    def paint_treasury_yield_curve():
        # Fetch the Treasury yields
        maturities, yields = __class__.get_treasury_yields()

        # Interpolate the yield curve
        yield_curve = __class__.interpolate_yield_curve(maturities, yields)

        # Generate x values for plotting the interpolated curve
        x_vals = [i/12 for i in range(1, 361)]  # Monthly points from 1 month to 30 years

        # Plot the original data points and the interpolated curve
        plt.figure(figsize=(10, 6))
        plt.plot(maturities, yields, 'o', label='Original Data Points')
        plt.plot(x_vals, yield_curve(x_vals), label='Interpolated Yield Curve')
        plt.title('US Treasury Yield Curve')
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Yield (%)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    RiskFreeCurve.paint_treasury_yield_curve()
