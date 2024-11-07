import pandas as pd
from dataLoader import load_yahoo_finance_data
from calibration import load_and_calibrate_heston_model
from pricing import heston_call_option_price, validate_heston_pricing
import numpy as np

def main():
    # Load and preprocess historical data
    tickers = ['AAPL', 'MSFT', 'AMZN']
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    historical_data = load_yahoo_finance_data(tickers, start_date, end_date)
    historical_data.to_csv('data/processed_data.csv', index=False)

    # Load market option data and calibrate the Heston model
    market_data_file = 'data/market_option_data.csv'
    initial_params = (1.5, 0.04, 0.3, -0.6)
    calibrated_model = load_and_calibrate_heston_model(market_data_file, initial_params)

    # Calculate a Heston model-implied call option price
    call_price = heston_call_option_price(
        s0=100,
        k=105,
        t=1,
        r=0.03,
        v0=calibrated_model.v0,
        kappa=calibrated_model.kappa,
        theta=calibrated_model.theta,
        sigma=calibrated_model.sigma,
        rho=calibrated_model.rho
    )
    print(f"Heston model call option price: {call_price:.2f}")

    # Validate the Heston model pricing against market data
    market_data = pd.read_csv(market_data_file)
    pricing_comparison = validate_heston_pricing(market_data, calibrated_model)
    print(pricing_comparison)

   



def plot_volatility_surface(calibrated_model, market_data):
    """
    Plot the volatility surface using strike price and maturity.
    """
    # Extract unique strike prices and maturities from the market data
    strikes = market_data['strike'].unique()
    maturities = market_data['maturity'].unique()

    # Create a meshgrid for strikes and maturities
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)

    # Calculate implied volatilities for each combination of strike and maturity
    vol_surface = np.zeros_like(strike_grid)

    for i in range(len(maturities)):
        for j in range(len(strikes)):
            strike = strikes[j]
            maturity = maturities[i]

            # Use the calibrated Heston model parameters to calculate volatility at each (strike, maturity)
            vol_surface[i, j] = calibrated_model.calc_implied_volatility(strike, maturity)

    
if __name__ == "__main__":
    main()
