import numpy as np
from scipy.optimize import minimize
from hestonModel import HestonModel

def objective_function(params, market_data, model):
    """
    Objective function for Heston model calibration.
    Calculates the mean squared error between model-implied and market option prices.

    Parameters:
    params (tuple): Heston model parameters (kappa, theta, sigma, rho)
    market_data (pandas.DataFrame): Market option data (strike, maturity, market_price, type)
    model (HestonModel): Instance of the HestonModel class

    Returns:
    float: Mean squared error between model-implied and market option prices
    """
    kappa, theta, sigma, rho = params
    model.kappa = kappa
    model.theta = theta
    model.sigma = sigma
    model.rho = rho

    total_mse = 0
    for _, row in market_data.iterrows():
        strike = row['strike']
        maturity_idx = int(row['maturity'] * model.n_steps / model.T)
        market_price = row['market_price']

        # Calculate model price based on option type
        if row['type'] == 'call':
            model_price = model.calc_call_option_price(strike, maturity_idx)
        else:
            model_price = model.calc_put_option_price(strike, maturity_idx)

        # Check for NaN or Inf values
        if np.isnan(model_price) or np.isinf(model_price):
            print(f"Invalid model price: {model_price} for params: {params}")
            return np.inf  # Return a large value to ignore these parameters

        mse = (model_price - market_price) ** 2
        total_mse += mse

    return total_mse / len(market_data)

def calibrate_heston_model(market_data, initial_params, model):
    """
    Calibrate the Heston model parameters using market option data.

    Parameters:
    market_data (pandas.DataFrame): Market option data (strike, maturity, market_price, type)
    initial_params (tuple): Initial values for the Heston model parameters (kappa, theta, sigma, rho)
    model (HestonModel): Instance of the HestonModel class

    Returns:
    tuple: Optimized Heston model parameters (kappa, theta, sigma, rho)
    """
    bounds = [(0, None), (0, None), (0, None), (-1, 1)]  # Bounds for kappa, theta, sigma, rho

    result = minimize(
        objective_function,
        initial_params,
        args=(market_data, model),
        method='L-BFGS-B',
        bounds=bounds
    )

    if result.success:
        print("Optimization successful.")
    else:
        print(f"Optimization failed: {result.message}")

    return result.x  # Optimized parameters

# Testing the calibration function
if __name__ == "__main__":
    import pandas as pd

    # Example market data (replace with actual data)
    market_data = pd.DataFrame({
        'strike': [100, 105, 110],
        'maturity': [0.5, 1.0, 1.5],
        'market_price': [10, 12, 15],
        'type': ['call', 'put', 'call']
    })

    # Initial parameters (kappa, theta, sigma, rho)
    initial_params = (1.0, 0.2, 0.3, -0.5)

    # Create HestonModel instance with placeholder parameters
    heston_model = HestonModel(s0=100, v0=0.04, kappa=1.0, theta=0.2, sigma=0.3, rho=-0.5, r=0.05, T=1.0, n_steps=100)

    # Run calibration
    optimized_params = calibrate_heston_model(market_data, initial_params, heston_model)
    print("Optimized parameters:", optimized_params)
