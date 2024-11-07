import numpy as np

def heston_call_option_price(s0, k, t, r, v0, kappa, theta, sigma, rho, n_paths=10000, n_steps=100):
    """
    Calculate the call option price using the Heston model via Monte Carlo simulation.

    Parameters:
    s0 (float): Initial asset price
    k (float): Strike price
    t (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    v0 (float): Initial volatility
    kappa (float): Mean-reversion speed
    theta (float): Long-term variance
    sigma (float): Volatility of volatility
    rho (float): Correlation between asset and volatility
    n_paths (int): Number of paths to simulate
    n_steps (int): Number of time steps

    Returns:
    float: Heston model call option price
    """
    dt = t / n_steps
    s_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))

    s_paths[:, 0] = s0
    v_paths[:, 0] = v0

    for t_idx in range(1, n_steps + 1):
        dw1 = np.random.normal(0, 1, n_paths)  # Brownian motion for asset price
        dw2 = np.random.normal(0, 1, n_paths)  # Brownian motion for volatility
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * dw2  # Correlated Brownian motions

        # Simulate volatility using mean reversion
        v_paths[:, t_idx] = np.maximum(
            v_paths[:, t_idx - 1] + kappa * (theta - v_paths[:, t_idx - 1]) * dt + sigma * np.sqrt(dt) * dw2,
            0.0
        )

        # Simulate asset price using geometric Brownian motion with stochastic volatility
        s_paths[:, t_idx] = s_paths[:, t_idx - 1] * np.exp(
            (r - 0.5 * v_paths[:, t_idx - 1]) * dt + np.sqrt(v_paths[:, t_idx - 1]) * np.sqrt(dt) * dw1
        )

    # Calculate payoff at maturity for call option
    payoffs = np.maximum(s_paths[:, -1] - k, 0)

    # Discount payoffs to present value
    call_price = np.exp(-r * t) * np.mean(payoffs)

    return call_price

def heston_put_option_price(s0, k, t, r, v0, kappa, theta, sigma, rho, n_paths=10000, n_steps=100):
    """
    Calculate the put option price using the Heston model via Monte Carlo simulation.

    Parameters:
    s0 (float): Initial asset price
    k (float): Strike price
    t (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    v0 (float): Initial volatility
    kappa (float): Mean-reversion speed
    theta (float): Long-term variance
    sigma (float): Volatility of volatility
    rho (float): Correlation between asset and volatility
    n_paths (int): Number of paths to simulate
    n_steps (int): Number of time steps

    Returns:
    float: Heston model put option price
    """
    dt = t / n_steps
    s_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))

    s_paths[:, 0] = s0
    v_paths[:, 0] = v0

    for t_idx in range(1, n_steps + 1):
        dw1 = np.random.normal(0, 1, n_paths)  # Brownian motion for asset price
        dw2 = np.random.normal(0, 1, n_paths)  # Brownian motion for volatility
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * dw2  # Correlated Brownian motions

        # Simulate volatility using mean reversion
        v_paths[:, t_idx] = np.maximum(
            v_paths[:, t_idx - 1] + kappa * (theta - v_paths[:, t_idx - 1]) * dt + sigma * np.sqrt(dt) * dw2,
            0.0
        )

        # Simulate asset price using geometric Brownian motion with stochastic volatility
        s_paths[:, t_idx] = s_paths[:, t_idx - 1] * np.exp(
            (r - 0.5 * v_paths[:, t_idx - 1]) * dt + np.sqrt(v_paths[:, t_idx - 1]) * np.sqrt(dt) * dw1
        )

    # Calculate payoff at maturity for put option
    payoffs = np.maximum(k - s_paths[:, -1], 0)

    # Discount payoffs to present value
    put_price = np.exp(-r * t) * np.mean(payoffs)

    return put_price
