import numpy as np

class HestonModel:
    def __init__(self, s0, v0, kappa, theta, sigma, rho, r, T, n_steps):
        """
        Initialize the Heston model parameters.

        Parameters:
        s0 (float): Initial asset price
        v0 (float): Initial volatility
        kappa (float): Mean-reversion speed
        theta (float): Long-term variance
        sigma (float): Volatility of volatility
        rho (float): Correlation between asset and volatility
        r (float): Risk-free interest rate
        T (float): Time to maturity (in years)
        n_steps (int): Number of time steps
        """
        self.s0 = s0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps

    def simulate_paths(self, n_paths):
        """
        Simulate asset price and volatility paths using the Heston model.

        Parameters:
        n_paths (int): Number of price and volatility paths to simulate

        Returns:
        numpy.ndarray, numpy.ndarray: Simulated asset prices and volatilities
        """
        s_paths = np.zeros((n_paths, self.n_steps + 1))
        v_paths = np.zeros((n_paths, self.n_steps + 1))

        s_paths[:, 0] = self.s0
        v_paths[:, 0] = self.v0

        for t in range(1, self.n_steps + 1):
            dw1 = np.random.normal(0, 1, n_paths)
            dw2 = np.random.normal(0, 1, n_paths)

            dv = self.kappa * (self.theta - np.maximum(v_paths[:, t - 1], 0)) * self.dt + self.sigma * np.sqrt(np.maximum(v_paths[:, t - 1], 0)) * np.sqrt(self.dt) * dw2
            v_paths[:, t] = np.maximum(v_paths[:, t - 1] + dv, 0)  # Ensure variance stays positive

            ds = np.sqrt(np.maximum(v_paths[:, t], 0)) * s_paths[:, t - 1] * np.sqrt(self.dt) * dw1
            s_paths[:, t] = s_paths[:, t - 1] * np.exp((self.r - 0.5 * v_paths[:, t]) * self.dt + ds)

        return s_paths, v_paths

    def calc_call_option_price(self, strike, maturity_idx, n_paths=10000):
        """
        Calculate the call option price using the simulated paths.

        Parameters:
        strike (float): Strike price of the call option
        maturity_idx (int): Index representing the maturity time step
        n_paths (int): Number of simulated paths

        Returns:
        float: Price of the call option
        """
        s_paths, v_paths = self.simulate_paths(n_paths)
        payoffs = np.maximum(s_paths[:, maturity_idx] - strike, 0)  # Call option payoff
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs  # Discount payoffs to present value
        call_price = np.mean(discounted_payoffs)
        return call_price

    def calc_put_option_price(self, strike, maturity_idx, n_paths=10000):
        """
        Calculate the put option price using the simulated paths.

        Parameters:
        strike (float): Strike price of the put option
        maturity_idx (int): Index representing the maturity time step
        n_paths (int): Number of simulated paths

        Returns:
        float: Price of the put option
        """
        s_paths, v_paths = self.simulate_paths(n_paths)
        payoffs = np.maximum(strike - s_paths[:, maturity_idx], 0)  # Put option payoff
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs  # Discount payoffs to present value
        put_price = np.mean(discounted_payoffs)
        return put_price
