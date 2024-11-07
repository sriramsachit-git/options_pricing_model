import numpy as np
import pandas as pd
from unittest.mock import patch
from calibration import objective_function, calibrate_heston_model, load_and_calibrate_heston_model
from heston_model import HestonModel

class TestCalibration:
    def setup_method(self):
        self.market_data = pd.DataFrame({
            'strike': [90, 100, 110],
            'maturity': [0.5, 1.0, 1.5],
            'market_price': [5.0, 7.0, 10.0],
            'type': ['call', 'call', 'call']
        })

        self.initial_params = (1.5, 0.04, 0.3, -0.6)

    def test_objective_function(self):
        model = HestonModel(
            s0=100,
            v0=0.04,
            kappa=1.5,
            theta=0.04,
            sigma=0.3,
            rho=-0.6,
            r=0.03,
            T=1,
            n_steps=252
        )

        mse = objective_function(self.initial_params, self.market_data, model)
        assert isinstance(mse, float)
        assert mse >= 0

    @patch('calibration.calibrate_heston_model')
    def test_load_and_calibrate_heston_model(self, mock_calibrate_heston_model):
        mock_calibrate_heston_model.return_value = self.initial_params

        model = load_and_calibrate_heston_model('market_data.csv', self.initial_params)
        assert isinstance(model, HestonModel)
        assert model.kappa == self.initial_params[0]
        assert model.theta == self.initial_params[1]
        assert model.sigma == self.initial_params[2]
        assert model.rho == self.initial_params[3]

    def test_calibrate_heston_model(self):
        model = HestonModel(
            s0=100,
            v0=0.04,
            kappa=self.initial_params[0],
            theta=self.initial_params[1],
            sigma=self.initial_params[2],
            rho=self.initial_params[3],
            r=0.03,
            T=1,
            n_steps=252
        )

        kappa, theta, sigma, rho = calibrate_heston_model(self.market_data, self.initial_params, model)
        assert isinstance(kappa, float)
        assert isinstance(theta, float)
        assert isinstance(sigma, float)
        assert isinstance(rho, float)
        assert kappa > 0
        assert theta > 0
        assert sigma > 0
        assert -1 < rho < 1