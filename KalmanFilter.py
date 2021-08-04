"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

class Kalman:
    def __init__(self, statistics, observation):
        self.statistics = statistics

        # Observation and obs error cov matrices
        self.H = observation.H
        self.R = observation.R

    def filter(self, forecasted_mean, forecasted_cov, obs):

        S = self.H @ forecasted_cov @ self.H.T + self.R
        K = forecasted_cov @ self.H.T @ np.linalg.inv(S)

        updated_mean = forecasted_mean + K @ (obs - self.H @ forecasted_mean)
        updated_covariance = forecasted_cov - K @ S @ K.T
        
        self.statistics.set(updated_mean, updated_covariance)

        return updated_mean, updated_covariance