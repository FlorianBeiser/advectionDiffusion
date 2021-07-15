"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

class Kalman:
    def __init__(self, statistics, observation):
        self.statistics = statistics

        # Model error cov matrix
        self.epsilon = self.statistics.simulator.noise
        
        # Observation and obs error cov matrices
        self.H = observation.H
        self.tau = observation.noise

    def filter(self, forecasted_mean, forecasted_cov, obs):

        Q = np.matmul(self.H,np.matmul(forecasted_cov,self.H.transpose())) + self.tau
        K = np.matmul(forecasted_cov,np.matmul(self.H.transpose(), np.linalg.inv(Q)))

        updated_mean = forecasted_mean + np.matmul(K, (obs - np.matmul(self.H,forecasted_mean)))
        updated_covariance = forecasted_cov - np.matmul(K,np.matmul(Q,K.transpose()))
        
        self.statistics.set(updated_mean, updated_covariance)

        return updated_mean, updated_covariance


