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

    def filter(self, forecasted_mean, forecasted_cov, obs, series=None):

        if series is not None:
            H = self.H[series,:]

            Q = np.matmul(H,np.matmul(forecasted_cov,H.transpose())) + self.R[series,series]
            K = np.matmul(forecasted_cov,H.transpose()) / Q

            updated_mean = forecasted_mean + K * (obs - np.matmul(H,forecasted_mean))
            updated_covariance = forecasted_cov -  Q * np.outer(K, K)

        else:
            Q = np.matmul(self.H,np.matmul(forecasted_cov,self.H.transpose())) + self.R
            K = np.matmul(forecasted_cov,np.matmul(self.H.transpose(), np.linalg.inv(Q)))

            updated_mean = forecasted_mean + np.matmul(K, (obs - np.matmul(self.H,forecasted_mean)))
            updated_covariance = forecasted_cov - np.matmul(K,np.matmul(Q,K.transpose()))
        
        self.statistics.set(updated_mean, updated_covariance)

        return updated_mean, updated_covariance