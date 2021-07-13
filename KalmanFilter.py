"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

class Kalman:
    def __init__(self, simulator, observation, statistics):
        self.statistics = statistics
        self.mean = self.statistics.mean
        self.cov = self.statistics.cov

        self.A = simulator.A
        self.epsilon = simulator.noise
        
        self.G = observation.G
        self.tau = observation.noise

    def filter(self, mean, cov, obs):
        R = np.matmul(self.A,np.matmul(cov, self.A)) + self.epsilon
        Q = np.matmul(self.G,np.matmul(R,self.G.transpose())) + self.tau
        K = np.matmul(R,np.matmul(self.G.transpose(), np.linalg.inv(Q)))

        updated_mean = np.matmul(self.A,mean) + np.matmul(K, (obs - np.matmul(self.G,np.matmul(self.A,mean))))
        updated_covariance = R - np.matmul(K,np.matmul(Q,K.transpose()))
        
        self.statistics.set(updated_mean, updated_covariance)

        return updated_mean, updated_covariance
        
        #FIXME: When adding the noise...
        # - adding a diagonal matrix 
        # OR 
        # - adding a fixed value
        # ???
        


