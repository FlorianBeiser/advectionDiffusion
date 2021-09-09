"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

from scipy.special import gammainc
from scipy.optimize import fsolve
from scipy.linalg import sqrtm 

class IEWParticle:
    def __init__(self, statistics, observation):
        self.statistics = statistics

        self.N_e = self.statistics.ensemble.N_e

        # From simulator
        self.N_x = self.statistics.simulator.grid.N_x
        self.Q = self.statistics.simulator.Q
        
        # From observation
        self.N_y = observation.N_y
        self.H = observation.H
        
        Pinv = np.linalg.inv( np.linalg.inv(self.Q) + self.H.T @ np.linalg.inv(observation.R) @ self.H )
        self.sqrtPinv = np.real(sqrtm(Pinv))

        self.S = np.linalg.inv( self.H @ self.Q @ self.H.T + observation.R)


    def filter(self, ensemble, obs):

        # Innovations and weights
        d = np.reshape(obs, (self.N_y,1)) - self.H @ ensemble

        phis = np.zeros(self.N_e)
        for e in range(self.N_e):
            phis[e] = d[:,e] @ self.S @ d[:,e]
        cs = np.max(phis) - phis

        updated_ensemble = np.zeros_like(ensemble)
        # Per ensemble member!
        for e in range(self.N_e):
            # Sampling random vectors
            eta = np.random.standard_normal(self.N_x)
            z = np.random.standard_normal(self.N_x)
            xi = z - eta * (z@eta)/(eta@eta)

            # Compute alpha
            fun = lambda alpha, m, x: gammainc(m, alpha*x)/gammainc(m, x)
            alpha = fsolve( lambda alpha: fun(alpha, self.N_x/2, eta@eta/2) - np.exp(-cs[e]/2), 0.5)
            
            # Update ensemble member
            member_proposal = ensemble[:,e] + self.Q @ self.H.T @ self.S @ d[:,e]
            member_update = member_proposal + 1 * self.sqrtPinv @ eta + np.sqrt(alpha) * self.sqrtPinv @ xi
            updated_ensemble[:,e] = member_update

        self.statistics.set_ensemble(updated_ensemble)

        return updated_ensemble
    