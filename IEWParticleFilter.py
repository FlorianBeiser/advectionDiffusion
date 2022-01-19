"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

from scipy.special import gammainc
from scipy.special import lambertw
from scipy.optimize import fsolve
from scipy.linalg import sqrtm 

import sys

class IEWParticle:
    def __init__(self, statistics, observation, beta=None, alpha=None):
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

        self.beta = beta
        self.betas = []

        # Only for debug purposes
        self.alpha = alpha


    def filter(self, ensemble, obs):

        # Innovations and weights
        d = np.reshape(obs, (self.N_y,1)) - self.H @ ensemble

        phis = np.zeros(self.N_e)
        etas = np.zeros((self.N_e, self.N_x))
        for e in range(self.N_e):
            phis[e] = d[:,e] @ self.S @ d[:,e]
            etas[e] = np.random.standard_normal(self.N_x)

        cs = phis - np.log(1/self.N_e)
        c_bar = np.average(cs)

        if self.beta is None:
            # Target weight
            tmp = np.zeros(self.N_e)
            for e in range(self.N_e):
                tmp[e] = (c_bar - cs[e])/(etas[e]@etas[e]) + 1

            beta = np.min(tmp)
        
        else:
            beta = self.beta

        self.betas.append(beta)

        # Get c_star
        c_stars = np.zeros(self.N_e)
        for e in range(self.N_e):
            c_stars[e] = c_bar - cs[e] - (beta-1)*(etas[e]@etas[e])
            #c_stars[e] = np.max(cs) - cs[e] - (beta-1)*(etas[e]@etas[e])

        updated_ensemble = np.zeros_like(ensemble)
        # Per ensemble member!
        for e in range(self.N_e):
            
            # Sampling random vectors
            z = np.random.standard_normal(self.N_x)
            xi = z - etas[e] * (z@etas[e])/(etas[e]@etas[e])

            if self.alpha is None:
                # Compute alpha
                # fun = lambda alpha, m, x: gammainc(m, alpha*x)/gammainc(m, x)
                # alpha = fsolve( lambda alpha: fun(alpha, self.N_x/2, etas[e]@etas[e]/2) - np.exp(-c_stars[e]/2), 0.5)

                alpha = np.real( -self.N_x/(etas[e]@etas[e]) * lambertw(-(etas[e]@etas[e])/self.N_x * np.exp(-(etas[e]@etas[e])/self.N_x) * np.exp(-c_stars[e]/self.N_x) ) )
            else:
                alpha = self.alpha

            # Update ensemble member
            member_proposal = ensemble[:,e] + self.Q @ self.H.T @ self.S @ d[:,e]
            member_update = member_proposal + np.sqrt(beta) * self.sqrtPinv @ etas[e] + np.sqrt(alpha) * self.sqrtPinv @ xi
            updated_ensemble[:,e] = member_update

        self.statistics.set_ensemble(updated_ensemble)

        return updated_ensemble
    