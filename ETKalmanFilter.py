"""
Kalman filter update for advection diffusion example.
"""

from Simulator import Simulator
from Statistics import Statistics
import numpy as np

class ETKalman:
    def __init__(self, statistics, observation):
        self.statistics = statistics

        # Model error cov matrix
        self.epsilon = self.statistics.simulator.noise
        
        # Observation and obs error cov matrices
        self.H = observation.H
        self.tau = observation.noise

    def filter(self, ensemble, obs, series=None):

        X_f_mean = np.average(ensemble, axis=1)
        X_f_pert = ensemble - np.reshape(X_f_mean, (self.statistics.simulator.grid.N_x,1))

        Rinv = np.linalg.inv(self.tau)

        HX_f =  self.H @ ensemble
        HX_f_mean = np.average(HX_f, axis=1)
        HX_f_pert = HX_f - np.reshape(HX_f_mean, (len(obs),1))

        D = obs - HX_f_mean

        A1 = (self.statistics.ensemble.N_e-1)*np.eye(self.statistics.ensemble.N_e)
        A2 = np.dot(HX_f_pert.T, np.dot(Rinv, HX_f_pert))
        A = A1 + A2
        P = np.linalg.inv(A)

        K = np.dot(X_f_pert, np.dot(P, np.dot(HX_f_pert.T, Rinv)))

        X_a_mean = X_f_mean + np.dot(K, D)
        sigma, V = np.linalg.eigh( (self.statistics.ensemble.N_e - 1) * P )

        X_a_pert = np.dot( X_f_pert, np.dot( V, np.dot( np.diag( np.sqrt( np.real(sigma) ) ), V.T )))

        X_a = X_a_pert + np.reshape(X_a_mean, (self.statistics.simulator.grid.N_x,1))

        self.statistics.set_ensemble(X_a)

        return X_a