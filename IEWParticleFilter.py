"""
Kalman filter update for advection diffusion example.
"""

import numpy as np

class IEWParticle:
    def __init__(self, statistics, observation):
        self.statistics = statistics
        
        # Observation and obs error cov matrices
        self.H = observation.H
        self.R = observation.R


    def filter(self, ensemble, obs):

        print("dummy only!")
    