import numpy as np

class Ensemble:
    def __init__(self, simulator, N_e):
        self.simulator = simulator
        self.N_e = N_e

    def initialize(self, mean, cov):
        self.ensemble = np.random.multivariate_normal(mean, cov, self.N_e).transpose()

    def set(self, ensemble):
        self.ensemble = ensemble