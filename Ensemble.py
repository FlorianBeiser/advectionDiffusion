  
import numpy as np

class Ensemble:
    def __init__(self, simulator, N_e):
        self.simulator = simulator
        self.N_e = N_e
        
        # Allocation 
        self.ensemble = np.zeros((self.simulator.grid.N_x, self.N_e))


    def initialize(self, prior_sampler):
        
        self.ensemble = prior_sampler.sample(self.N_e)


    def set(self, ensemble):
        self.ensemble = ensemble
