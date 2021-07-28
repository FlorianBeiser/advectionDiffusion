  
from Simulator import Simulator
import numpy as np

class Ensemble:
    def __init__(self, simulator, N_e):
        self.simulator = simulator
        self.N_e = N_e

    def initialize(self, mean, cov, var_mesh):
        # NOTE: For periodic boundary conditions the covariance matrix 
        # becomes numerical problems with the semi-positive definiteness.
        # To avoid negative Eigenvalues a small nugget on the diagonal is added.
        nugget = 0.001

        self.ensemble = np.random.multivariate_normal(mean, cov + nugget*np.eye(self.simulator.grid.N_x), self.N_e).transpose()

    def initializeFFT(self, mean, cov, var_mesh=None):
        if var_mesh is None:
            var_mesh = np.ones((self.simulator.grid.ny, self.simulator.grid.nx))
            
        self.ensemble = np.zeros((self.simulator.grid.N_x, self.N_e))

        # Sampling Gaussian random fields using the FFT
        # What is utilizing the Toepitz structure of the covariance matrix.
        # In the end, it is transformed with the mean and point variances
        # NOTE: For periodic boundary conditions the covariance matrix 
        # becomes numerical problems with the semi-positive definiteness
        # what forbids to use classical np.random.multivariate_normal sampling
        # but the FFT approach for Toepitz matrixes circumvents those problems
        cov_toepitz = np.reshape(cov[0,:], (self.simulator.grid.ny, self.simulator.grid.nx))
        cmf = np.real(np.fft.fft2(cov_toepitz))

        for e in range(self.N_e):
            u = np.random.normal(size=(self.simulator.grid.ny, self.simulator.grid.nx))
            uif = np.fft.ifft2(u)

            xf = np.real(np.fft.fft2(cmf*uif))

            self.ensemble[:,e] = mean + np.reshape(var_mesh, self.simulator.grid.N_x)*np.reshape(xf, self.simulator.grid.N_x)

    def set(self, ensemble):
        self.ensemble = ensemble