"""
Mean and Variance for the advection diffusion example
"""

import Ensemble

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class Statistics:
    def __init__(self, simulator, ensemble=0):
        """Class for handling the mean and cov throughout times"""
        self.simulator = simulator
        print("Please remember to set priors!")
        self.mean = np.zeros([self.simulator.grid.N_x])
        self.var  = np.zeros([self.simulator.grid.N_x])
        self.cov  = np.zeros([self.simulator.grid.N_x,self.simulator.grid.N_x])
        # Default is analytical 
        if ensemble > 0:
            self.ensemble = Ensemble.Ensemble(simulator, ensemble)
        else:
            self.ensemble = None


    def set(self, mean, cov):
        """Setting the member variables from input arguments"""
        self.mean = mean
        self.var = np.diag(cov)
        self.cov = cov
        if self.ensemble is not None:
            self.ensemble.initialize(mean, cov)


    def plot(self):
        """Plotting mean, var, and cov in a unified graphics"""
        fig, axs = plt.subplots(1,3, figsize=(12,4))

        mean = np.reshape(self.mean, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig0 = axs[0].imshow(mean, origin = "lower", vmin=0.0, vmax=0.5)
        axs[0].set_title("Mean")
        ax_divider = make_axes_locatable(axs[0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig0, cax=ax_cb, orientation="horizontal")

        var = np.reshape(self.var, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig1 = axs[1].imshow(var, origin = "lower", vmin=0.0, vmax=0.5)
        axs[1].set_title("Variance")
        ax_divider = make_axes_locatable(axs[1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig1, cax=ax_cb, orientation="horizontal")

        fig2 = axs[2].imshow(self.cov, vmin=0.0, vmax=0.25)
        axs[2].set_title("Covariance Matrix")
        ax_divider = make_axes_locatable(axs[2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig2, cax=ax_cb, orientation="horizontal")


    def propagate(self, nt):
        """Propagating the model for nt simulator time steps.
        NOTE: nt simulator steps are 1 model time step 
        wherefore a distinged (DA) model matrix is constructed"""
        self.M = np.eye(self.simulator.grid.N_x)
        for t in range(nt):
            self.M = np.matmul(self.simulator.M, self.M)

        if self.ensemble is None:
            self.mean = np.matmul(self.M, self.mean)
            self.cov = np.matmul(self.M, np.matmul(self.cov, self.M.transpose())) + self.simulator.noise
        else:
            self.ensemble.set( np.matmul(self.M, self.ensemble.ensemble))
            self.mean = np.average(self.ensemble.ensemble, axis = 1)
