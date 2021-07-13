"""
Mean and Variance for the advection diffusion example
"""

from Simulator import Simulator
import numpy as np
from random import random

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from numpy.core.fromnumeric import var

class Statistics:
    def __init__(self, simulator, phi, ensemble_flag=False):
        self.simulator = simulator
        self.mean, self.var, self.cov = self.prior(phi)

    def prior(self, phi):
        """
        Constructing a Matern-type covariance prior with Matern-parameter phi
        """
        mean = np.arange(3,9,6/self.simulator.grid.N_x)
        var  = np.arange(4,7,3/self.simulator.grid.N_x) + random()

        cov = np.copy(self.simulator.grid.H)
        R = (1+phi*cov)*np.exp(-phi*cov) 
        for i in range(self.simulator.grid.N_x):
            for j in range(self.simulator.grid.N_x):
                    cov[i][j] = np.sqrt(var[i])*np.sqrt(var[j])*R[i][j]

        return mean, var, cov


    def set(self, mean, cov):
        self.mean = mean
        self.var = np.diag(cov)
        self.cov = cov


    def plot(self):
        fig, axs = plt.subplots(1,3, figsize=(12,4))

        mean = np.reshape(self.mean, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig0 = axs[0].imshow(mean, origin = "lower")
        axs[0].set_title("Mean")
        ax_divider = make_axes_locatable(axs[0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        colorbar(fig0, cax=ax_cb, orientation="horizontal")

        var = np.reshape(self.var, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig1 = axs[1].imshow(var, origin = "lower")
        axs[1].set_title("Variance")
        ax_divider = make_axes_locatable(axs[1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        colorbar(fig1, cax=ax_cb, orientation="horizontal")

        fig2 = axs[2].imshow(self.cov)
        axs[2].set_title("Covariance Matrix")
        ax_divider = make_axes_locatable(axs[2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        colorbar(fig2, cax=ax_cb, orientation="horizontal")