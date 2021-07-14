"""
Mean and Variance for the advection diffusion example
"""

from Simulator import Simulator
import numpy as np
from random import random

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class Statistics:
    def __init__(self, simulator, ensemble_flag=False):
        self.simulator = simulator
        print("Please remember to set priors!")
        self.mean = np.zeros([self.simulator.grid.N_x])
        self.var  = np.zeros([self.simulator.grid.N_x])
        self.cov  = np.zeros([self.simulator.grid.N_x,self.simulator.grid.N_x])


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
        plt.colorbar(fig0, cax=ax_cb, orientation="horizontal")

        var = np.reshape(self.var, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig1 = axs[1].imshow(var, origin = "lower")
        axs[1].set_title("Variance")
        ax_divider = make_axes_locatable(axs[1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig1, cax=ax_cb, orientation="horizontal")

        fig2 = axs[2].imshow(self.cov)
        axs[2].set_title("Covariance Matrix")
        ax_divider = make_axes_locatable(axs[2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig2, cax=ax_cb, orientation="horizontal")


    def propagate(self, nt):
        for t in range(nt):
            self.mean, self.cov = self.simulator.forecast(self.mean, self.cov)

