"""
Mean and Variance for the advection diffusion example
(eventually in ensemble representation)
"""

import Ensemble
import Sampler

import numpy as np
import linecache

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class Statistics:
    def __init__(self, simulator, N_e=0):
        """Class for handling the mean and cov throughout times"""
        self.simulator = simulator
        
        # Allocation
        self.mean   = np.zeros([self.simulator.grid.N_x])
        self.stddev = np.zeros([self.simulator.grid.N_x])
        self.cov    = np.zeros([self.simulator.grid.N_x,self.simulator.grid.N_x])
        
        # Default is analytical 
        if N_e > 0:
            self.ensemble = Ensemble.Ensemble(simulator, N_e)
        else:
            self.ensemble = None
            print("Please remember to set priors!")


    def ensemble_statistics(self):
        self.mean = np.average(self.ensemble.ensemble, axis = 1)
        if self.ensemble.N_e > 1: 
            self.cov = 1/(self.ensemble.N_e-1)*\
                (self.ensemble.ensemble - np.reshape(self.mean, (self.simulator.grid.N_x,1))) \
                @ (self.ensemble.ensemble - np.reshape(self.mean, (self.simulator.grid.N_x,1))).transpose()
            self.stddev = np.sqrt(np.diag(self.cov))
        

    def set(self, mean, cov):
        """Setting the member variables from input arguments"""
        self.mean = mean
        self.stddev = np.sqrt(np.diag(cov))
        self.cov = cov


    def set_prior(self, prior_args):
        
        prior_sampler = Sampler.Sampler(self.simulator.grid, prior_args)
            
        if self.ensemble is not None:
            self.ensemble.initialize(prior_sampler)
            self.ensemble_statistics()
        else:
            self.mean = prior_sampler.mean
            self.cov  = prior_sampler.cov
            self.stddev  = np.sqrt(np.diag(self.cov))

        self.vmin_mean = np.min(prior_sampler.mean) - 0.5
        self.vmax_mean = np.max(prior_sampler.mean) + 0.5 

        self.vmax_cov = np.max(self.cov)


    def set_ensemble(self, ensemble):
        self.ensemble.set(ensemble)
        self.ensemble_statistics()


    def plot(self, mean=None, stddev=None, cov=None):
        """Plotting mean, stddev, and cov in a unified graphics"""
        fig, axs = plt.subplots(1,3, figsize=(12,4))

        if mean is None:
            mean = np.reshape(self.mean, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig0 = axs[0].imshow(mean, origin = "lower", vmin=self.vmin_mean, vmax=self.vmax_mean)
        axs[0].set_title("Mean")
        ax_divider = make_axes_locatable(axs[0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig0, cax=ax_cb, orientation="horizontal")

        if stddev is None:
            stddev = np.reshape(self.stddev, (self.simulator.grid.ny,self.simulator.grid.nx))
        fig1 = axs[1].imshow(stddev, origin = "lower", vmin=0.0, vmax=self.vmax_cov)
        axs[1].set_title("Standard Deviation")
        ax_divider = make_axes_locatable(axs[1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig1, cax=ax_cb, orientation="horizontal")

        if cov is None:
            cov = self.cov
        fig2 = axs[2].imshow(cov, vmin=0.0, vmax=self.vmax_cov)
        axs[2].set_title("Covariance Matrix")
        ax_divider = make_axes_locatable(axs[2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig2, cax=ax_cb, orientation="horizontal")

        plt.show()


    def propagate(self, nt):
        """Propagating the model for nt simulator time steps.
        NOTE: nt simulator steps are 1 model time step 
        wherefore a distinged (DA) model matrix is constructed"""
        # Construct forward step matrix (by multiple steps from simulator matrix)
        self.M = np.linalg.matrix_power(self.simulator.M, nt)

        # Propagate 
        # - with model error for ensembles
        # - without model error for analytical distributions
        if self.ensemble is None:
            self.mean = self.M @ self.mean
            self.cov = self.M @ self.cov @ self.M.T + self.simulator.Q
        else:
            forecast = self.M @ self.ensemble.ensemble + self.simulator.noise_sampler.sample(self.ensemble.N_e)
            self.ensemble.set(forecast)
            self.ensemble_statistics()


def prior_args_from_file(timestamp):
    f = "experiment_files/experiment_"+timestamp+"/setup"
    mean_upshift = float(linecache.getline(f, 23)[15:-1])
    bell_center = linecache.getline(f, 24)[14:-1].strip('][').split(', ')
    bell_center[0] = float(bell_center[0])
    bell_center[1] = float(bell_center[1])
    bell_sharpness = float(linecache.getline(f, 25)[17:-1])
    bell_scaling = float(linecache.getline(f, 26)[15:-1])
    matern_phi = float(linecache.getline(f, 27)[13:-1])
    stddev = float(linecache.getline(f, 28)[8:-1])

    prior_args = {"mean_upshift" : mean_upshift,
                "bell_center"    : bell_center,
                "bell_sharpness" : bell_sharpness,
                "bell_scaling"   : bell_scaling,
                "matern_phi"     : matern_phi , 
                "stddev"         : stddev}

    return prior_args