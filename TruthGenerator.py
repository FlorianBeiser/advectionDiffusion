import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import Sampler

def plot_kernel(grid, prior_args):
    """Plotting 1D Matern kernel for given phi on the half x domain of the grid"""
    matern_phi = prior_args["matern_phi"]

    # Plot kernel
    h = np.arange(int(max(grid.nx,grid.ny)/2))
    matern_kernel = (1+matern_phi*grid.dx*h)*np.exp(-matern_phi*grid.dx*h)
    plt.title("Matern covariance kernel")
    plt.plot(matern_kernel)
    plt.xlabel("Distance in grid cells")
    plt.ylabel("Correlation")

    # Plot threshold
    plt.plot(0.05*np.ones_like(h), "g")

    # Plot desired cutting area
    corr_min = int(max(grid.nx,grid.ny)/4)
    corr_max = int(max(grid.nx,grid.ny)/3)
    plt.fill_between(range(len(h)), 0, 0.1, where=(h >= corr_min) & (h <= corr_max), color="r", alpha=0.5)

    plt.legend(["cov kernel", "correlation range", "desired cut"])
    plt.show()


def plot_corr_radius(grid, prior_args):
    prior_sampler = Sampler.Sampler(grid, prior_args)
    corr = prior_sampler.corr
    corr0 = corr[0]
    plt.imshow(np.reshape((corr0>0.05), (grid.ny, grid.nx)), origin="lower")
    plt.title("Correlation area for (0,0)")
    plt.show()



def plot_xlims(statistics, prior_args):
    prior_sampler = Sampler.Sampler(statistics.simulator.grid, prior_args)
    vmin = np.min(prior_sampler.mean) - 0.5
    vmax = np.max(prior_sampler.mean) + 0.5 
    return vmin, vmax 


def plot_truth(state, grid, vmin=None, vmax=None):
     mean = np.reshape(state, (grid.ny,grid.nx))

     plt.imshow(mean, origin = "lower", vmin=vmin, vmax=vmax)
     plt.title("Truth")
     plt.colorbar(orientation="horizontal")
     plt.show()


def to_file(timestamp, simulator, prior_args, observation):
    """Write the permanent file for the experimental set-up"""
    simulator.to_file(timestamp)

    file = "experiment_files/experiment_" + timestamp + "/setup"

    f = open(file, "a")
    f.write("--------------------------------------------\n")
    f.write("Prior for the advection diffusion experiment\n")
    f.write("(Parameters for mean and cov of Gaussian distribution):\n")
    f.write("mean_upshift = " + str(prior_args["mean_upshift"]) +  "\n")
    f.write("bell_center = " + str(prior_args["bell_center"]) + "\n")
    f.write("bell_sharpness = " + str(prior_args["bell_sharpness"]) + "\n")
    f.write("bell_scaling = " + str(prior_args["bell_scaling"]) + "\n")
    f.write("matern_phi = " + str(prior_args["matern_phi"]) + "\n")
    f.write("stddev = " + str(prior_args["stddev"]) + "\n")

    observation.setup_to_file(timestamp)

    observation.positions_to_file(timestamp)

    