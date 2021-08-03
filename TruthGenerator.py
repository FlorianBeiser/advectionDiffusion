import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_kernel(grid, prior_args):
    """Plotting 1D Matern kernel for given phi on the half x domain of the grid"""
    matern_phi = prior_args["matern_phi"]

    # Plot kernel
    h = np.arange(int(max(grid.nx,grid.ny)/2))
    matern_kernel = (1+matern_phi*h)*np.exp(-matern_phi*h)
    plt.title("Matern covariance kernel")
    plt.plot(matern_kernel)
    plt.xlabel("Distance")
    plt.ylabel("Correlation")

    # Plot threshold
    plt.plot(0.05*np.ones_like(h), "g")

    # Plot desired cutting area
    corr_min = int(max(grid.nx,grid.ny)/8)
    corr_max = int(max(grid.nx,grid.ny)/6)
    plt.fill_between(range(len(h)), 0, 0.1, where=(h >= corr_min) & (h <= corr_max), color="r", alpha=0.5)

    plt.legend(["cov kernel", "correlation range", "desired cut"])
    plt.show()


def plot_truth(state, grid):
     mean = np.reshape(state, (grid.ny,grid.nx))

     plt.imshow(mean, origin = "lower")
     plt.title("Truth")
     plt.colorbar(orientation="horizontal")
     plt.show()


def to_file(timestamp, simulator, prior_args, observation):
    """Write the permanent file for the experimental set-up"""
    simulator.to_file(timestamp)

    file = "experiment_files/experiment_" + timestamp + "/setup_" + timestamp

    f = open(file, "a")
    f.write("--------------------------------------------\n")
    f.write("Prior for the advection diffusion experiment\n")
    f.write("(Parameters for mean and cov of Gaussian distribution):\n")
    f.write("mean_upshift = " + str(prior_args["mean_upshift"]) +  "\n")
    f.write("bell_center = " + str(prior_args["bell_center"]) + "\n")
    f.write("bell_sharpness = " + str(prior_args["bell_sharpness"]) + "\n")
    f.write("bell_scaling = " + str(prior_args["bell_scaling"]) + "\n")
    f.write("matern_phi = " + str(prior_args["matern_phi"]) + "\n")

    observation.setup_to_file(timestamp)

    