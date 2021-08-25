import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import scipy.stats

from statsmodels.distributions.empirical_distribution import ECDF


class Comparer:

    def __init__(self, statistics_kf, statistics_etkf, statistics_letkf, statistics_iewpf):
        self.statistics_kf    = statistics_kf
        self.statistics_etkf  = statistics_etkf
        self.statistics_letkf = statistics_letkf
        self.statistics_iewpf = statistics_iewpf

        self.grid = self.statistics_kf.simulator.grid

        self.poi = []

    def mean_plots(self):
        fig, axs = plt.subplots(2,4, figsize=(12,8))

        fig00 = axs[0,0].imshow(np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=10, vmax=15)
        axs[0,0].set_title("KF Mean")
        ax_divider = make_axes_locatable(axs[0,0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig00, cax=ax_cb, orientation="horizontal")

        fig01 = axs[0,1].imshow(np.reshape(self.statistics_etkf.mean, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=10, vmax=15)
        axs[0,1].set_title("ETKF Mean")
        ax_divider = make_axes_locatable(axs[0,1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig01, cax=ax_cb, orientation="horizontal")

        fig02 = axs[0,2].imshow(np.reshape(self.statistics_letkf.mean, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=10, vmax=15)
        axs[0,2].set_title("LETKF Mean")
        ax_divider = make_axes_locatable(axs[0,2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig02, cax=ax_cb, orientation="horizontal")

        fig03 = axs[0,3].imshow(np.reshape(self.statistics_iewpf.mean, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=10, vmax=15)
        axs[0,3].set_title("IEWPF Mean")
        ax_divider = make_axes_locatable(axs[0,3])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig03, cax=ax_cb, orientation="horizontal")


        mean_err_kf = np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))

        fig10 = axs[1,0].imshow(mean_err_kf, origin = "lower", vmin=-0.1, vmax=0.1)
        axs[1,0].set_title("KF Error")
        ax_divider = make_axes_locatable(axs[1,0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig10, cax=ax_cb, orientation="horizontal")

        mean_err_etkf = np.reshape(self.statistics_etkf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))

        fig11 = axs[1,1].imshow(mean_err_etkf, origin = "lower", vmin=-0.1, vmax=0.1)
        axs[1,1].set_title("ETKF Error")
        ax_divider = make_axes_locatable(axs[1,1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig11, cax=ax_cb, orientation="horizontal")

        mean_err_letkf = np.reshape(self.statistics_letkf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))

        fig12 = axs[1,2].imshow(mean_err_letkf, origin = "lower", vmin=-0.1, vmax=0.1)
        axs[1,2].set_title("LETKF Error")
        ax_divider = make_axes_locatable(axs[1,2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig12, cax=ax_cb, orientation="horizontal")

        mean_err_iewpf = np.reshape(self.statistics_iewpf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))

        fig13 = axs[1,3].imshow(mean_err_iewpf, origin = "lower", vmin=-0.1, vmax=0.1)
        axs[1,3].set_title("IEWPF Error")
        ax_divider = make_axes_locatable(axs[1,3])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig13, cax=ax_cb, orientation="horizontal")

        plt.show()

    
    def mean_rmse(self):
        mean_err_kf = np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))
        mean_err_etkf = np.reshape(self.statistics_etkf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))
        mean_err_letkf = np.reshape(self.statistics_letkf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))
        mean_err_iewpf = np.reshape(self.statistics_iewpf.mean, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx))

        mean_rmse_kf = np.sqrt(np.sum(mean_err_kf**2))
        mean_rmse_etkf = np.sqrt(np.sum(mean_err_etkf**2))
        mean_rmse_letkf = np.sqrt(np.sum(mean_err_letkf**2))
        mean_rmse_iewpf = np.sqrt(np.sum(mean_err_iewpf**2))

        return mean_rmse_kf, mean_rmse_etkf, mean_rmse_letkf, mean_rmse_iewpf


    def stddev_plots(self):
        fig, axs = plt.subplots(2,4, figsize=(12,8))

        fig00 = axs[0,0].imshow(np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=0, vmax=0.2)
        axs[0,0].set_title("KF Stddev")
        ax_divider = make_axes_locatable(axs[0,0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig00, cax=ax_cb, orientation="horizontal")

        fig01 = axs[0,1].imshow(np.reshape(self.statistics_etkf.stddev, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=0, vmax=0.2)
        axs[0,1].set_title("ETKF Stddev")
        ax_divider = make_axes_locatable(axs[0,1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig01, cax=ax_cb, orientation="horizontal")

        fig02 = axs[0,2].imshow(np.reshape(self.statistics_letkf.stddev, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=0, vmax=0.2)
        axs[0,2].set_title("LETKF Stddev")
        ax_divider = make_axes_locatable(axs[0,2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig02, cax=ax_cb, orientation="horizontal")

        fig03 = axs[0,3].imshow(np.reshape(self.statistics_iewpf.stddev, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=0, vmax=0.2)
        axs[0,3].set_title("IEWPF Stddev")
        ax_divider = make_axes_locatable(axs[0,3])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig03, cax=ax_cb, orientation="horizontal")


        stddev_err_kf = np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))

        fig10 = axs[1,0].imshow(stddev_err_kf, origin = "lower", vmin=-0.05, vmax=0.05)
        axs[1,0].set_title("KF Error")
        ax_divider = make_axes_locatable(axs[1,0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig10, cax=ax_cb, orientation="horizontal")

        stddev_err_etkf = np.reshape(self.statistics_etkf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))

        fig11 = axs[1,1].imshow(stddev_err_etkf, origin = "lower", vmin=-0.05, vmax=0.05)
        axs[1,1].set_title("ETKF Error")
        ax_divider = make_axes_locatable(axs[1,1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig11, cax=ax_cb, orientation="horizontal")

        stddev_err_letkf = np.reshape(self.statistics_letkf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))

        fig12 = axs[1,2].imshow(stddev_err_letkf, origin = "lower", vmin=-0.05, vmax=0.05)
        axs[1,2].set_title("LETKF Error")
        ax_divider = make_axes_locatable(axs[1,2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig12, cax=ax_cb, orientation="horizontal")

        stddev_err_iewpf = np.reshape(self.statistics_iewpf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))

        fig13 = axs[1,3].imshow(stddev_err_iewpf, origin = "lower", vmin=-0.05, vmax=0.05)
        axs[1,3].set_title("LETKF Error")
        ax_divider = make_axes_locatable(axs[1,3])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig13, cax=ax_cb, orientation="horizontal")


        plt.show()


    def stddev_rmse(self):
        stddev_err_kf = np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))
        stddev_err_etkf = np.reshape(self.statistics_etkf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))
        stddev_err_letkf = np.reshape(self.statistics_letkf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))
        stddev_err_iewpf = np.reshape(self.statistics_iewpf.stddev, (self.grid.ny, self.grid.nx)) - np.reshape(self.statistics_kf.stddev, (self.grid.ny, self.grid.nx))

        stddev_rmse_kf = np.sqrt(np.sum(stddev_err_kf**2))
        stddev_rmse_etkf = np.sqrt(np.sum(stddev_err_etkf**2))
        stddev_rmse_letkf = np.sqrt(np.sum(stddev_err_letkf**2))
        stddev_rmse_iewpf = np.sqrt(np.sum(stddev_err_iewpf**2))

        return stddev_rmse_kf, stddev_rmse_etkf, stddev_rmse_letkf, stddev_rmse_iewpf


    def cov_plots(self):
        fig, axs = plt.subplots(2,4, figsize=(12,8))

        fig00 = axs[0,0].imshow(self.statistics_kf.cov,vmin=0, vmax=0.01)
        axs[0,0].set_title("KF Cov")
        ax_divider = make_axes_locatable(axs[0,0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig00, cax=ax_cb, orientation="horizontal")

        fig01 = axs[0,1].imshow(self.statistics_etkf.cov,vmin=0, vmax=0.01)
        axs[0,1].set_title("ETKF Cov")
        ax_divider = make_axes_locatable(axs[0,1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig01, cax=ax_cb, orientation="horizontal")

        fig02 = axs[0,2].imshow(self.statistics_letkf.cov,vmin=0, vmax=0.01)
        axs[0,2].set_title("LETKF Cov")
        ax_divider = make_axes_locatable(axs[0,2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig02, cax=ax_cb, orientation="horizontal")

        fig03 = axs[0,3].imshow(self.statistics_iewpf.cov,vmin=0, vmax=0.01)
        axs[0,3].set_title("IEWPF Cov")
        ax_divider = make_axes_locatable(axs[0,3])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig03, cax=ax_cb, orientation="horizontal")


        fig10 = axs[1,0].imshow(abs(self.statistics_kf.cov-self.statistics_kf.cov),vmin=0.0, vmax=0.005)
        axs[1,0].set_title("KF Error")
        ax_divider = make_axes_locatable(axs[1,0])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig10, cax=ax_cb, orientation="horizontal")

        fig11 = axs[1,1].imshow(abs(self.statistics_kf.cov-self.statistics_etkf.cov),vmin=0.0, vmax=0.005)
        axs[1,1].set_title("ETKF Error")
        ax_divider = make_axes_locatable(axs[1,1])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig11, cax=ax_cb, orientation="horizontal")

        fig12 = axs[1,2].imshow(abs(self.statistics_kf.cov-self.statistics_letkf.cov),vmin=0.0, vmax=0.005)
        axs[1,2].set_title("LETKF Error")
        ax_divider = make_axes_locatable(axs[1,2])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig12, cax=ax_cb, orientation="horizontal")

        fig13 = axs[1,3].imshow(abs(self.statistics_kf.cov-self.statistics_iewpf.cov),vmin=0.0, vmax=0.005)
        axs[1,3].set_title("IEWPF Error")
        ax_divider = make_axes_locatable(axs[1,3])
        ax_cb = ax_divider.append_axes("bottom", size="10%", pad="20%")
        plt.colorbar(fig13, cax=ax_cb, orientation="horizontal")

        plt.show()


    def cov_frobenius_dist(self):
        cov_frob_kf = np.linalg.norm(self.statistics_kf.cov - self.statistics_kf.cov)
        cov_frob_etkf = np.linalg.norm(self.statistics_kf.cov - self.statistics_etkf.cov)
        cov_frob_letkf = np.linalg.norm(self.statistics_kf.cov - self.statistics_letkf.cov)
        cov_frob_iewpf = np.linalg.norm(self.statistics_kf.cov - self.statistics_iewpf.cov)

        return cov_frob_kf, cov_frob_etkf, cov_frob_letkf, cov_frob_iewpf


    def set_poi(self, pos):
        indicator_field = np.zeros((self.grid.ny, self.grid.nx))
        indicator_field[pos[1],pos[0]] = 1.0

        idx = np.where(indicator_field.flatten() != 0 )[0][0]

        self.poi.append(idx)


    def poi_plot(self, observation=None):
        plt.imshow(np.reshape(self.statistics_kf.mean, (self.grid.ny, self.grid.nx)), origin = "lower", vmin=10, vmax=15)
        if observation is not None:
            plt.scatter(observation.positions[:,0],observation.positions[:,1], c="red")
        plt.xlim(0, self.grid.nx-1)
        plt.ylim(0, self.grid.ny-1)

        for i in range(len(self.poi)):
            indicator = np.zeros(self.grid.N_x)
            indicator[self.poi[i]] = 1.0
            indicator_field = np.reshape(indicator, (self.grid.ny, self.grid.nx))
            plt.scatter(np.where(indicator_field != 0)[1][0], np.where(indicator_field != 0 )[0][0], c="black", s=250)
            plt.text(np.where(indicator_field != 0)[1][0], np.where(indicator_field != 0 )[0][0], str(i), c="white", fontsize=12)
        plt.show()

    def poi_hist(self, i):

        xmin = self.statistics_kf.mean[self.poi[i]] - 3*self.statistics_kf.stddev[self.poi[i]]
        xmax = self.statistics_kf.mean[self.poi[i]] + 3*self.statistics_kf.stddev[self.poi[i]]

        x = np.arange(xmin,xmax,0.01)
        density = scipy.stats.norm.pdf(x, loc=self.statistics_kf.mean[self.poi[i]], scale=self.statistics_kf.stddev[self.poi[i]])

        ymax = np.max(density)+1

        fig, axs = plt.subplots(1,4, figsize=(12,4))

        axs[0].plot(x, density)
        axs[0].set_title("KF Density")
        axs[0].set_ylabel("Point of Interest" + str(i))
        axs[0].set_ylim([0.0,ymax])

        axs[1].hist(self.statistics_etkf.ensemble.ensemble[self.poi[i],:], density=True, bins=40, range=(xmin,xmax))
        axs[1].plot(x, density)
        axs[1].set_title("ETKF Density")
        axs[1].set_ylim([0.0,ymax])

        axs[2].hist(self.statistics_letkf.ensemble.ensemble[self.poi[i],:], density=True, bins=40, range=(xmin,xmax))
        axs[2].plot(x, density)
        axs[2].set_title("LETKF Density")
        axs[2].set_ylim([0.0,ymax])
    
        axs[3].hist(self.statistics_iewpf.ensemble.ensemble[self.poi[i],:], density=True, bins=40, range=(xmin,xmax))
        axs[3].plot(x, density)
        axs[3].set_title("IEWPF Density")
        axs[3].set_ylim([0.0,ymax])

        plt.show()


    def poi_ecdf_plots(self, i):
        cdf = lambda x: scipy.stats.norm.cdf(x, \
                    loc=self.statistics_kf.mean[self.poi[i]], \
                    scale=self.statistics_kf.stddev[self.poi[i]])

        ecdf_etkf = ECDF(self.statistics_etkf.ensemble.ensemble[self.poi[i],:])
        ecdf_letkf = ECDF(self.statistics_letkf.ensemble.ensemble[self.poi[i],:])
        ecdf_iewpf = ECDF(self.statistics_iewpf.ensemble.ensemble[self.poi[i],:])

        xmin = self.statistics_kf.mean[self.poi[i]] - 3*self.statistics_kf.stddev[self.poi[i]]
        xmax = self.statistics_kf.mean[self.poi[i]] + 3*self.statistics_kf.stddev[self.poi[i]]

        X = np.arange(xmin, xmax, 0.01)

        fig, axs = plt.subplots(1,4, figsize=(12,4))

        axs[0].plot(X, cdf(X))
        axs[0].set_title("KF CDF")
        axs[0].set_ylabel("Point of Interest" + str(i))

        axs[1].plot(X, cdf(X))
        axs[1].plot(X, ecdf_etkf(X))
        axs[1].set_title("ETKF ECDF")

        axs[2].plot(X, cdf(X))
        axs[2].plot(X, ecdf_letkf(X))
        axs[2].set_title("LETKF ECDF")

        axs[3].plot(X, cdf(X))
        axs[3].plot(X, ecdf_iewpf(X))
        axs[3].set_title("IEWPF ECDF")

        plt.show()

    
    def poi_ecdf_err(self, i):

        cdf = lambda x: scipy.stats.norm.cdf(x, \
                    loc=self.statistics_kf.mean[self.poi[i]], \
                    scale=self.statistics_kf.stddev[self.poi[i]])

        ecdf_etkf = ECDF(self.statistics_etkf.ensemble.ensemble[self.poi[i],:])
        ecdf_letkf = ECDF(self.statistics_letkf.ensemble.ensemble[self.poi[i],:])
        ecdf_iewpf = ECDF(self.statistics_iewpf.ensemble.ensemble[self.poi[i],:])

        diff_etkf  = lambda x: abs(cdf(x)-ecdf_etkf(x))
        diff_letkf = lambda x: abs(cdf(x)-ecdf_letkf(x))
        diff_iewpf = lambda x: abs(cdf(x)-ecdf_iewpf(x))

        xmin = self.statistics_kf.mean[self.poi[i]] - 3*self.statistics_kf.stddev[self.poi[i]]
        xmax = self.statistics_kf.mean[self.poi[i]] + 3*self.statistics_kf.stddev[self.poi[i]]

        ecdf_err_etkf = scipy.integrate.quad(diff_etkf, xmin, xmax, limit=100)[0]
        ecdf_err_letkf = scipy.integrate.quad(diff_letkf, xmin, xmax, limit=100)[0]
        ecdf_err_iewpf = scipy.integrate.quad(diff_iewpf, xmin, xmax, limit=100)[0]

        return ecdf_err_etkf, ecdf_err_letkf, ecdf_err_iewpf
        


