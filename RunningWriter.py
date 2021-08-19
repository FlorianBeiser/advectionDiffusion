import numpy as np
import os 
import datetime

class RunningWriter:

    def __init__(self, trials, N_poi):
        self.trials = trials
        self.N_poi  = N_poi

        self.mean_rmse_etkfs  = np.zeros(trials)
        self.mean_rmse_letkfs = np.zeros(trials)
        self.cov_frob_etkfs   = np.zeros(trials)
        self.cov_frob_letkfs  = np.zeros(trials)
        self.ecdf_err_etkfs   = np.zeros((N_poi, trials))
        self.ecdf_err_letkfs  = np.zeros((N_poi, trials))


    def results2file(self, timestamp=None):
        if timestamp is not None:
            result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            file = "experiment_files/experiment_" + timestamp + "/results_" + result_timestamp

            f = open(file, "a")
            f.write("--------------------------------------------\n")
            f.write("Results from the Comparison of ETKF and LETKF\n")
            f.write("versus the analytical posterior from the KF\n")
            f.write("--------------------------------------------\n")
            f.write("Mean RMSE EKTF  = " + str(np.average(self.mean_rmse_etkfs)) + "\n")
            f.write("Mean RMSE LEKTF = " + str(np.average(self.mean_rmse_letkfs)) + "\n")
            f.write("\n")
            f.write("Cov Frobenius ETKF  = " + str(np.average(self.cov_frob_etkfs)) + "\n")
            f.write("Cov Frobenius LETKF = " + str(np.average(self.cov_frob_letkfs)) + "\n")
            f.write("\n")

            for p in range(self.N_poi):
                f.write("ECDF Dist at PoI" + str(p) + " ETKF  = " + str(np.average(self.ecdf_err_etkfs[p])) + "\n")
                f.write("ECDF Dist at PoI" + str(p) + " LETKF = " + str(np.average(self.ecdf_err_letkfs[p])) + "\n")
