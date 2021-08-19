import numpy as np
import os 
import datetime

class RunningWriter:

    def __init__(self, trials):
        self.trials = trials

        self.mean_rmse_etkfs = np.zeros(trials)
        self.mean_rmse_letkfs = np.zeros(trials)
        self.cov_frob_etkfs = np.zeros(trials)
        self.cov_frob_letkfs = np.zeros(trials)
        self.ecdf_err_etkf0s = np.zeros(trials)
        self.ecdf_err_etkf1s = np.zeros(trials)
        self.ecdf_err_letkf0s = np.zeros(trials)
        self.ecdf_err_letkf1s = np.zeros(trials)


    def results2file(self, timestamp=None):
        if timestamp is not None:
            result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            file = "experiment_files/experiment_" + timestamp + "/results_" + result_timestamp

            print("Mean RMSE EKTF", np.average(self.mean_rmse_etkfs))
            print("Mean RMSE LEKTF", np.average(self.mean_rmse_letkfs))

            print("Cov Frobenius ETKF", np.average(self.cov_frob_etkfs))
            print("Cov Frobenius LETKF", np.average(self.cov_frob_letkfs))

            print("ECDF Dist at Pos0 ETKF", np.average(self.ecdf_err_etkf0s))
            print("ECDF Dist at Pos0 LETKF", np.average(self.ecdf_err_letkf0s))

            print("ECDF Dist at Pos1 ETKF", np.average(self.ecdf_err_etkf1s))
            print("ECDF Dist at Pos1 LETKF", np.average(self.ecdf_err_letkf1s))