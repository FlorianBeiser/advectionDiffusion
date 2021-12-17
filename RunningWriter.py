import numpy as np
import os 
import datetime

class RunningWriter:

    def __init__(self, trials, N_poi):
        self.trials = trials
        self.N_poi  = N_poi

        self.mean_rmse_etkfs  = np.zeros(trials)
        self.mean_rmse_letkfs = np.zeros(trials)
        self.mean_rmse_iewpfs = np.zeros(trials)

        self.stddev_rmse_etkfs  = np.zeros(trials)
        self.stddev_rmse_letkfs = np.zeros(trials)
        self.stddev_rmse_iewpfs = np.zeros(trials)

        self.cov_frob_etkfs   = np.zeros(trials)
        self.cov_frob_letkfs  = np.zeros(trials)
        self.cov_frob_iewpfs  = np.zeros(trials)

        self.ecdf_err_etkfs   = np.zeros((N_poi, trials))
        self.ecdf_err_letkfs  = np.zeros((N_poi, trials))
        self.ecdf_err_iewpfs  = np.zeros((N_poi, trials))

        self.corr_p2p_err_etkf  = np.zeros(trials)
        self.corr_p2p_err_letkf = np.zeros(trials)
        self.corr_p2p_err_iewpf = np.zeros(trials)


    def header2file(self, N_e, trails_truth, trails_init, timestamp=None):
        if timestamp is not None:
            self.result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            file = "experiment_files/experiment_" + timestamp + "/results_" + self.result_timestamp

            f = open(file, "a")
            f.write("--------------------------------------------\n")
            f.write("Same model, ")
            f.write(str(trails_truth) + " truthes, ")
            f.write(str(trails_init) + " ensemble initialisations ")
            f.write("with " + str(N_e) + " ensemble members\n")
            f.write("--------------------------------------------\n")
            f.close()


    def results(self):
        avg_mean_rmse_etkfs  = np.average(self.mean_rmse_etkfs)
        avg_mean_rmse_letkfs = np.average(self.mean_rmse_letkfs)
        avg_mean_rmse_iewpfs = np.average(self.mean_rmse_iewpfs)

        avg_stddev_rmse_etkfs  = np.average(self.stddev_rmse_etkfs)
        avg_stddev_rmse_letkfs = np.average(self.stddev_rmse_letkfs)
        avg_stddev_rmse_iewpfs = np.average(self.stddev_rmse_iewpfs)

        avg_cov_frob_etkfs   = np.average(self.cov_frob_etkfs)
        avg_cov_frob_letkfs  = np.average(self.cov_frob_letkfs)
        avg_cov_frob_iewpfs  = np.average(self.cov_frob_iewpfs)

        avg_ecdf_err_etkfs  = []
        avg_ecdf_err_letkfs = []
        avg_ecdf_err_iewpfs = []
        for p in range(self.N_poi):
            avg_ecdf_err_etkfs.append(np.average(self.ecdf_err_etkfs[p]))
            avg_ecdf_err_letkfs.append(np.average(self.ecdf_err_letkfs[p]))
            avg_ecdf_err_iewpfs.append(np.average(self.ecdf_err_iewpfs[p]))
            
        avg_corr_p2p_err_etkfs  = np.average(self.corr_p2p_err_etkf)
        avg_corr_p2p_err_letkfs = np.average(self.corr_p2p_err_letkf)
        avg_corr_p2p_err_iewpf  = np.average(self.corr_p2p_err_iewpf)

        return avg_mean_rmse_etkfs, avg_mean_rmse_letkfs, avg_mean_rmse_iewpfs, \
                avg_stddev_rmse_etkfs, avg_stddev_rmse_letkfs, avg_stddev_rmse_iewpfs, \
                avg_cov_frob_etkfs, avg_cov_frob_letkfs, avg_cov_frob_iewpfs, \
                avg_ecdf_err_etkfs, avg_ecdf_err_letkfs, avg_ecdf_err_iewpfs, \
                avg_corr_p2p_err_etkfs, avg_corr_p2p_err_letkfs, avg_corr_p2p_err_iewpf


    def results2file(self, timestamp=None, as_table=None):

        avg_mean_rmse_etkfs, avg_mean_rmse_letkfs, avg_mean_rmse_iewpfs, \
        avg_stddev_rmse_etkfs, avg_stddev_rmse_letkfs, avg_stddev_rmse_iewpfs, \
        avg_cov_frob_etkfs, avg_cov_frob_letkfs, avg_cov_frob_iewpfs, \
        avg_ecdf_err_etkfs, avg_ecdf_err_letkfs, avg_ecdf_err_iewpfs, \
        avg_corr_p2p_err_etkfs, avg_corr_p2p_err_letkfs, avg_corr_p2p_err_iewpf = self.results()
        
        if timestamp is not None:

            if as_table is None:
                file = "experiment_files/experiment_" + timestamp + "/results_" + self.result_timestamp

                f = open(file, "a")
                f.write("--------------------------------------------\n")
                f.write("Results from the Comparison of ETKF and LETKF\n")
                f.write("versus the analytical posterior from the KF\n")
                f.write("--------------------------------------------\n")
                f.write("Mean RMSE EKTF  = " + str(avg_mean_rmse_etkfs) + "\n")
                f.write("Mean RMSE LEKTF = " + str(avg_mean_rmse_letkfs) + "\n")
                f.write("Mean RMSE IEWPF = " + str(avg_mean_rmse_iewpfs) + "\n")
                f.write("\n")
                f.write("Stddev RMSE EKTF  = " + str(avg_stddev_rmse_etkfs) + "\n")
                f.write("Stddev RMSE LEKTF = " + str(avg_stddev_rmse_letkfs) + "\n")
                f.write("Stddev RMSE IEWPF = " + str(avg_stddev_rmse_iewpfs) + "\n")
                f.write("\n")
                f.write("Cov Frobenius ETKF  = " + str(avg_cov_frob_etkfs) + "\n")
                f.write("Cov Frobenius LETKF = " + str(avg_cov_frob_letkfs) + "\n")
                f.write("Cov Frobenius IEWPF = " + str(avg_cov_frob_iewpfs) + "\n")
                f.write("\n")

                for p in range(self.N_poi):
                    f.write("ECDF Dist at PoI" + str(p) + " ETKF  = " + str(avg_ecdf_err_etkfs[p]) + "\n")
                    f.write("ECDF Dist at PoI" + str(p) + " LETKF = " + str(avg_ecdf_err_letkfs[p]) + "\n")
                    f.write("ECDF Dist at PoI" + str(p) + " IEWPF = " + str(avg_ecdf_err_iewpfs[p]) + "\n")
            
            else:
                result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                file = "experiment_files/experiment_" + timestamp + "/results_" + result_timestamp

                table = np.column_stack((as_table, \
                    self.mean_rmse_etkfs, \
                    self.mean_rmse_letkfs, \
                    self.mean_rmse_iewpfs, \
                    self.stddev_rmse_etkfs, \
                    self.stddev_rmse_letkfs, \
                    self.stddev_rmse_iewpfs, \
                    self.cov_frob_etkfs, \
                    self.cov_frob_letkfs, \
                    self.cov_frob_iewpfs))

                for p in range(self.N_poi):
                    table = np.column_stack((table, self.ecdf_err_etkfs[p], self.ecdf_err_letkfs[p], self.ecdf_err_iewpfs[p]))

                np.savetxt(file, table)