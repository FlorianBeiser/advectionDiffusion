import numpy as np
import os 
import datetime

class RunningWriter:

    def __init__(self, trials, N_poi, N_corr_poi):
        self.trials = trials
        self.N_poi  = N_poi
        self.N_corr_poi = N_corr_poi

        self.mean_rmse_etkfs  = np.zeros(trials)
        self.mean_rmse_letkfs = np.zeros(trials)
        self.mean_rmse_iewpfs = np.zeros(trials)
        self.mean_rmse_mcs    = np.zeros(trials)

        self.stddev_rmse_etkfs  = np.zeros(trials)
        self.stddev_rmse_letkfs = np.zeros(trials)
        self.stddev_rmse_iewpfs = np.zeros(trials)
        self.stddev_rmse_mcs    = np.zeros(trials)

        self.cov_frob_etkfs   = np.zeros(trials)
        self.cov_frob_letkfs  = np.zeros(trials)
        self.cov_frob_iewpfs  = np.zeros(trials)
        self.cov_frob_mcs     = np.zeros(trials)

        self.cov_frob_etkfs_close   = np.zeros(trials)
        self.cov_frob_letkfs_close  = np.zeros(trials)
        self.cov_frob_iewpfs_close  = np.zeros(trials)
        self.cov_frob_mcs_close     = np.zeros(trials)

        self.cov_frob_etkfs_far   = np.zeros(trials)
        self.cov_frob_letkfs_far  = np.zeros(trials)
        self.cov_frob_iewpfs_far  = np.zeros(trials)
        self.cov_frob_mcs_far     = np.zeros(trials)

        self.ecdf_err_etkfs   = np.zeros((N_poi, trials))
        self.ecdf_err_letkfs  = np.zeros((N_poi, trials))
        self.ecdf_err_iewpfs  = np.zeros((N_poi, trials))
        self.ecdf_err_mcs     = np.zeros((N_poi, trials))

        self.corr_p2p_err_etkf  = np.zeros((N_corr_poi, trials))
        self.corr_p2p_err_letkf = np.zeros((N_corr_poi, trials))
        self.corr_p2p_err_iewpf = np.zeros((N_corr_poi, trials))
        self.corr_p2p_err_mc    = np.zeros((N_corr_poi, trials))


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


    def results(self, mode="avg"):

        vars_names = list(filter(lambda n: n[0]!="N" and  n[0]!="_" and n!="trials" and n!="results" and "2" not in n , dir(self)))

        stats = {}

        for var_name in vars_names:
            var = getattr(self, var_name)
            if len(var.shape) > 1: 
                if mode == "avg":
                    avgs = np.zeros(var.shape[0])
                    for i in range(var.shape[0]):
                        avgs[i] = np.average(var[i])
                    stats["avg_"+var_name] = avgs
                else:
                    stds = np.zeros(var.shape[0])
                    for i in range(var.shape[0]):
                        stds[i] = np.std(var[i])
                    stats["avg_"+var_name] = stds

            else:
                if mode == "avg":
                    stats["avg_"+var_name] = np.average(var)
                if mode == "std":
                    stats["std_"+var_name] = np.std(var)

        return  stats


    def results2file(self, timestamp=None, table=None, title=None, mode="avg"):

        
        if timestamp is not None:

            if table is None:
                file = "experiment_files/experiment_" + timestamp + "/results_" + self.result_timestamp

                stats = self.results()

                f = open(file, "a")
                f.write("--------------------------------------------\n")
                f.write("Results from the Comparison of ETKF and LETKF\n")
                f.write("versus the analytical posterior from the KF\n")
                f.write("--------------------------------------------\n")
                f.write("Mean RMSE EKTF  = " + str(stats("avg_mean_rmse_etkfs")) + "\n")
                f.write("Mean RMSE LEKTF = " + str(stats("avg_mean_rmse_letkfs")) + "\n")
                f.write("Mean RMSE IEWPF = " + str(stats("avg_mean_rmse_iewpfs")) + "\n")
                f.write("Mean RMSE MC    = " + str(stats("avg_mean_rmse_mcs")) + "\n")
                f.write("\n")
                f.write("Stddev RMSE EKTF  = " + str(stats("avg_stddev_rmse_etkfs")) + "\n")
                f.write("Stddev RMSE LEKTF = " + str(stats("avg_stddev_rmse_letkfs")) + "\n")
                f.write("Stddev RMSE IEWPF = " + str(stats("avg_stddev_rmse_iewpfs")) + "\n")
                f.write("Stddev RMSE MC    = " + str(stats("avg_stddev_rmse_mcs")) + "\n")
                f.write("\n")
                f.write("Cov Frobenius ETKF  = " + str(stats("avg_cov_frob_etkfs")) + "\n")
                f.write("Cov Frobenius LETKF = " + str(stats("avg_cov_frob_letkfs")) + "\n")
                f.write("Cov Frobenius IEWPF = " + str(stats("avg_cov_frob_iewpfs")) + "\n")
                f.write("Cov Frobenius MC    = " + str(stats("avg_cov_frob_mcs")) + "\n")
                f.write("\n")
                f.write("Cov Frobenius ETKF (close)  = " + str(stats("avg_cov_frob_etkfs_close")) + "\n")
                f.write("Cov Frobenius LETKF (close) = " + str(stats("avg_cov_frob_letkfs_close")) + "\n")
                f.write("Cov Frobenius IEWPF (close) = " + str(stats("avg_cov_frob_iewpfs_close")) + "\n")
                f.write("Cov Frobenius MC    (close) = " + str(stats("avg_cov_frob_mcs_close")) + "\n")
                f.write("\n")
                f.write("Cov Frobenius ETKF (far)  = " + str(stats("avg_cov_frob_etkfs_far")) + "\n")
                f.write("Cov Frobenius LETKF (far) = " + str(stats("avg_cov_frob_letkfs_far")) + "\n")
                f.write("Cov Frobenius IEWPF (far) = " + str(stats("avg_cov_frob_iewpfs_far")) + "\n")
                f.write("Cov Frobenius MC    (far) = " + str(stats("avg_cov_frob_mcs_far")) + "\n")
                f.write("\n")


                for p in range(self.N_poi):
                    f.write("ECDF Dist at PoI" + str(p) + " ETKF  = " + str(stats("avg_ecdf_err_etkfs")[p]) + "\n")
                    f.write("ECDF Dist at PoI" + str(p) + " LETKF = " + str(stats("avg_ecdf_err_letkfs")[p]) + "\n")
                    f.write("ECDF Dist at PoI" + str(p) + " IEWPF = " + str(stats("avg_ecdf_err_iewpfs")[p]) + "\n")

                for p in range(self.N_poi):     
                    f.write("Correlation error from point"+str(p)+" ETKF  = " + str(stats("avg_corr_p2p_err_etkfs")[p]) + "\n")
                    f.write("Correlation error from point"+str(p)+" LETKF = " + str(stats("avg_corr_p2p_err_letkfs")[p]) + "\n")
                    f.write("Correlation error from point"+str(p)+" IEWPF = " + str(stats("avg_corr_p2p_err_iewpfs")[p]) + "\n")
                    f.write("\n")
            
            else:
                result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                file = "experiment_files/experiment_" + timestamp + "/results_" + result_timestamp + "_" + mode

                headers = title

                vars_names = sorted(list(filter(lambda n: n[0]!="N" and  n[0]!="_" and n!="trials" and n!="results" and "2" not in n, dir(self))))

                for var_name in vars_names:
                    var = getattr(self, var_name)
                    if len(var.shape) > 1: 
                        for i in range(var.shape[0]):
                            headers = headers +" "+ var_name+str(i)
                            table = np.column_stack((table, var[i]))
                    else:
                        headers = headers +" "+mode+"_"+ var_name
                        table = np.column_stack((table, var))

                np.savetxt(file, table, header=headers)


    def results2write(self, stats, trial):

        for key in stats.keys():
            value = stats[key]
            if np.isscalar(value):
                getattr(self,key[4:])[trial] = value
            else: 
                getattr(self,key[4:])[:,trial] = value
            
            # print(key)
            # print(value)
            # print(getattr(self,key[4:]))

            