
import Simulator
import Observation
import Statistics

import KalmanFilter
import ETKalmanFilter
import SLETKalmanFilter

import Comparer

import numpy as np

# Initialisation

timestamp = "2021_08_11-14_10_29"
grid, simulator = Simulator.from_file(timestamp)

obs_timestamp = "2021_08_11-14_10_37"
observation = Observation.from_file(grid, timestamp, obs_timestamp)

prior_args = Statistics.prior_args_from_file(timestamp)

statistics_kf = Statistics.Statistics(simulator)
statistics_kf.set_prior(prior_args)

# Analytical solution

kalmanFilter = KalmanFilter.Kalman(statistics_kf, observation)

for t in range(observation.N_obs):
    statistics_kf.propagate(25)
    kalmanFilter.filter(statistics_kf.mean, statistics_kf.cov, observation.obses[t])


# Repeated ensemble runs 
trials = 100
mean_rmse_etkfs = np.zeros(trials)
mean_rmse_letkfs = np.zeros(trials)
cov_frob_etkfs = np.zeros(trials)
cov_frob_letkfs = np.zeros(trials)
ecdf_err_etkf0s = np.zeros(trials)
ecdf_err_etkf1s = np.zeros(trials)
ecdf_err_letkf0s = np.zeros(trials)
ecdf_err_letkf1s = np.zeros(trials)

for r in range(trials):

    # ETKF 
    statistics_etkf = Statistics.Statistics(simulator, 100)
    statistics_etkf.set_prior(prior_args)

    etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

    for t in range(observation.N_obs):
        statistics_etkf.propagate(25)
        etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])

    # LETKF
    statistics_letkf = Statistics.Statistics(simulator, 100)
    statistics_letkf.set_prior(prior_args)

    scale_r = 8
    sletkFilter = SLETKalmanFilter.SLETKalman(statistics_letkf, observation, scale_r)

    for t in range(observation.N_obs):
        statistics_letkf.propagate(25)
        sletkFilter.filter(statistics_letkf.ensemble.ensemble, observation.obses[t])
        statistics_letkf.plot()

    # Comparison
    comparer = Comparer.Comparer(statistics_kf, statistics_etkf, statistics_letkf)

    mean_rmse_kf, mean_rmse_etkfs[r], mean_rmse_letkfs[r] = comparer.mean_rmse()
    cov_frob_kf, cov_frob_etkfs[r], cov_frob_letkfs[r] = comparer.cov_frobenius_dist()

    comparer.set_poi([0,0])
    comparer.set_poi([15,25])

    ecdf_err_etkf0s[r], ecdf_err_letkf0s[r] = comparer.poi_ecdf_err(0)
    ecdf_err_etkf1s[r], ecdf_err_letkf1s[r] = comparer.poi_ecdf_err(1)


print("Mean RMSE EKTF", np.average(mean_rmse_etkfs))
print("Mean RMSE LEKTF", np.average(mean_rmse_letkfs))

print("Cov Frobenius ETKF", np.average(cov_frob_etkfs))
print("Cov Frobenius LETKF", np.average(cov_frob_letkfs))

print("ECDF Dist at Pos0 ETKF", np.average(ecdf_err_etkf0s))
print("ECDF Dist at Pos0 LETKF", np.average(ecdf_err_letkf0s))

print("ECDF Dist at Pos1 ETKF", np.average(ecdf_err_etkf1s))
print("ECDF Dist at Pos1 LETKF", np.average(ecdf_err_letkf1s))