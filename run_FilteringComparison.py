
from datetime import time
import Simulator
import Observation
import Statistics

import KalmanFilter
import ETKalmanFilter
import SLETKalmanFilter

import Comparer
import RunningWriter

import numpy as np

# Initialisation

print("Initalising...")

timestamp = "2021_08_11-14_10_29"
grid, simulator = Simulator.from_file(timestamp)

obs_timestamp = "2021_08_11-14_10_37"
observation = Observation.from_file(grid, timestamp, obs_timestamp)

prior_args = Statistics.prior_args_from_file(timestamp)

statistics_kf = Statistics.Statistics(simulator)
statistics_kf.set_prior(prior_args)

print("done\n")


# Analytical solution

print("Analytical Solution...")

kalmanFilter = KalmanFilter.Kalman(statistics_kf, observation)

for t in range(observation.N_obs):
    statistics_kf.propagate(25)
    kalmanFilter.filter(statistics_kf.mean, statistics_kf.cov, observation.obses[t])

print("done\n")


# Repeated ensemble runs 

runningWriter = RunningWriter.RunningWriter(trials=100, N_poi=2)

for r in range(runningWriter.trials):
    print("Trial ", r, "...")

    # ETKF 
    print("ETKF DA")
    statistics_etkf = Statistics.Statistics(simulator, 100)
    statistics_etkf.set_prior(prior_args)

    etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

    for t in range(observation.N_obs):
        statistics_etkf.propagate(25)
        etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])

    # LETKF
    print("LETKF DA")
    statistics_letkf = Statistics.Statistics(simulator, 100)
    statistics_letkf.set_prior(prior_args)

    scale_r = 8
    sletkFilter = SLETKalmanFilter.SLETKalman(statistics_letkf, observation, scale_r)

    for t in range(observation.N_obs):
        statistics_letkf.propagate(25)
        sletkFilter.filter(statistics_letkf.ensemble.ensemble, observation.obses[t])

    # Comparison
    print("Comparing")
    comparer = Comparer.Comparer(statistics_kf, statistics_etkf, statistics_letkf)

    mean_rmse_kf, runningWriter.mean_rmse_etkfs[r], runningWriter.mean_rmse_letkfs[r] = comparer.mean_rmse()
    cov_frob_kf, runningWriter.cov_frob_etkfs[r], runningWriter.cov_frob_letkfs[r] = comparer.cov_frobenius_dist()

    comparer.set_poi([0,0])
    comparer.set_poi([15,25])

    runningWriter.ecdf_err_etkfs[0][r], runningWriter.ecdf_err_letkfs[0][r] = comparer.poi_ecdf_err(0)
    runningWriter.ecdf_err_etkfs[1][r], runningWriter.ecdf_err_letkfs[1][r] = comparer.poi_ecdf_err(1)
    
    print("done\n")

runningWriter.results2file(timestamp)
