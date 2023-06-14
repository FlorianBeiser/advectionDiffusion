# %% 
"""
Example:
python run_FilteringComparison.py -m ensemble_size
"""

# %% 
import numpy as np

# %%
import Sampler
import Simulator
import Observation
import Statistics

import KalmanFilter
import ETKalmanFilter
import SLETKalmanFilter
import IEWParticleFilter

import Comparer
import RunningWriter


# %%
# Initialisation

print("Initialising...")

timestamp = "2022_03_02-12_44_46"
grid, simulator = Simulator.from_file(timestamp)

observation = Observation.from_file(grid, timestamp)

prior_args = Statistics.prior_args_from_file(timestamp)

print("done\n")


# %%
trials_truth = 20
trials_init  = 5

N_e = 50

# %%
prior_kfmeans = np.zeros((trials_truth*trials_init, grid.nx*grid.ny))
prior_kfcovs = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, grid.nx*grid.ny))

prior_states_iewpf = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, N_e))
prior_states_etkf = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, N_e))
prior_states_letkf = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, N_e))

posterior_kfmeans = np.zeros((trials_truth*trials_init, grid.nx*grid.ny))
posterior_kfcovs = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, grid.nx*grid.ny))

posterior_states_iewpf = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, N_e))
posterior_states_etkf = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, N_e))
posterior_states_letkf = np.zeros((trials_truth*trials_init, grid.nx*grid.ny, N_e))

# %%
# Repeating ensemble runs
for trail_truth in range(trials_truth):
    # Truth
    print("\nTruth", trail_truth)
    observation.clear_observations()

    statistics_truth = Statistics.Statistics(simulator, 1)
    statistics_truth.set_prior(prior_args)

    for t in range(10):
        statistics_truth.propagate(25)
        observation.observe(statistics_truth.mean)
    
    # KF 
    print("KF DA")
    statistics_kf = Statistics.Statistics(simulator, safe_history=True)
    statistics_kf.set_prior(prior_args)

    kalmanFilter = KalmanFilter.Kalman(statistics_kf, observation)

    for t in range(observation.N_obs):
        if t == 9:
            prior_kfmeans[trail_truth*trials_init:(trail_truth+1)*trials_init] = statistics_kf.mean
            prior_kfcovs[trail_truth*trials_init:(trail_truth+1)*trials_init]  = statistics_kf.cov
        statistics_kf.propagate(25)
        kalmanFilter.filter(statistics_kf.mean, statistics_kf.cov, observation.obses[t])
        if t == 9:
            posterior_kfmeans[trail_truth*trials_init:(trail_truth+1)*trials_init] = statistics_kf.mean
            posterior_kfcovs[trail_truth*trials_init:(trail_truth+1)*trials_init]  = statistics_kf.cov


    for trial_init in range(trials_init):
        trial = trail_truth*trials_init + trial_init
        print("\nTruth", trail_truth, ", Init", trial_init)

        # ETKF 
        print("ETKF DA")
        statistics_etkf = Statistics.Statistics(simulator, N_e, safe_history=True)
        statistics_etkf.set_prior(prior_args)

        etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

        for t in range(observation.N_obs):
            if t == 9:
                prior_states_etkf[trial] = statistics_etkf.ensemble.ensemble
            statistics_etkf.propagate(25)
            etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])
            if t == 9:
                posterior_states_etkf[trial] = statistics_etkf.ensemble.ensemble

        # LETKF
        print("LETKF DA")
        statistics_letkf = Statistics.Statistics(simulator, N_e, safe_history=True)
        statistics_letkf.set_prior(prior_args)

        sletkFilter = SLETKalmanFilter.SLETKalman(statistics_letkf, observation, 6)

        for t in range(observation.N_obs):
            if t == 9:
                prior_states_letkf[trial] = statistics_letkf.ensemble.ensemble
            statistics_letkf.propagate(25)
            sletkFilter.filter(statistics_letkf.ensemble.ensemble, observation.obses[t])
            if t == 9:
                posterior_states_letkf[trial] = statistics_letkf.ensemble.ensemble

        # IEWPF
        print("IEWPF DA")
        statistics_iewpf = Statistics.Statistics(simulator, N_e, safe_history=True)
        statistics_iewpf.set_prior(prior_args)

        iewpFilter = IEWParticleFilter.IEWParticle(statistics_iewpf, observation, beta=0.55)

        for t in range(observation.N_obs):
            if t == 9:
                prior_states_iewpf[trial] = statistics_iewpf.ensemble.ensemble
            statistics_iewpf.propagate(25, model_error=False)
            iewpFilter.filter(statistics_iewpf.ensemble.ensemble, observation.obses[t])
            if t == 9:
                posterior_states_iewpf[trial] = statistics_iewpf.ensemble.ensemble




# %%
import datetime
result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


np.save("experiment_files/experiment_" + timestamp + "/priorKFmeans_"+result_timestamp+".npy", prior_kfmeans)
np.save("experiment_files/experiment_" + timestamp + "/priorKFcovs_"+result_timestamp+".npy", prior_kfcovs)
np.save("experiment_files/experiment_" + timestamp + "/posteriorKFmeans_"+result_timestamp+".npy", posterior_kfmeans)
np.save("experiment_files/experiment_" + timestamp + "/posteriorKFcovs_"+result_timestamp+".npy", posterior_kfcovs)
np.save("experiment_files/experiment_" + timestamp + "/priorIEWPF_"+result_timestamp+".npy", prior_states_iewpf)
np.save("experiment_files/experiment_" + timestamp + "/posteriorIEWPF_"+result_timestamp+".npy", posterior_states_iewpf)
np.save("experiment_files/experiment_" + timestamp + "/priorLETKF_"+result_timestamp+".npy", prior_states_letkf)
np.save("experiment_files/experiment_" + timestamp + "/posteriorLETKF_"+result_timestamp+".npy", posterior_states_letkf)
np.save("experiment_files/experiment_" + timestamp + "/priorETKF_"+result_timestamp+".npy", prior_states_etkf)
np.save("experiment_files/experiment_" + timestamp + "/posteriorETKF_"+result_timestamp+".npy", posterior_states_etkf)

