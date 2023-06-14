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

observation = Observation.Observation(grid)
observation.set_positions([[25,15]])

prior_args = Statistics.prior_args_from_file(timestamp)

print("done\n")


# %% 
# LOCALISATION IEWPF
iewpfQphis = [3.0, 5.0, 7.0, 11.0]

iewpfQs = [ Sampler.Sampler(grid, {"mean_upshift" : 0.0, "matern_phi" : phi, "stddev" : simulator.noise_stddev} ).cov for phi in iewpfQphis]


# %%
# LOCALISATION LETKF
scale_rs = [100,9,6,3]

# %%
trials_truth = 20
trials_init  = 5

N_e = 50

# %%
kfmeans = np.zeros((len(iewpfQphis), trials_truth*trials_init, grid.nx*grid.ny))
kfcovs = np.zeros((len(iewpfQphis), trials_truth*trials_init, grid.nx*grid.ny, grid.nx*grid.ny))

states_iewpf = np.zeros((len(iewpfQphis), trials_truth*trials_init, grid.nx*grid.ny, N_e))
states_letkf = np.zeros((len(iewpfQphis), trials_truth*trials_init, grid.nx*grid.ny, N_e))

# %%
# Repeating ensemble runs
for trial_model in range(len(iewpfQphis)):

    for trail_truth in range(trials_truth):
        # Truth
        print("\nModel", trial_model, ", Truth", trail_truth)
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
            statistics_kf.propagate(25)
            kalmanFilter.filter(statistics_kf.mean, statistics_kf.cov, observation.obses[t])


        for trial_init in range(trials_init):
            print("\nModel", trial_model, ", Truth", trail_truth, ", Init", trial_init)

            # ETKF 
            if trial_model == 0:
                print("ETKF DA")
                statistics_etkf = Statistics.Statistics(simulator, N_e, safe_history=True)
                statistics_etkf.set_prior(prior_args)

                etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

                for t in range(observation.N_obs):
                    statistics_etkf.propagate(25)
                    etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])

            # LETKF
            if trial_model > 0:
                print("LETKF DA")
                statistics_letkf = Statistics.Statistics(simulator, N_e, safe_history=True)
                statistics_letkf.set_prior(prior_args)

                sletkFilter = SLETKalmanFilter.SLETKalman(statistics_letkf, observation, scale_rs[trial_model])

                for t in range(observation.N_obs):
                    statistics_letkf.propagate(25)
                    sletkFilter.filter(statistics_letkf.ensemble.ensemble, observation.obses[t])

            # IEWPF
            print("IEWPF DA")
            statistics_iewpf = Statistics.Statistics(simulator, N_e, safe_history=True)
            statistics_iewpf.set_prior(prior_args)

            iewpFilter = IEWParticleFilter.IEWParticle(statistics_iewpf, observation, beta=0.55, Q=iewpfQs[trial_model])

            for t in range(observation.N_obs):
                statistics_iewpf.propagate(25, model_error=False)
                iewpFilter.filter(statistics_iewpf.ensemble.ensemble, observation.obses[t])


            # Comparison
            print("Storing")
            trial = trail_truth*trials_init + trial_init

            kfmeans[trial_model, trial] = statistics_kf.mean
            kfcovs[trial_model, trial]  = statistics_kf.cov

            states_iewpf[trial_model, trial] = statistics_iewpf.ensemble.ensemble
            if trial_model == 0:
                states_letkf[trial_model, trial] = statistics_etkf.ensemble.ensemble
            else:
                states_letkf[trial_model, trial] = statistics_letkf.ensemble.ensemble

            print("done")


# %%
import datetime
result_timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

file = "experiment_files/experiment_" + timestamp + "/localisation_results_" + result_timestamp
f = open(file, "a")
f.write("IEWPF phis: " + ",".join([str(phi) for phi in iewpfQphis]))

np.save("experiment_files/experiment_" + timestamp + "/locSingle_KFmeans_"+result_timestamp+".npy", kfmeans)
np.save("experiment_files/experiment_" + timestamp + "/locSingle_KFcovs_"+result_timestamp+".npy", kfcovs)
np.save("experiment_files/experiment_" + timestamp + "/locSingle_IEWPFQ_"+result_timestamp+".npy", states_iewpf)
np.save("experiment_files/experiment_" + timestamp + "/locSingle_LETKFr_"+result_timestamp+".npy", states_letkf)
