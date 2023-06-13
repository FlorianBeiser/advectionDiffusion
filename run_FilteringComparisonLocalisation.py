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
# LOCALISATION IEWPF
iewpfQphis = [5.0, 7.0, 11.0]

iewpfQs = [ Sampler.Sampler(grid, {"mean_upshift" : 0.0, "matern_phi" : phi, "stddev" : simulator.noise_stddev} ).cov for phi in iewpfQphis]

from matplotlib import pyplot as plt
for m in range(len(iewpfQphis)):
    plt.imshow(iewpfQs[m][0].reshape(grid.ny, grid.nx), vmin=0.0)


# %%
# LOCALISATION LETKF
scale_rs = [9,6,3]

W0s = []
for s, scale_r in enumerate(scale_rs):
    W0s.append( SLETKalmanFilter.SLETKalman.getCombinedWeights(np.zeros((1,2)), scale_r, grid.dx, grid.dy, grid.nx, grid.ny, None, 1.0) )
    plt.imshow(W0s[s])


# %%
plt.plot(1/simulator.noise_stddev**2*iewpfQs[0][0][0:15], c="C0")
plt.plot(1/simulator.noise_stddev**2*iewpfQs[1][0][0:15], c="C1")
plt.plot(1/simulator.noise_stddev**2*iewpfQs[2][0][0:15], c="C2")

plt.plot(W0s[0][0][0:15], c="C0", ls="--")
plt.plot(W0s[1][0][0:15], c="C1", ls="--")
plt.plot(W0s[2][0][0:15], c="C2", ls="--")

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
            # print("ETKF DA")
            # statistics_etkf = Statistics.Statistics(simulator, N_e, safe_history=True)
            # statistics_etkf.set_prior(prior_args)

            # etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

            # for t in range(observation.N_obs):
            #     statistics_etkf.propagate(25)
            #     etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])

            # LETKF
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

            # MC
            # print("MC")
            # statistics_mc = Statistics.Statistics(simulator, N_e, safe_history=True)
            # statistics_mc.set_prior(prior_args)

            # for t in range(observation.N_obs):
            #     statistics_mc.propagate(25)


            # Comparison
            print("Storing")
            trial = trail_truth*trials_init + trial_init

            kfmeans[trial_model, trial] = statistics_kf.mean
            kfcovs[trial_model, trial]  = statistics_kf.cov

            states_iewpf[trial_model, trial] = statistics_iewpf.ensemble.ensemble
            states_letkf[trial_model, trial] = statistics_letkf.ensemble.ensemble

            print("done")


# %%
np.save("loc_KFmeans.npy", kfmeans)
np.save("loc_KFcovs.npy", kfcovs)
np.save("loc_IEWPFQ.npy", states_iewpf)
np.save("loc_LETKFr.npy", states_letkf)