"""
Example:
python run_FilteringComparison.py -m ensemble_size
"""
import Simulator
import Observation
import Statistics

import KalmanFilter
import ETKalmanFilter
import SLETKalmanFilter
import IEWParticleFilter

import Comparer
import RunningWriter

# Initialisation

print("Initialising...")

timestamp = "2022_03_02-12_44_46"
grid, simulator = Simulator.from_file(timestamp)

observation = Observation.from_file(grid, timestamp)

prior_args = Statistics.prior_args_from_file(timestamp)

print("done\n")


# Repeated ensemble runs 

pois = [[0,0], [25,15], [0,1]]

corr_ref_pois = [[20,10],[21,10],[25,15]]

# Setting mode

import argparse
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
            '-m', required=True, dest='mode', choices=["ensemble_size", "observation_size", "advection", "model_noise"],
            help='specifying which parameter is changed throughout experiments')
parser.add_argument(
            '-tt', default=20, type=int, dest='trials_truth', help='how often the truth is re-initialized in the repeated experiments')
parser.add_argument(
            '-ti', default=5,  type=int, dest='trials_init', help='how often the ensemble is re-initialized in the repeated experiments'
)

args = parser.parse_args(sys.argv[1:])
mode = args.mode

if mode == "ensemble_size": 
    N_es = [25, 50, 100, 250, 1000, 5000]
    runningModelWriter = RunningWriter.RunningWriter(trials=len(N_es), N_poi=len(pois), N_corr_poi=len(corr_ref_pois))
    runningModelWriterSTD = RunningWriter.RunningWriter(trials=len(N_es), N_poi=len(pois), N_corr_poi=len(corr_ref_pois))
if mode == "observation_size":
    N_ys = [3, 4, 5, 10, 15]
    runningModelWriter = RunningWriter.RunningWriter(trials=len(N_ys), N_poi=len(pois), N_corr_poi=len(corr_ref_pois))
    runningModelWriterSTD = RunningWriter.RunningWriter(trials=len(N_ys), N_poi=len(pois), N_corr_poi=len(corr_ref_pois))
if mode == "advection":
    vs = [[0.5,0.5], [1.0, 0.5], [1.5, 0.5], [2.0, 0.5]]
    runningModelWriter = RunningWriter.RunningWriter(trials=len(vs), N_poi=len(pois), N_corr_poi=len(corr_ref_pois))
if mode == "model_noise":
    noise_stddevs = [0.05, 0.1, 0.25, 0.5]
    runningModelWriter = RunningWriter.RunningWriter(trials=len(noise_stddevs), N_poi=len(pois), N_corr_poi=len(corr_ref_pois))


# Repeating ensemble runs
for trial_model in range(runningModelWriter.trials):
    print("Changing the model! Set up ", trial_model)

    if mode == "ensemble_size":
        N_e = N_es[trial_model]
    else:
        N_e = 50

    if mode == "observation_size":
        observation.set_regular_positions(N_ys[trial_model])
        N_ys[trial_model] = observation.N_y

    if mode == "advection":
        simulator.v = vs[trial_model]

    if mode == "model_noise":
        prior_args["stddev"] = noise_stddevs[trial_model]


    trials_truth = args.trials_truth
    trials_init  = args.trials_init

    if N_e >= 1000:
        trials_truth = 5
        trials_init = 3 

    runningWriter = RunningWriter.RunningWriter(trials=trials_truth*trials_init, N_poi=len(pois), N_corr_poi=len(corr_ref_pois))

    for trail_truth in range(trials_truth):
        # Truth
        print("New true observations", trail_truth)
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
            print("Ensemble init ", trial_init)

            # ETKF 
            print("ETKF DA")
            statistics_etkf = Statistics.Statistics(simulator, N_e, safe_history=True)
            statistics_etkf.set_prior(prior_args)

            etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

            for t in range(observation.N_obs):
                statistics_etkf.propagate(25)
                etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])

            # LETKF
            print("LETKF DA")
            statistics_letkf = Statistics.Statistics(simulator, N_e, safe_history=True)
            statistics_letkf.set_prior(prior_args)

            scale_r = 6
            sletkFilter = SLETKalmanFilter.SLETKalman(statistics_letkf, observation, scale_r)

            for t in range(observation.N_obs):
                statistics_letkf.propagate(25)
                sletkFilter.filter(statistics_letkf.ensemble.ensemble, observation.obses[t])

            # IEWPF
            print("IEWPF DA")
            statistics_iewpf = Statistics.Statistics(simulator, N_e, safe_history=True)
            statistics_iewpf.set_prior(prior_args)

            iewpFilter = IEWParticleFilter.IEWParticle(statistics_iewpf, observation, beta=0.55)

            for t in range(observation.N_obs):
                statistics_iewpf.propagate(25, model_error=False)
                iewpFilter.filter(statistics_iewpf.ensemble.ensemble, observation.obses[t])

            # MC
            print("MC")
            statistics_mc = Statistics.Statistics(simulator, N_e, safe_history=True)
            statistics_mc.set_prior(prior_args)

            for t in range(observation.N_obs):
                statistics_mc.propagate(25)


            # Comparison
            print("Comparing")
            trial = trail_truth*trials_init + trial_init
            comparer = Comparer.Comparer(statistics_kf, statistics_etkf, statistics_letkf, statistics_iewpf, statistics_mc)

            mean_rmse_kf, runningWriter.mean_rmse_etkfs[trial], runningWriter.mean_rmse_letkfs[trial], runningWriter.mean_rmse_iewpfs[trial], runningWriter.mean_rmse_mcs[trial] = comparer.mean_rmse()
            stddev_rmse_kf, runningWriter.stddev_rmse_etkfs[trial], runningWriter.stddev_rmse_letkfs[trial], runningWriter.stddev_rmse_iewpfs[trial], runningWriter.stddev_rmse_mcs[trial] = comparer.stddev_rmse()
            cov_frob_kf, runningWriter.cov_frob_etkfs[trial], runningWriter.cov_frob_letkfs[trial], runningWriter.cov_frob_iewpfs[trial], runningWriter.cov_frob_mcs[trial] = comparer.cov_frobenius_dist()
            cov_frob_kf_close, runningWriter.cov_frob_etkfs_close[trial], runningWriter.cov_frob_letkfs_close[trial], runningWriter.cov_frob_iewpfs_close[trial], runningWriter.cov_frob_mcs_close[trial] = comparer.cov_frobenius_dist_close()
            cov_frob_kf_far, runningWriter.cov_frob_etkfs_far[trial], runningWriter.cov_frob_letkfs_far[trial], runningWriter.cov_frob_iewpfs_far[trial], runningWriter.cov_frob_mcs_far[trial] = comparer.cov_frobenius_dist_far()

            for p in range(len(pois)):
                comparer.set_poi(pois[p])
                
            for p in range(len(pois)):
                runningWriter.ecdf_err_etkfs[p][trial], runningWriter.ecdf_err_letkfs[p][trial], runningWriter.ecdf_err_iewpfs[p][trial], runningWriter.ecdf_err_mcs[p][trial] = comparer.poi_ecdf_err(p)

            comparer.set_corr_ref_pois(corr_ref_pois)

            for p in range(len(corr_ref_pois)):
                runningWriter.corr_err_etkf[p][trial], runningWriter.corr_err_letkf[p][trial], runningWriter.corr_err_iewpf[p][trial] = comparer.corr_p2p_err(p)

            print("done\n")


    runningModelWriter.results2write(runningWriter.results(), trial_model)
    runningModelWriterSTD.results2write(runningWriter.results("std"), trial_model)


if mode == "ensemble_size": 
    runningModelWriter.results2file(timestamp, N_es, "N_e", "avg")
    runningModelWriterSTD.results2file(timestamp, N_es, "N_e", "std")
if mode == "observation_size":
    runningModelWriter.results2file(timestamp, N_ys, "N_y")
if mode == "advection":
    runningModelWriter.results2file(timestamp, [v[0] for v in vs])
if mode == "model_noise":
    runningModelWriter.results2file(timestamp, noise_stddevs)