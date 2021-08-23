
import Simulator
import Observation
import Statistics

import KalmanFilter
import ETKalmanFilter
import SLETKalmanFilter

import Comparer
import RunningWriter

# Initialisation

print("Initialising...")

timestamp = "2021_08_11-14_10_29"
grid, simulator = Simulator.from_file(timestamp)

observation = Observation.from_file(grid, timestamp)

prior_args = Statistics.prior_args_from_file(timestamp)

print("done\n")


# Repeated ensemble runs 

N_es = [25, 50, 100, 250, 1000]

pois = [[0,0], [25,15], [0,1]]

runningModelWriter = RunningWriter.RunningWriter(trials=len(N_es), N_poi=len(pois))

for trial_model in range(runningModelWriter.trials):
    print("Changing the model! Set up ", trial_model)

    N_e = N_es[trial_model]

    trials_truth = 20
    trials_init  = 5

    runningWriter = RunningWriter.RunningWriter(trials=trials_truth*trials_init, N_poi=len(pois))
    runningWriter.header2file(N_e, trials_truth, trials_init, timestamp)

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
        statistics_kf = Statistics.Statistics(simulator)
        statistics_kf.set_prior(prior_args)

        kalmanFilter = KalmanFilter.Kalman(statistics_kf, observation)

        for t in range(observation.N_obs):
            statistics_kf.propagate(25)
            kalmanFilter.filter(statistics_kf.mean, statistics_kf.cov, observation.obses[t])


        for trial_init in range(trials_init):
            print("Ensemble init ", trial_init)

            # ETKF 
            print("ETKF DA")
            statistics_etkf = Statistics.Statistics(simulator, N_e)
            statistics_etkf.set_prior(prior_args)

            etkFilter = ETKalmanFilter.ETKalman(statistics_etkf, observation)

            for t in range(observation.N_obs):
                statistics_etkf.propagate(25)
                etkFilter.filter(statistics_etkf.ensemble.ensemble, observation.obses[t])

            # LETKF
            print("LETKF DA")
            statistics_letkf = Statistics.Statistics(simulator, N_e)
            statistics_letkf.set_prior(prior_args)

            scale_r = 8
            sletkFilter = SLETKalmanFilter.SLETKalman(statistics_letkf, observation, scale_r)

            for t in range(observation.N_obs):
                statistics_letkf.propagate(25)
                sletkFilter.filter(statistics_letkf.ensemble.ensemble, observation.obses[t])

            # Comparison
            print("Comparing")
            trial = trail_truth*trials_init + trial_init
            comparer = Comparer.Comparer(statistics_kf, statistics_etkf, statistics_letkf)

            mean_rmse_kf, runningWriter.mean_rmse_etkfs[trial], runningWriter.mean_rmse_letkfs[trial] = comparer.mean_rmse()
            cov_frob_kf, runningWriter.cov_frob_etkfs[trial], runningWriter.cov_frob_letkfs[trial] = comparer.cov_frobenius_dist()

            for p in range(len(pois)):
                comparer.set_poi(pois[p])
            
            for p in range(len(pois)):
                runningWriter.ecdf_err_etkfs[p][trial], runningWriter.ecdf_err_letkfs[p][trial] = comparer.poi_ecdf_err(p)
        
            print("done\n")


    runningModelWriter.mean_rmse_etkfs[trial_model], \
    runningModelWriter.mean_rmse_letkfs[trial_model], \
    runningModelWriter.cov_frob_etkfs[trial_model], \
    runningModelWriter.cov_frob_letkfs[trial_model], \
    runningModelWriter.ecdf_err_etkfs[:,trial_model], \
    runningModelWriter.ecdf_err_letkfs[:,trial_model]\
    = runningWriter.results()

    runningWriter.results2file(timestamp)

runningModelWriter.results2file(timestamp, N_es)