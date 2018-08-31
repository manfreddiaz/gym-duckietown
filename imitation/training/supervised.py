from imitation.training._settings import *

def supervised(env, teacher, experiment_iteration, selected_parametrization, selected_optimization, selected_horizon,
               selected_episode):

    learner = NeuralNetworkPolicy(
        parametrization=parametrization(
            iteration=selected_parametrization,
            optimization=selected_optimization
        ),
        storage_location=experimental_entry(
            algorithm='supervised',
            experiment_iteration=experiment_iteration,
            selected_parametrization=selected_parametrization,
            selected_horizon=selected_horizon,
            selected_episode=selected_episode,
            selected_optimization=selected_optimization
        )
    )

    return SupervisedLearning(env=env,
                              teacher=teacher,
                              learner=learner,
                              horizon=HORIZONS[selected_horizon],
                              episodes=EPISODES[selected_episode],
    )

if __name__ == '__main__':
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 0
    optimization_iteration = 3

    # training
    environment = simulation(at=MAP_STARTING_POSES[iteration])

    algorithm = supervised(
        env=environment,
        teacher=teacher(environment),
        experiment_iteration=iteration,
        selected_parametrization=parametrization_iteration,
        selected_optimization=optimization_iteration,
        selected_horizon=horizon_iteration,
        selected_episode=horizon_iteration
    )
    disk_entry = experimental_entry(
        algorithm='supervised',
        experiment_iteration=iteration,
        selected_parametrization=parametrization_iteration,
        selected_horizon=horizon_iteration,
        selected_episode=horizon_iteration,
        selected_optimization=optimization_iteration
    )
    logs = IILTrainingLogger(
        env=environment,
        algorithm=algorithm,
        log_file=disk_entry + 'training.log',
        data_file=disk_entry + 'dataset_evolution.pkl',
        horizon=HORIZONS[horizon_iteration],
        episodes=EPISODES[horizon_iteration]
    )

    algorithm.train(debug=DEBUG)

    environment.close()