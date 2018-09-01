from imitation.training._settings import *
from imitation.training._parametrization import *
from imitation.training._optimization import *

from imitation.learners import NeuralNetworkPolicy
from imitation.algorithms import SupervisedLearning
from imitation.training._logger import IILTrainingLogger

def supervised(env, teacher, experiment_iteration, selected_parametrization, selected_optimization, selected_learning_rate,
               selected_horizon, selected_episode):

    task_horizon = HORIZONS[selected_horizon]
    task_episodes = EPISODES[selected_episode]

    policy_parametrization = parametrization(
        iteration=selected_parametrization,
        extra_parameters={'samples': 25, 'dropout': 0.9}
    )

    policy_optimizer = optimizer(
        optimizer_iteration=selected_optimization,
        learning_rate_iteration=selected_learning_rate,
        parametrization=policy_parametrization,
        task_metadata=[task_horizon, task_episodes, 10]
    )

    learner = NeuralNetworkPolicy(
        parametrization=policy_parametrization,
        optimizer=policy_optimizer,
        storage_location=experimental_entry(
            algorithm='supervised',
            experiment_iteration=experiment_iteration,
            parametrization_name=PARAMETRIZATIONS_NAMES[selected_parametrization],
            horizon=task_horizon,
            episodes=task_episodes,
            optimization_name=OPTIMIZATION_METHODS_NAMES[selected_optimization],
            learning_rate=LEARNING_RATES[selected_learning_rate]
        )
    )

    return SupervisedLearning(env=env,
                              teacher=teacher,
                              learner=learner,
                              horizon=task_horizon,
                              episodes=task_episodes,
    )


if __name__ == '__main__':
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 0
    optimization_iteration = 2
    learning_rate_iteration = 1

    # training
    environment = simulation(at=MAP_STARTING_POSES[iteration])

    algorithm = supervised(
        env=environment,
        teacher=teacher(environment),
        experiment_iteration=iteration,
        selected_parametrization=parametrization_iteration,
        selected_optimization=optimization_iteration,
        selected_horizon=horizon_iteration,
        selected_episode=horizon_iteration,
        selected_learning_rate=learning_rate_iteration
    )
    disk_entry = experimental_entry(
            algorithm='supervised',
            experiment_iteration=iteration,
            parametrization_name=PARAMETRIZATIONS_NAMES[parametrization_iteration],
            horizon=HORIZONS[horizon_iteration],
            episodes=EPISODES[horizon_iteration],
            optimization_name=OPTIMIZATION_METHODS_NAMES[optimization_iteration],
            learning_rate=LEARNING_RATES[learning_rate_iteration]
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