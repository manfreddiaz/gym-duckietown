from ._settings import *
from ._optimization import *
from ._parametrization import *

from imitation.algorithms import DAgger
from imitation.learners import NeuralNetworkPolicy
from imitation.training._loggers import IILTrainingLogger

MIXING_DECAYS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def dagger(env, teacher, experiment_iteration, selected_parametrization, selected_optimization, selected_learning_rate,
           selected_horizon, selected_episode, selected_mixing_decay):

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
        task_metadata=[task_horizon, task_episodes, 1]
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
        ),
        batch_size=32,
        epochs=10
    )

    return DAgger(env=env,
                  teacher=teacher,
                  learner=learner,
                  horizon=task_horizon,
                  episodes=task_episodes,
                  alpha=MIXING_DECAYS[selected_mixing_decay]
    )


if __name__ == '__main__':
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 1
    optimization_iteration = 5
    learning_rate_iteration = 0
    mixing_decay_iteration = 0

    # training
    environment = simulation(at=MAP_STARTING_POSES[iteration])

    algorithm = dagger(
        env=environment,
        teacher=teacher(environment),
        experiment_iteration=iteration,
        selected_parametrization=parametrization_iteration,
        selected_optimization=optimization_iteration,
        selected_horizon=horizon_iteration,
        selected_episode=horizon_iteration,
        selected_learning_rate=learning_rate_iteration,
        selected_mixing_decay=mixing_decay_iteration
    )
    disk_entry = experimental_entry(
            algorithm='dagger',
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