from ._settings import *
from ._parametrization import *
from ._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES

from imitation.learners import NeuralNetworkPolicy
from imitation.training._loggers import IILTestingLogger


def test(selected_algorithm, experiment_iteration, selected_parametrization, selected_optimization, selected_learning_rate,
               selected_horizon, selected_episode):

    task_horizon = HORIZONS[selected_horizon]
    task_episodes = EPISODES[selected_episode]

    policy_parametrization = parametrization(
            iteration=selected_parametrization,
            extra_parameters={'samples': 25, 'dropout': 0.9}
    )

    policy = NeuralNetworkPolicy(
        parametrization=policy_parametrization,
        storage_location=experimental_entry(
            algorithm=ALGORITHMS[selected_algorithm],
            experiment_iteration=experiment_iteration,
            parametrization_name=PARAMETRIZATIONS_NAMES[selected_parametrization],
            horizon=task_horizon,
            episodes=task_episodes,
            optimization_name=OPTIMIZATION_METHODS_NAMES[selected_optimization],
            learning_rate=LEARNING_RATES[selected_learning_rate]
        ),
        batch_size=32,
        epochs=10,
        training=False
    )

    return policy

if __name__ == '__main__':
    algorithm = 0
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 1
    optimization_iteration = 5
    learning_rate_iteration = 0

    # training
    environment = simulation(at=MAP_STARTING_POSES[iteration])

    policy = test(
        selected_algorithm=algorithm,
        experiment_iteration=iteration,
        selected_parametrization=parametrization_iteration,
        selected_optimization=optimization_iteration,
        selected_learning_rate=learning_rate_iteration,
        selected_horizon=horizon_iteration,
        selected_episode=horizon_iteration
    )


    environment.close()