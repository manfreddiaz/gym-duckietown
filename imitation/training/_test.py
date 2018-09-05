from imitation.training._drivers import Icra2019Driver
from imitation.training._settings import *
from imitation.training._parametrization import *
from imitation.training._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES

from imitation.learners import NeuralNetworkPolicy
from imitation.training._loggers import IILTestingLogger
from imitation.algorithms.iil_testing import InteractiveImitationTesting

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
        training=False
    )

    return policy

if __name__ == '__main__':
    algorithm = 0
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 0
    optimization_iteration = 1
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

    testing = InteractiveImitationTesting(
        env=environment,
        teacher=teacher(environment),
        learner=policy,
        horizon=HORIZONS[horizon_iteration],
        episodes=EPISODES[horizon_iteration]
    )

    # observers

    driver = Icra2019Driver(
        env=environment,
        at=MAP_STARTING_POSES[iteration],
        routine=testing
    )

    logger = IILTestingLogger(
        a
    )

    testing.test(debug=True)

    environment.close()