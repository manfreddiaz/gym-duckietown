import ast

from imitation.training._drivers import Icra2019Driver
from imitation.training._settings import *
from imitation.training._parametrization import *
from imitation.training._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES

from imitation.learners import NeuralNetworkPolicy
from imitation.training._loggers import IILTestingLogger
from imitation.algorithms.iil_testing import InteractiveImitationTesting

def test(selected_algorithm, experiment_iteration, selected_parametrization, selected_optimization, selected_learning_rate,
               selected_horizon, selected_episode, metadata=None):

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
            learning_rate=LEARNING_RATES[selected_learning_rate],
            metadata=ast.literal_eval(metadata)
        ),
        training=False
    )

    return policy

if __name__ == '__main__':
    parser = process_args()

    config = parser.parse_args()

    print(config)

    # training
    environment = simulation(at=MAP_STARTING_POSES[config.iteration])

    policy = test(
        selected_algorithm=config.algorithm,
        experiment_iteration=config.iteration,
        selected_parametrization=config.parametrization,
        selected_optimization=config.optimization,
        selected_learning_rate=config.learning_rate,
        selected_horizon=config.horizon,
        selected_episode=config.horizon,
        metadata=config.metadata
    )

    testing = InteractiveImitationTesting(
        env=environment,
        teacher=teacher(environment),
        learner=policy,
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon]
    )

    # observers
    driver = Icra2019Driver(
        env=environment,
        at=MAP_STARTING_POSES[config.iteration],
        routine=testing
    )
    logging_entry = experimental_entry(
        algorithm=ALGORITHMS[config.algorithm],
        experiment_iteration=config.iteration,
        parametrization_name=PARAMETRIZATIONS_NAMES[config.parametrization],
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
        learning_rate=LEARNING_RATES[config.learning_rate]

    )
    print(logging_entry)
    logger = IILTestingLogger(
        env=environment,
        routine=testing,
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        log_file=logging_entry + 'testing.log'
    )

    testing.test(debug=False)

    environment.close()
