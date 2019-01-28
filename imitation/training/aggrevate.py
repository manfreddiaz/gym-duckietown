from imitation.training._behaviors import Icra2019Behavior
from imitation.training._settings import *
from imitation.training._optimization import *
from imitation.training._parametrization import *

from imitation.algorithms import AggreVaTe
from imitation.learners import NeuralNetworkPolicy, UARandomExploration
from imitation.training._loggers import IILTrainingLogger

ALGORITHM_NAME = ALGORITHMS[2]
SEEDS = [19048, 27009, 43831, 13603, 37218]  # Google Random number generator (1-50000)


def aggrevate(env, teacher, experiment_iteration, selected_parametrization, selected_optimization, selected_learning_rate,
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
            algorithm=ALGORITHM_NAME,
            experiment_iteration=config.iteration,
            parametrization_name=PARAMETRIZATIONS_NAMES[config.parametrization],
            horizon=HORIZONS[config.horizon],
            episodes=EPISODES[config.horizon],
            optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
            learning_rate=LEARNING_RATES[config.learning_rate],
            metadata={
                'decay': MIXING_DECAYS[config.decay],
                'seed': SEEDS[config.seed]
            }
        ),
        batch_size=32,
        epochs=10
    )

    return AggreVaTe(env=env,
        teacher=teacher,
        learner=learner,
        explorer=UARandomExploration(),
        horizon=task_horizon,
        episodes=task_episodes,
        alpha=MIXING_DECAYS[selected_mixing_decay],
        seed=SEEDS[config.seed]
    )


if __name__ == '__main__':
    parser = process_args()
    parser.add_argument('--decay', '-d', default=0, type=int)
    parser.add_argument('--seed', '-s', default=0, type=int)

    config = parser.parse_args()
    # training
    environment = simulation(at=MAP_STARTING_POSES[config.iteration])

    algorithm = aggrevate(
        env=environment,
        teacher=teacher(environment),
        experiment_iteration=config.iteration,
        selected_parametrization=config.parametrization,
        selected_optimization=config.optimization,
        selected_horizon=config.horizon,
        selected_episode=config.horizon,
        selected_learning_rate=config.learning_rate,
        selected_mixing_decay=config.decay
    )

    # observers
    driver = Icra2019Behavior(
        env=environment,
        at=MAP_STARTING_POSES[config.iteration],
        routine=algorithm
    )

    disk_entry = experimental_entry(
        algorithm=ALGORITHM_NAME,
        experiment_iteration=config.iteration,
        parametrization_name=PARAMETRIZATIONS_NAMES[config.parametrization],
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
        learning_rate=LEARNING_RATES[config.learning_rate],
        metadata={
            'decay': MIXING_DECAYS[config.decay],
            'seed': SEEDS[config.seed]
        }
    )
    logs = IILTrainingLogger(
        env=environment,
        routine=algorithm,
        log_file=disk_entry + 'training.log',
        data_file=disk_entry + 'dataset_evolution.pkl',
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon]
    )

    algorithm.train(debug=DEBUG)

    environment.close()