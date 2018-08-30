import argparse
import math
import threading
import inspect

import numpy as np
import pyglet
import gym

from gym_duckietown.envs import DuckietownEnv, MultiMapEnv
from controllers import JoystickController

from learning_iil.algorithms import DAggerLearning, AggreVaTeLearning, SupervisedLearning, UPMSLearning, \
    DropoutDAggerLearning, UPMSSelfLearning, UPMSDataAggregationLearning
from learning_iil.algorithms.ua_pms_da_sl import UPMSDataAggregationSelfLearning
from learning_iil.iil_recorder import ImitationLearningRecorder
from learning_iil.learners import UARandomExploration, UANeuralNetworkPolicy, \
    NeuralNetworkPolicy, RandomExploration
from learning_iil.teachers import UncertaintyAwareHumanController, UncertaintyAwarePurePursuitController
from learning_iil.learners.parametrizations.tf.baselines import ResnetOneRegression, ResnetOneMixture
from learning_iil.learners.parametrizations.tf.uncertainty import MonteCarloDropoutResnetOneRegression, \
    MonteCarloDropoutResnetOneMixture, FortifiedResnetOneRegression, FortifiedResnetOneMixture

SEEDS = [123, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012]

TRAINING_STARTING_POSITIONS = [
    [[0.8, 0.0, 1.5], 10.90],
    [[0.8, 0.0, 2.5], 10.90],
    [[1.5, 0.0, 3.5], 12.56],
    [[2.5, 0.0, 3.5], 12.56],
    [[4.1, 0.0, 2.0], 14.14],
    [[2.8, 0.0, 0.8], 15.71],
]

DEFAULT_ITERATION = 3
base_directory = 'trained_models/alg_upms/{}/ror_64_32_adag/'.format(DEFAULT_ITERATION)
DEFAULT_HORIZON_LENGTH = 1024
DEFAULT_EPISODES = 64

np.random.seed(SEEDS[DEFAULT_ITERATION])


def primary_parser():
    parser = argparse.ArgumentParser()

    # training arguments
    # parser.add_argument('--algorithm', choices=[algorithm for algorithm in ALGORITHMS.keys()])
    # parser.add_argument('--model', choices=[model for model in PARAMETRIZATIONS.keys()])
    # parser.add_argument('--episodes', default=DEFAULT_EPISODES)
    # parser.add_argument('--horizon', default=DEFAULT_HORIZON_LENGTH)
    # parser.add_argument('--iteration', default=1)
    #
    # simulator arguments
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--domain-rand', default=False, action='store_true')
    parser.add_argument('--controller', default='joystick')
    parser.add_argument('--controller_mapping', default='mappings/record.joystick.logitech.yaml')

    return parser.parse_args()


def create_environment(args):
    if args.env_name == 'SimpleSim-v0':
        environment = DuckietownEnv(
            domain_rand=True,
            max_steps=math.inf,
        )
    else:
        environment = gym.make(args.env_name)

    return environment


def create_algorithmic_training(environment, arguments):

    # human controller

    # human_teacher.load_mapping(arguments.controller_mapping)

    tf_model = MonteCarloDropoutResnetOneRegression()
    tf_learner = UANeuralNetworkPolicy(env=environment, parametrization=tf_model, storage_location=base_directory)
    # explorer
    random_controller = UARandomExploration(environment)

    starting_position = TRAINING_STARTING_POSITIONS[np.random.randint(0, len(TRAINING_STARTING_POSITIONS))]
    iil_learning = UPMSLearning(env=environment,
                                teacher=human_teacher,
                                learner=tf_learner,
                                explorer=random_controller,
                                horizon=DEFAULT_HORIZON_LENGTH,
                                episodes=DEFAULT_EPISODES,
                                starting_position=starting_position[0],
                                starting_angle=starting_position[1],
                                safety_coefficient=20)

    # iil_learning = DropoutDAggerLearning(env=environment,
    #                             teacher=human_teacher,
    #                             learner=tf_learner,
    #                             horizon=DEFAULT_HORIZON_LENGTH,
    #                             episodes=DEFAULT_EPISODES,
    #                             starting_position=starting_position[0],
    #                             starting_angle=starting_position[1],
    #                             threshold=0.1)

    # iil_learning = SupervisedLearning(env=environment,
    #                                   teacher=human_teacher,
    #                                   learner=tf_learner,
    #                                   horizon=DEFAULT_HORIZON_LENGTH,
    #                                   episodes=DEFAULT_EPISODES,
    #                                   starting_position=starting_position[0],
    #                                   starting_angle=starting_position[1])

    # iil_learning = DAggerLearning(env=environment,
    #                               teacher=human_teacher,
    #                               learner=tf_learner,
    #                               # explorer=random_controller,
    #                               horizon=DEFAULT_HORIZON_LENGTH,
    #                               episodes=DEFAULT_EPISODES,
    #                               starting_position=starting_position[0],
    #                               starting_angle=starting_position[1])

    recorder = ImitationLearningRecorder(env, iil_learning, base_directory + 'training.pkl',
                                         horizon=DEFAULT_HORIZON_LENGTH, iterations=DEFAULT_EPISODES)

    return recorder


def create_learning_algorithm(environment, arguments):

    # human controller
    human_teacher = UncertaintyAwareHumanController(environment)
    human_teacher.load_mapping(arguments.controller_mapping)

    tf_model = MonteCarloDropoutResnetOneRegression()
    tf_learner = UANeuralNetworkPolicy(env=environment, parametrization=tf_model, storage_location=base_directory)
    # explorer
    random_controller = UARandomExploration(environment)

    starting_position = TRAINING_STARTING_POSITIONS[np.random.randint(0, len(TRAINING_STARTING_POSITIONS))]
    iil_learning = UPMSDataAggregationSelfLearning(env=environment,
                                teacher=human_teacher,
                                learner=tf_learner,
                                explorer=random_controller,
                                horizon=DEFAULT_HORIZON_LENGTH,
                                episodes=DEFAULT_EPISODES,
                                starting_position=starting_position[0],
                                starting_angle=starting_position[1],
                                safety_coefficient=30)

    # iil_learning = DropoutDAggerLearning(env=environment,
    #                             teacher=human_teacher,
    #                             learner=tf_learner,
    #                             horizon=DEFAULT_HORIZON_LENGTH,
    #                             episodes=DEFAULT_EPISODES,
    #                             starting_position=starting_position[0],
    #                             starting_angle=starting_position[1],
    #                             threshold=0.1)

    # iil_learning = SupervisedLearning(env=environment,
    #                                   teacher=human_teacher,
    #                                   learner=tf_learner,
    #                                   horizon=DEFAULT_HORIZON_LENGTH,
    #                                   episodes=DEFAULT_EPISODES,
    #                                   starting_position=starting_position[0],
    #                                   starting_angle=starting_position[1])

    # iil_learning = DAggerLearning(env=environment,
    #                               teacher=human_teacher,
    #                               learner=tf_learner,
    #                               # explorer=random_controller,
    #                               horizon=DEFAULT_HORIZON_LENGTH,
    #                               episodes=DEFAULT_EPISODES,
    #                               starting_position=starting_position[0],
    #                               starting_angle=starting_position[1])

    # recorder = ImitationLearningRecorder(env, iil_learning, base_directory + 'training.pkl',
    #                                      horizon=DEFAULT_HORIZON_LENGTH, iterations=DEFAULT_EPISODES)

    return iil_learning


if __name__ == '__main__':
    # train(primary_parser())
    args = primary_parser()
    print(threading.current_thread().name)
    env = create_environment(args)
    env.reset()
    env.render()

    recording = create_algorithmic_training(environment=env, arguments=args)
    recording.configure()
    recording.open()
    recording.reset()

    # print('Press [START] to record....')
    recording.record(None)
    pyglet.app.run()
    recording.stop(None)

    recording.close()
    env.close()
