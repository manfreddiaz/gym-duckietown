import argparse
import math
import threading
import inspect

import numpy as np
import pyglet
import gym

from controllers import JoystickController
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

from learning_iil.algorithms import DAggerLearning, AggreVaTeLearning, SupervisedLearning, UPMSLearning, DropoutDAggerLearning
from learning_iil.algorithms.ua_pms_sl import UPMSSelfLearning
from learning_iil.iil_recorder import ImitationLearningRecorder
from learning_iil.learners import UncertaintyAwareRandomController, UncertaintyAwareNNController, \
    NeuralNetworkController, RandomController
from learning_iil.teachers import UncertaintyAwareHumanController
from learning_iil.learners.models.tf.baselines import ResnetOneRegression, ResnetOneMixture
from learning_iil.learners.models.tf.uncertainty import MonteCarloDropoutResnetOneRegression, \
    MonteCarloDropoutResnetOneMixture, FortifiedResnetOneRegression, FortifiedResnetOneMixture

SEEDS = [123, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012]

np.random.seed(1234)

TRAINING_STARTING_POSITIONS = [
    [(0.8, 0.0, 1.5), 10.90],
    [(0.8, 0.0, 2.5), 10.90],
    [(1.5, 0.0, 3.5), 12.56],
    [(2.5, 0.0, 3.5), 12.56],
    [(4.1, 0.0, 2.0), 14.14],
    [(2.8, 0.0, 0.8), 15.71],
]

DEFAULT_ITERATION = 1
base_directory = 'trained_models/upms_ne/{}/ror_64_32_adag/'.format(DEFAULT_ITERATION)
DEFAULT_HORIZON_LENGTH = 512
DEFAULT_EPISODES = 10


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


def create_environment(args, with_heading=True):
    if args.env_name == 'SimpleSim-v0':
        environment = SimpleSimEnv(
            max_steps=math.inf,
            domain_rand=args.domain_rand,
            draw_curve=False,
            map_name=args.map_name
        )
    else:
        environment = gym.make(args.env_name)
    if with_heading:
        environment = HeadingWrapper(environment)

    return environment


def create_learning_algorithm(environment, arguments):

    # human controller
    human_teacher = UncertaintyAwareHumanController(environment)
    human_teacher.load_mapping(arguments.controller_mapping)

    tf_model = MonteCarloDropoutResnetOneRegression()
    tf_learner = UncertaintyAwareNNController(env=environment,
                                              learner=tf_model,
                                              storage_location=base_directory)
    # explorer
    random_controller = UncertaintyAwareRandomController(environment)

    starting_position = TRAINING_STARTING_POSITIONS[np.random.randint(0, len(TRAINING_STARTING_POSITIONS))]
    iil_learning = UPMSLearning(env=environment,
                                teacher=human_teacher,
                                learner=tf_learner,
                                explorer=tf_learner,
                                horizon=DEFAULT_HORIZON_LENGTH,
                                episodes=DEFAULT_EPISODES,
                                starting_position=starting_position[0],
                                starting_angle=starting_position[1],
                                safety_coefficient=30)

    # iil_learning = SupervisedLearning(env=environment,
    #                                   teacher=human_teacher,
    #                                   learner=tf_learner,
    #                                   horizon=horizon,
    #                                   episodes=iterations,
    #                                   starting_angle=0,
    #                                   starting_position=starting_position)

    recorder = ImitationLearningRecorder(env, iil_learning, base_directory + 'training.pkl', horizon=DEFAULT_HORIZON_LENGTH,
                                         iterations=DEFAULT_EPISODES)

    return recorder


if __name__ == '__main__':
    # train(primary_parser())
    args = primary_parser()
    print(threading.current_thread().name)
    env = create_environment(args)
    env.reset()
    env.render()

    recording = create_learning_algorithm(environment=env, arguments=args)
    recording.configure()
    recording.open()
    recording.reset()

    # print('Press [START] to record....')
    recording.record(None)
    pyglet.app.run()
    recording.stop(None)

    recording.close()
    env.close()
