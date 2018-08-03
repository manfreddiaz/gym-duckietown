import argparse
import math
import threading

import numpy as np
import pyglet
import gym

from controllers import JoystickController
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

from learning_iil.algorithms import DAggerLearning, AggreVaTeLearning, SupervisedLearning, UPMSLearning
from learning_iil.iil_recorder import ImitationLearningRecorder
from learning_iil.learners import UncertaintyAwareRandomController, UncertaintyAwareNNController, NeuralNetworkController
from learning_iil.teachers import UncertaintyAwareHumanController
from learning_iil.learners.models.tf.baselines import ResnetOneRegression, ResnetOneMixture
from learning_iil.learners.models.tf.uncertainty import MonteCarloDropoutResnetOneRegression, \
    MonteCarloDropoutResnetOneMixture, FortifiedResnetOneRegression, FortifiedResnetOneMixture

np.random.seed(1234)

TRAINING_STARTING_POSITIONS = [
    (0.8, 0.0, 1.5),
    (0.8, 0.0, 2.5),
    (1.5, 0.0, 3.5),
    (2.5, 0.0, 3.5),
    (4.1, 0.0, 2.0),
    (2.8, 0.0, 0.8),
]
TESTING_STARTING_POSITIONS = [
    (2.2, 0.0, 2.8),
    (1.4, 0.0, 2.3),
    (2.0, 0.0, 1.4),
    (2.0, 0.0, 2.5)
]
TESTING_STARTING_POSITIONS.extend(TRAINING_STARTING_POSITIONS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--controller', default='joystick')
    parser.add_argument('--controller_mapping', default='mappings/record.joystick.logitech.yaml')
    return parser.parse_args()


def create_environment(args, with_heading=True):
    if args.env_name == 'SimpleSim-v0':
        environment = SimpleSimEnv(
            max_steps=math.inf,
            domain_rand=False,
            draw_curve=False
        )
    else:
        environment = gym.make(args.env_name)
    if with_heading:
        environment = HeadingWrapper(environment)

    return environment


def create_learning_algorithm(environment, arguments):
    iteration = 1
    base_directory = 'trained_models/supervised/{}/ror_adag_64/'.format(iteration)
    horizon = 512
    iterations = 10

    # human controllder
    human_teacher = JoystickController(environment)
    human_teacher.load_mapping(arguments.controller_mapping)

    tf_model = ResnetOneRegression()
    tf_learner = NeuralNetworkController(env=environment,
                                         learner=tf_model,
                                         storage_location=base_directory)

    # explorer
    random_controller = UncertaintyAwareRandomController(environment)

    starting_position = TRAINING_STARTING_POSITIONS[np.random.randint(0, len(TRAINING_STARTING_POSITIONS))]
    # iil_learning = UPMSLearning(env=environment,
    #                              teacher=human_teacher,
    #                              learner=tf_learner,
    #                              explorer=random_controller,
    #                              horizon=horizon,
    #                              episodes=iterations,
    #                              starting_position=starting_position,
    #                              starting_angle=0.0)

    iil_learning = SupervisedLearning(env=environment,
                                      teacher=human_teacher,
                                      learner=tf_learner,
                                      horizon=horizon,
                                      episodes=iterations,
                                      starting_angle=0,
                                      starting_position=starting_position)

    recorder = ImitationLearningRecorder(env, iil_learning, base_directory + 'training.pkl', horizon=horizon,
                                         iterations=iterations)

    return recorder


if __name__ == '__main__':
    args = parse_args()
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
