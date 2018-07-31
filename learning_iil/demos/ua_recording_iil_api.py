import argparse
import math
import threading

import pyglet
import gym
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

from learning_iil.algorithms import DAggerLearning, AggreVaTeLearning, SupervisedLearning, UPMSLearning
from learning_iil.iil_recording_controller import ImitationLearningRecorder
from learning_iil.learners import UncertaintyAwareRandomController, UncertaintyAwareNNController
from learning_iil.teachers import UncertaintyAwareHumanController
from learning_iil.learners.models.tf.baselines import ResnetOneRegression, ResnetOneMixture


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
    # human controller
    joystick_controller = UncertaintyAwareHumanController(environment)
    joystick_controller.load_mapping(arguments.controller_mapping)

    tf_learner = ResnetOneMixture()
    tf_controller = UncertaintyAwareNNController(env=environment,
                                                 learner=tf_learner,
                                                 storage_location='trained_models/upms/cnn_reg_adagrad_1/')

    # explorer
    random_controller = UncertaintyAwareRandomController(environment)

    iil_algorithm = UPMSLearning(env=environment,
                                 teacher=joystick_controller,
                                 learner=tf_controller,
                                 explorer=random_controller,
                                 horizon=512,
                                 episodes=10,
                                 starting_position=(1.5, 0.0, 3.5),
                                 starting_angle=0.0)

    recorder = ImitationLearningRecorder(env, iil_algorithm, 'trained_models/upms/cnn_reg_adagrad_1/data.pkl')

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

    recording.close()
    env.close()
