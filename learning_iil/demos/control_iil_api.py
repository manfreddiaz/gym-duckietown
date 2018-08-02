import argparse
import pyglet
import gym
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper
from controllers import JoystickController, ParallelController, SharedController

from learning_iil.learners import NeuralNetworkController
from learning_iil.learners.models.tf.baselines import ResnetOneRegression, ResnetOneMixture
from learning_iil.learners.models.tf.uncertainty import MonteCarloDropoutResnetOneMixture, \
    MonteCarloDropoutResnetOneRegression, FortifiedResnetOneRegression, FortifiedResnetOneMixture


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--controller', default='joystick')
    parser.add_argument('--controller_mapping', default='mappings/shared.joystick.logitech.yaml')
    return parser.parse_args()


def create_environment(args, with_heading=True):
    if args.env_name == 'SimpleSim-v0':
        environment = SimpleSimEnv(
            map_name=args.map_name,
            max_steps=5000,
            domain_rand=False,
            draw_curve=False
        )
    else:
        environment = gym.make(args.env_name)
    if with_heading:
        environment = HeadingWrapper(environment)

    return environment


def create_dagger_controller(environment, arguments):
    # human controller
    joystick_controller = JoystickController(environment)
    joystick_controller.load_mapping(arguments.controller_mapping)

    # nn controller

    tf_model = FortifiedResnetOneRegression()
    tf_controller = NeuralNetworkController(env=environment,
                                            learner=tf_model,
                                            storage_location='trained_models/upms/fort_cnn_reg_adagrad_1/',
                                            training=False)

    iil_algorithm = SharedController(env, joystick_controller, tf_controller)

    return iil_algorithm


if __name__ == '__main__':
    args = parse_args()

    env = create_environment(args)
    env.reset()
    env.render()

    dagger_controller = create_dagger_controller(environment=env, arguments=args)
    dagger_controller.configure()
    dagger_controller.open()
    print('Press [START] to switch controllers...')

    pyglet.app.run()

    dagger_controller.close()
    env.close()
