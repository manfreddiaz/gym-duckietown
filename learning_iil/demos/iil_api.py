import argparse
import math
import pyglet
import gym
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper
from controllers import JoystickController

from learning_iil.algorithms.dagger import DAggerLearning
from learning_iil.learners.mock_straight import MockStraightController


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--controller', default='joystick')
    parser.add_argument('--controller_mapping', default='demos/shared.joystick.logitech.yaml')
    return parser.parse_args()


def create_environment(args, with_heading=True):
    if args.env_name == 'SimpleSim-v0':
        environment = SimpleSimEnv(
            max_steps=math.inf,
            domain_rand=False,
            draw_curve=True
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
    tf_controller = MockStraightController(environment)

    shared = DAggerLearning(env, joystick_controller, tf_controller, 100, 100)

    return shared


if __name__ == '__main__':
    args = parse_args()

    env = create_environment(args)
    env.reset()
    env.render()

    dagger_controller = create_dagger_controller(environment=env, arguments=args)
    dagger_controller.configure()
    dagger_controller.open()

    pyglet.app.run()

    dagger_controller.close()
    env.close()
