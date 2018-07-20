import sys
import argparse
import math
import pyglet
import gym
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper
from controllers import JoystickController, TensorflowNNController, SharedController

AVAILABLE_CONTROLLERS = {
    'joystick': JoystickController,
    'nn_tf': TensorflowNNController,
    'shared': SharedController
}
AVAILABLE_MAPPINGS = {
    'joystick': {
        'logitech': 'devices/mappings/joystick.logitech.yaml',
        'generic': 'devices/mappings/joystick.generic.yaml'
    }

}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--controller', default='joystick')
    parser.add_argument('--controller_mapping', default='logitech')
    return parser.parse_args()


def create_environment(args, with_heading=True):
    if args.env_name == 'SimpleSim-v0':
        environment = SimpleSimEnv(
            max_steps=math.inf,
            domain_rand=False
        )
    else:
        environment = gym.make(args.env_name)
    if with_heading:
        environment = HeadingWrapper(environment)

    return environment


def create_controller(environment, arguments):
    if arguments.controller not in AVAILABLE_CONTROLLERS:
        raise NotImplementedError('No implementation found for controller = {}'
                                  .format(arguments.controller))

    CONTROLLER_CLASS = AVAILABLE_CONTROLLERS[arguments.controller]
    controller = CONTROLLER_CLASS(environment)

    if arguments.controller_mapping not in AVAILABLE_MAPPINGS[arguments.controller]:
        raise NotImplementedError('No mapping found for controller = {} and model = {}'
                                  .format(arguments.controller, arguments.controller_mapping))

    controller.load_mapping(AVAILABLE_MAPPINGS[arguments.controller][arguments.controller_mapping])


def create_shared_controller(environment, arguments):
    # human controller
    joystick_controller = JoystickController(env)
    joystick_controller.load_mapping(AVAILABLE_MAPPINGS['joystick']['logitech'])

    # nn controller
    tf_controller = TensorflowNNController(env)

    shared = SharedController(env, joystick_controller, tf_controller)


if __name__ == '__main__':
    args = parse_args()

    env = create_environment(args)
    env.reset()
    env.render()

    create_shared_controller(environment=env, arguments=args)

    pyglet.app.run()

    env.close()
