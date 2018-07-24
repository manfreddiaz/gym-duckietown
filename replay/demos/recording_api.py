import argparse

import math

import gym
import pyglet

from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

from replay.recording_wrapper import RecordingWrapper
from controllers import JoystickController

AVAILABLE_MAPPINGS = {
    'joystick': {
        'logitech': '../controllers/devices/mappings/joystick.logitech.yaml',
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
            domain_rand=False,
            draw_curve=True
        )
    else:
        environment = gym.make(args.env_name)
    if with_heading:
        environment = HeadingWrapper(environment)

    return environment


if __name__ == '__main__':
    env = create_environment(parse_args())
    env = RecordingWrapper(env, 'record.pkl')

    env.reset()
    env.render()

    joystick_controller = JoystickController(env)
    joystick_controller.load_mapping('demos/mappings/joystick.logitech.yaml')

    pyglet.app.run()

    env.close()
