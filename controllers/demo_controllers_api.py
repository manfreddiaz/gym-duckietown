import sys
import argparse
import math
import pyglet
import gym
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper
from controllers import JoystickController

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
    parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
    parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
    return parser.parse_args()


def create_environment(args, with_heading=True):
    if args.env_name == 'SimpleSim-v0':
        environment = SimpleSimEnv(
            map_name = args.map_name,
            domain_rand = args.domain_rand,
            max_steps = math.inf
        )
    else:
        environment = gym.make(args.env_name)
    if with_heading:
        environment = HeadingWrapper(environment)

    return environment


env = create_environment(parse_args())
env.reset()
env.render()

controller = JoystickController(env, device_id=0)
controller.load_mapping('joystick', 'logitec')

pyglet.app.run()

env.close()
