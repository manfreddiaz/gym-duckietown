import argparse

import math

import gym
import pyglet

from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

from replay.replay_controller import ReplayController


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='SimpleSim-v0')
    parser.add_argument('--map-name', default='udem1')
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

    env.reset()
    env.render()

    replay_controller = ReplayController(env, 'training.pkl')
    replay_controller.configure()
    replay_controller.open()

    pyglet.app.run()

    env.close()
