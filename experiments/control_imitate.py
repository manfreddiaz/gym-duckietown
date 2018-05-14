#!/usr/bin/env python3

import time
import sys
import argparse
import math

import torch

import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

from imitate import Model
from utils import make_var

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='SimpleSim-v0')
parser.add_argument('--map-file', default='gym_duckietown/maps/udem1.yaml')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--no-random', action='store_true', help='disable domain randomization')
parser.add_argument('--full-res', action='store_true', help='render at full window resolution')
args = parser.parse_args()

if args.env_name == 'SimpleSim-v0':
    env = SimpleSimEnv(
        map_file = args.map_file,
        draw_curve = args.draw_curve,
        domain_rand = not args.no_random
    )
    env.max_steps = math.inf
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

model = Model()
model.load_state_dict(torch.load('trained_models/imitate.pt'))
model.eval()
model.cuda()

#try:
while True:
    obs = obs.transpose(2, 0, 1)
    obs = make_var(obs).unsqueeze(0)

    vels = model(obs)

    vels = vels.squeeze().data.cpu().numpy()

    print(vels)

    vels *= 0.8

    obs, reward, done, info = env.step(vels)
    #print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

    env.render()

    if done:
        print('done!')
        env.reset()
        env.render()

    #time.sleep(0.1)
    #time.sleep(0.015)

#except:
#    print('closing env')
#    env.close()
#    time.sleep(0.25)
