#!/usr/bin/env python3

import time
import sys
import argparse
import math

import torch

import numpy as np
import gym
import gym_duckietown

from imitate import Model
from utils import make_var

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='SimpleSim-v0')
args = parser.parse_args()

env = gym.make(args.env_name)
env.max_steps = math.inf

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

    time.sleep(0.1)

#except:
#    print('closing env')
#    env.close()
#    time.sleep(0.25)
