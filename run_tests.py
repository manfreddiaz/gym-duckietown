#!/usr/bin/env python3

import os
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

env = gym.make('SimpleSim-v0')

# Check that the human rendering resembles the agent's view
first_obs = env.reset()
first_render = env.render('rgb_array')
m0 = first_obs.mean()
m1 = first_render.mean()
assert m0 > 0 and m0 < 255
assert abs(m0 - m1) < 5

# Check that stepping works
obs, _, _, _ = env.step(np.array([0.1, 0.1]))

# Try loading each of the available map files
for map_file in os.listdir('gym_duckietown/maps'):
    map_name = map_file.split('.')[0]
    env = SimpleSimEnv(map_name=map_name)
