#!/usr/bin/env python3

import time
import random
from functools import reduce
import operator
from itertools import chain
import math

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

from utils import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def gen_actions(seq_len):
    actions = []

    for i in range(0, seq_len):
        vels = np.random.uniform(low=0.3, high=1.0, size=(2,))
        actions.append(vels)

    return actions

# TODO: try biasing mutations to end
# TODO: try mutating by small adjustments
def mutate_actions(actions):
    actions = actions[:]

    for i in range(0, len(actions)):
        if np.random.uniform(0, 1) < (1 / len(actions)):
            vels = np.random.uniform(low=0.3, high=1.0, size=(2,))
            actions[i] = vels

        if np.random.uniform(0, 1) < (1 / len(actions)):
            vels = actions[i] + np.random.uniform(low=-0.1, high=0.1, size=(2,))
            actions[i] = vels

    return actions

def eval_actions(env, seed, actions):
    env.seed(seed)
    env.reset()

    total_reward = 0

    positions = []

    for i in range(0, len(actions)):
        vels = actions[i]

        positions.append((env.cur_pos, env.cur_angle))

        obs, reward, done, info = env.step(vels)

        total_reward += reward

        if done:
            #print('failed')
            break

    return positions, total_reward

def render_drive(env, seed, actions):
    env.seed(seed)
    env.reset()

    env.render('human')
    time.sleep(0.2)

    for i in range(0, len(actions)):

        vels = actions[i]
        obs, reward, done, info = env.step(vels)

        env.render('human')
        time.sleep(0.12)

        if done:
            #print('failed')
            break

    time.sleep(0.2)


def gen_action_batch(env, seed, num_actions):
    best_actions = gen_actions(num_actions)
    best_r = -math.inf

    env.graphics = False

    for epoch in range(1, 500):

        new_actions = mutate_actions(best_actions)
        positions, r = eval_actions(env, seed, new_actions)

        #print(new_actions)
        #print(r)

        if r > best_r:
            best_r = r
            best_actions = new_actions
            #print('epoch %d, r=%f' % (epoch, r))

    env.graphics = True

    print('r=%f' % best_r)

    return positions, best_actions


env = SimpleSimEnv()

positions = []
actions = []

while True:
    seed = random.randint(0, 0xFFFFFFFF)
    p, a = gen_action_batch(env, seed, 20)

    if len(p) < 20:
        continue

    p = list(map(lambda p: [ p[0].tolist(), p[1] ], p))
    a = list(map(lambda a: a.tolist(), a))

    positions += p
    actions += a

    print('num steps: %d' % len(positions))

    import json
    with open('experiments/data.json', 'w') as outfile:
        json.dump({ 'positions': positions, 'actions':actions }, outfile)





#env.graphics = True
#while True:
#    render_drive(env, best_actions)
