#!/usr/bin/env python3

import time
import random
from functools import reduce
import operator
from itertools import chain

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

from utils import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        # tanh output so z is in [-1, 1]
        # Do we want just a linear layer instead?
        self.convs_to_actions = nn.Sequential(
            nn.Linear(32 * 8 * 11, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 32),
            nn.LeakyReLU(),

            nn.Linear(32, 2),
            nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, img):
        x = self.convs(img)

        x = x.view(x.size(0), -1)

        actions = self.convs_to_actions(x)

        # Idea: add recon as auxiliary task?

        return actions












def gen_trajectory(env, len=5):

    obs = env.reset()

    total_reward = 0

    for i in range(0, len):

        obs = obs.transpose(2, 0, 1)

        vels = np.random.uniform(low=0.3, high=1.0, size=(2,))

        if i == 0:
            first_obs = obs
            first_vels = vels

        obs, reward, done, info = env.step(vels)

        total_reward += reward

    return (first_obs, first_vels, total_reward)

def gen_data(num_trajs=10):

    observs = []
    actions = []
    rewards = []

    #var = make_var(np.stack(array))

    for i in range(0, num_trajs):
        print(i)

        obs, vels, reward = gen_trajectory(env)

        observs.append(obs)
        actions.append(vels)
        rewards.append(reward)

    observs = make_var(np.stack(observs))
    actions = make_var(np.stack(actions))

    rewards = np.array(rewards).reshape((num_trajs, 1))
    rewards = make_var(rewards)

    return observs, actions, rewards



def eval_model(model, env, num_trajs=20, len=10):
    env.seed(0)

    total_reward = 0

    for i in range(0, num_trajs):

        obs = env.reset()

        for step in range(0, len):

            obs = obs.transpose(2, 0, 1)
            obs = make_var(obs).unsqueeze(0)

            vels = model(obs)
            vels = vels.squeeze(0).cpu()
            vels = vels.data.numpy()

            obs, reward, done, info = env.step(vels)

            total_reward += reward

    return total_reward






if __name__ == "__main__":
    env = SimpleSimEnv()
    env.reset()

    model = Model()
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    print_model_info(model)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001
        #weight_decay=1e-3
    )

    num_trajs = 10000
    observs, actions, rewards = gen_data(num_trajs)


    num_epochs = 120000
    for epoch in range(1, num_epochs+1):

        # Sample a batch
        idx = random.randint(0, num_trajs-1 - 8)
        obs = observs[idx:idx+8]
        target_vels = actions[idx:idx+8]
        r = rewards[idx:idx+8]


        vels = model(obs)

        loss = (r * (vels - target_vels).norm(2)).mean()



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if epoch % 200 == 0:
            #print('epoch %d' % epoch)

            r = eval_model(model, env)

            print('epoch %d, r = %f' % (epoch, r))
