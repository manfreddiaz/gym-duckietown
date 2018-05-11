#!/usr/bin/env python3

import time
import random
from functools import reduce
import operator
from itertools import chain
import math
import json

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

        self.encoder = nn.Sequential(
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

        self.enc_to_vels = nn.Sequential(
            nn.Linear(32 * 8 * 11, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            #nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, img):
        x = self.encoder(img)
        x = x.view(x.size(0), -1)
        vels = self.enc_to_vels(x)

        return vels

def load_data():
    global positions
    global actions

    with open('experiments/data.json') as f:
        data = json.load(f)
    positions = data['positions']
    actions = data['actions']

def gen_data():
    idx = random.randint(0, len(positions) - 1)
    cur_pos = np.array(positions[idx][0])
    cur_angle = positions[idx][1]
    vels = np.array(actions[idx])

    env.cur_pos = cur_pos
    env.cur_angle = cur_angle

    obs = env._render_obs().copy()
    obs = obs.transpose(2, 0, 1)

    return obs, vels

if __name__ == "__main__":
    load_data()

    env = SimpleSimEnv()
    env2 = SimpleSimEnv()

    model = Model()
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    print_model_info(model)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        #chain(model.encoder.parameters(), model.decoder.parameters()),
        model.parameters(),
        lr=0.001,
        weight_decay=1e-3
    )

    avg_loss = 0
    num_epochs = 2000000

    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()

        env.reset()
        obs, vels = gen_batch(gen_data)

        model_vels = model(obs)

        loss = (model_vels - vels).norm(2).mean()
        loss.backward()
        optimizer.step()

        loss = loss.data[0]
        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        #print('gen time: %d ms' % genTime)
        #print('train time: %d ms' % trainTime)

        if epoch % 200 == 0:
            torch.save(model.state_dict(), 'trained_models/imitate.pt')

        if epoch % 200 == 0:
            load_data()

        if epoch % 4 != 0:
            continue

        obs2 = env2._render_obs()
        obs2 = obs2.transpose(2, 0, 1)
        obs2 = make_var(obs2).unsqueeze(0)

        vels = model(obs2)
        vels = vels.squeeze()
        vels = vels.data.cpu().numpy()
        #print(vels)

        obs, reward, done, info = env2.step(vels)

        env2.render('human')

        if done:
            print('failed')
            env2.reset()

        time.sleep(0.2)
