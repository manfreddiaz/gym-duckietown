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
            nn.Linear(256, 2),
            #nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, img):
        x = self.encoder(img)
        x = x.view(x.size(0), -1)
        vels = self.enc_to_vels(x)

        return vels








seed = 3

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

def eval_actions(env, actions):
    env.seed(seed)
    env.reset()

    total_reward = 0

    for i in range(0, len(actions)):
        vels = actions[i]

        obs, reward, done, info = env.step(vels)

        total_reward += reward

        if done:
            #print('failed')
            break

    return total_reward

def render_drive(env, actions):
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




env = SimpleSimEnv()

best_actions = gen_actions(30)
best_r = -math.inf

env.graphics = False

for epoch in range(1, 1000):

    new_actions = mutate_actions(best_actions)
    r = eval_actions(env, new_actions)

    #print(new_actions)
    #print(r)

    if r > best_r:
        best_r = r
        best_actions = new_actions
        print('epoch %d, r=%f' % (epoch, r))

env.graphics = True




obss = []
env.seed(seed)
obs = env.reset()
for vels in best_actions:
    obss.append(obs.transpose(2, 0, 1))
    obs, reward, done, info = env.step(vels)
assert len(obss) == len(best_actions)








model = Model()
model.train()
if torch.cuda.is_available():
    model.cuda()
print_model_info(model)

# weight_decay is L2 regularization, helps avoid overfitting
optimizer = optim.Adam(
    #chain(model.encoder.parameters(), model.decoder.parameters()),
    model.parameters(),
    lr=0.001
    #weight_decay=1e-3
)

avg_loss = 0
num_epochs = 20000
for epoch in range(1, num_epochs+1):
    print(epoch)

    optimizer.zero_grad()



    idx = random.randint(0, len(best_actions)-1)

    obs = make_var(obss[idx]).unsqueeze(0)
    vels = make_var(best_actions[idx]).unsqueeze(0)


    model_vels = model(obs)


    loss = (model_vels - vels).norm(2).mean()
    loss.backward()
    optimizer.step()

    loss = loss.data[0]
    avg_loss = avg_loss * 0.995 + loss * 0.005

    print('epoch %d, loss=%.3f' % (epoch, avg_loss))

    #loss = loss.data[0]
    #avg_loss = avg_loss * 0.995 + loss * 0.005

    #print('gen time: %d ms' % genTime)
    #print('train time: %d ms' % trainTime)
    #print('epoch %d, loss=%.3f' % (epoch, avg_loss))

    #if epoch == 100 or epoch % 1000 == 0:
    #    test_model(model)








#while True:
#    render_drive(env, best_actions)
