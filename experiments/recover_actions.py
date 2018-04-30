#!/usr/bin/env python3

import time
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

class AutoEnc(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()

        self.z_dim = z_dim

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

        # tanh output so z is in [-1, 1]
        # Do we want just a linear layer instead?
        self.encoder_to_z = nn.Sequential(
            nn.Linear(32 * 8 * 11, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.z_dim),
            #nn.Tanh()
        )

        self.z_to_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32 * 8 * 11),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 32, 5, stride=2, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 3, 6, stride=3),
            nn.LeakyReLU()
        )

        self.apply(init_weights)

    def forward(self, img):
        x = self.encoder(img)
        x = x.view(x.size(0), -1)
        z = self.encoder_to_z(x)

        x = self.z_to_decoder(z)
        x = x.view(x.size(0), -1, 8, 11)
        recon = self.decoder(x)
        recon = recon[:, :, 3:123, 1:161]

        return recon, z

class Model(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()

        self.z_dim = z_dim

        self.autoenc = AutoEnc(z_dim)

        self.z_to_vels = nn.Linear(2 * self.z_dim, 2)

        self.apply(init_weights)

    def forward(self, obs, obs2):
        dec, z0 = self.autoenc(obs)

        dec2, z1 = self.autoenc(obs2)

        z0z1 = torch.cat((z0, z1), dim=1)

        vels = self.z_to_vels(z0z1)

        return dec, dec2, vels





def gen_data():
    obs = env.reset().copy()
    obs = obs.transpose(2, 0, 1)

    # Generate random velocities
    vels = np.random.uniform(low=0.3, high=1.0, size=(2,))
    #vels = np.array([0.8, 0.8])

    obs2, reward, done, info = env.step(vels)
    obs2 = obs2.transpose(2, 0, 1)
    return obs, vels, obs2

def test_model(model):
    obs, vels, obs2 = gen_batch(gen_data)
    img0 = obs[0:1]
    vels = make_var(np.array([0.8, 0.8])).unsqueeze(0)

    dec, dec2, vels = model(img0, img0)

    save_img('img_sim.png', img0)
    save_img('img_dec.png', dec)
    #save_img('img_obs2.png', obs2)

    for i in range(0, 1000):
        try:
            img = load_img('real_images_2/img_%03d.png' % i)
            dec, dec2, vels = model(img, img)
            save_img('real_images_2/img_%03d_recon.png' % i, dec)
        except Exception as e:
            print(e)

def train_loop(model, optimizer, loss_fn, num_epochs):
    avg_loss = 0

    for epoch in range(1, num_epochs+1):
        startTime = time.time()
        obs, vels, obs2 = gen_batch(gen_data, batch_size=8)
        genTime = int(1000 * (time.time() - startTime))

        startTime = time.time()
        optimizer.zero_grad()
        loss = loss_fn(model, obs, vels, obs2)
        loss.backward()
        optimizer.step()
        trainTime = int(1000 * (time.time() - startTime))

        loss = loss.data[0]
        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('gen time: %d ms' % genTime)
        print('train time: %d ms' % trainTime)
        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        if epoch == 100 or epoch % 1000 == 0:
            test_model(model)



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
        #chain(model.encoder.parameters(), model.decoder.parameters()),
        model.parameters(),
        lr=0.001
        #weight_decay=1e-3
    )

    def loss_fn(model, obs, vels, obs2):
        dec, dec2, r_vels = model(obs, obs2)

        dec_loss = (obs - dec).norm(2).mean()
        dec2_loss = (obs2 - dec2).norm(2).mean()
        vel_loss = (r_vels - vels).norm(2).mean()

        print(vel_loss.data[0])

        return dec_loss + dec2_loss + vel_loss

    train_loop(model, optimizer, loss_fn, 120000)
