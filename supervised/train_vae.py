#!/usr/bin/env python3

import time

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

from utils import *
from vae import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def gen_data():
    obs = env.reset().copy()
    obs = obs.transpose(2, 0, 1)

    # Generate random velocities
    #vels = np.random.uniform(low=0.3, high=1.0, size=(2,))
    #vels = np.array([0.8, 0.8])

    #obs2, reward, done, info = env.step(vels)
    #obs2 = obs2.transpose(2, 0, 1)

    return (obs,)

def test_model(model):
    data = gen_batch(gen_data)
    img = data[0][0:1]

    recon, mu, logvar = model(img)

    print(img.size())
    print(recon.size())

    save_img('img_sim.png', img)
    save_img('img_dec.png', recon)

    for i in range(0, 180):
        try:
            img = load_img('real_images/img_%03d.png' % i)
            recon, mu, logvar = model(img)
            save_img('real_images/img_%03d_recon.png' % i, recon)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    env = SimpleSimEnv()
    env.reset()

    model = VAE()
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    print_model_info(model)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
        #weight_decay=1e-3
    )



    avg_loss = 0

    num_epochs = 1000000
    for epoch in range(1, num_epochs+1):
        data = gen_batch(gen_data, batch_size = 16)
        img = data[0]

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = vae_loss_fn(recon_batch, img, mu, logvar)
        loss.backward()
        optimizer.step()

        loss = loss.data[0]
        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        if epoch == 100 or epoch % 1000 == 0:
            test_model(model)
