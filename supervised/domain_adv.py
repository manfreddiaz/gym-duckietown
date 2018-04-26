#!/usr/bin/env python3

import time
import random

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

class DomainAdv(nn.Module):
    def __init__(self):
        super().__init__()

        self.vae = VAE()

        self.grad_rev = GradReverse()

        self.categorize = nn.Sequential(
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        recon, mu, log_var = self.vae(x)

        z = self.vae.reparameterize(mu, log_var)

        z = self.grad_rev(z)
        cat = self.categorize(z)

        return recon, mu, log_var, cat

# Load the real images
real_imgs = []
for i in range(0, 1000):
    img = load_img('real_images_2/img_%03d.png' % i)
    real_imgs.append(img)

def gen_data():
    obs = env.reset().copy()
    obs = obs.transpose(2, 0, 1)
    return (obs, 0)

def test_model(model):
    data = gen_batch(gen_data)
    img = data[0][0:1]

    recon, mu, logvar, cat = model(img)

    print(img.size())
    print(recon.size())

    save_img('img_orig.png', img)
    save_img('img_recon.png', recon)

    for i in range(0, 1000):
        img = load_img('real_images_2/img_%03d.png' % i)
        recon, mu, logvar, cat = model(img)
        save_img('real_images_2/img_%03d_recon.png' % i, recon)

if __name__ == "__main__":
    env = SimpleSimEnv()
    env.reset()

    model = DomainAdv()
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    print_model_info(model)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-3
    )

    avg_loss = 0

    num_epochs = 1000000
    for epoch in range(1, num_epochs+1):
        data = gen_batch(gen_data, batch_size = 16)
        img = data[0]

        optimizer.zero_grad()
        recon_batch, mu, logvar, cat = model(img)
        vae_loss = vae_loss_fn(recon_batch, img, mu, logvar)

        fake_img = img[0:1]
        real_img = random.choice(real_imgs)
        zero = Variable(torch.zeros(1)).long().cuda()
        one = Variable(torch.ones(1)).long().cuda()
        img, target = random.choice([(fake_img, zero), (real_img, one)])
        _, _, _, cat_output = model(img)
        class_loss = F.nll_loss(cat_output, target)
        print(class_loss.data[0])

        #total_loss = vae_loss
        total_loss = vae_loss + 0 * class_loss
        total_loss.backward()
        optimizer.step()

        total_loss = total_loss.data[0]
        avg_loss = avg_loss * 0.995 + total_loss * 0.005

        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        if (epoch == 25) or (epoch % 1000 == 0):
            test_model(model)

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), 'trained_models/domain_adv_model.pt')
