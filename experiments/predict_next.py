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

def init_weights(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        nn.init.orthogonal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, enc_size):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 6, stride=3)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)

        self.linear1 = nn.Linear(32 * 8 * 11, 256)
        self.linear2 = nn.Linear(256, enc_size)

    def forward(self, img):
        batch_size = img.size(0)

        x = img

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        conv_out = F.leaky_relu(x)

        #print(conv_out.size())

        conv_out = conv_out.view(batch_size, -1)
        x = F.leaky_relu(self.linear1(conv_out))
        mid = F.leaky_relu(self.linear2(x))

        return mid

class Decoder(nn.Module):
    def __init__(self, enc_size):
        super().__init__()

        self.linear1 = nn.Linear(enc_size, 256)
        self.linear2 = nn.Linear(256, 32 * 8 * 11)

        self.deconv1 = nn.ConvTranspose2d(32, 32, 5, stride=2, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 5, stride=2, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 6, stride=3)

    def forward(self, enc):
        batch_size = enc.size(0)

        x = F.leaky_relu(self.linear1(enc))
        x = F.leaky_relu(self.linear2(x))
        x = x.view(batch_size, 32, 8, 11)

        x = self.deconv1(x)
        x = F.leaky_relu(x)

        x = self.deconv2(x)
        x = F.leaky_relu(x)

        x = self.deconv3(x)
        x = F.leaky_relu(x)

        #print(x.size())
        x = x[:, :, 3:123, 1:161]
        #print(x.size())

        return x

class Predictor(nn.Module):
    def __init__(self, enc_size):
        super().__init__()

        self.enc_linear = nn.Linear(enc_size, 96)
        self.vel_linear = nn.Linear(2, 32)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, enc_size)

    def forward(self, enc, vels):
        x0 = F.leaky_relu(self.enc_linear(enc))
        x1 = F.leaky_relu(self.vel_linear(vels))
        x = torch.cat((x0, x1), dim=1)
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))

        return x

class Model(nn.Module):
    def __init__(self, enc_size=8):
        super().__init__()

        self.encoder = Encoder(enc_size)
        self.decoder = Decoder(enc_size)
        self.predictor = Predictor(enc_size)

        self.apply(init_weights)

    def forward(self, obs, vels):
        batch_size = obs.size(0)

        enc = self.encoder(obs)
        dec = self.decoder(enc)

        enc2 = self.predictor(enc, vels)
        dec2 = self.decoder(enc2)

        return dec, dec2

    def print_info(self):
        modelSize = 0
        for p in self.parameters():
            pSize = reduce(operator.mul, p.size(), 1)
            modelSize += pSize
        print(str(self))
        print('Total model size: %d' % modelSize)

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))

def gen_data():
    obs = env.reset().copy()
    obs = obs.transpose(2, 0, 1)

    # Generate random velocities
    #vels = np.random.uniform(low=0.3, high=1.0, size=(2,))
    vels = np.array([0.8, 0.8])

    obs2, reward, done, info = env.step(vels)
    obs2 = obs2.transpose(2, 0, 1)

    return obs, vels, obs2

def test_model(model):
    obs, vels, obs2 = gen_batch(gen_data)
    img0 = obs[0:1]
    vels = make_var(np.array([0.8, 0.8])).unsqueeze(0)

    dec, obs2 = model(img0, vels)

    save_img('seg_img.png', img0)
    save_img('img_dec.png', dec)
    save_img('img_obs2.png', obs2)

    for i in range(0, 180):
        try:
            img = load_img('real_images/img_%03d.png' % i)
            _, out = model(img, vels)
            save_img('real_images/img_%03d_recon.png' % i, out)
        except Exception as e:
            print(e)

def train_loop(model, optimizer, loss_fn, num_epochs):
    avg_loss = 0

    for epoch in range(1, num_epochs+1):
        startTime = time.time()
        obs, vels, obs2 = gen_batch(gen_data)
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

        #if epoch % 1000 == 0:
        #    model.save('trained_models/angle_model.pt')

if __name__ == "__main__":
    env = SimpleSimEnv()
    env.reset()

    model = Model()
    model.print_info()
    if torch.cuda.is_available():
        model.cuda()

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        chain(model.encoder.parameters(), model.decoder.parameters()),
        lr=0.001
        #weight_decay=1e-3
    )

    def autoenc_loss(model, obs, vels, obs2):
        dec, obs2 = model(obs, vels)
        return (obs - dec).norm(2).mean()

    train_loop(model, optimizer, autoenc_loss, 120000)

    optimizer = optim.Adam(
        model.predictor.parameters(),
        lr=0.001
        #weight_decay=1e-3
    )

    # FIXME: temporarily testing pure reconstruction
    def obs2_loss(model, obs, vels, obs2):
        dec, dec2 = model(obs, vels)
        return (dec2 - obs2).norm(2).mean()

    train_loop(model, optimizer, obs2_loss, 120000)
