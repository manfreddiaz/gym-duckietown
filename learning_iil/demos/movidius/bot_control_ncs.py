#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a model trained with imitation
learning, and visualize the result.
"""

import time
import numpy as np

from learning_iil.demos.movidius.bot_differential_env import DifferentialDuckiebotEnv
from learning_iil.learners import NeuralNetworkPolicy, UANeuralNetworkPolicy
from learning_iil.learners.parametrizations.tf.uncertainty import MonteCarloDropoutResnetOneRegression


env = DifferentialDuckiebotEnv()
env.max_steps = np.inf

obs = env.reset()
env.render()

avg_frame_time = 0
max_frame_time = 0

tf_model = MonteCarloDropoutResnetOneRegression()
tf_controller = UANeuralNetworkPolicy(env=env,
                                      parametrization=tf_model,
                                      storage_location='trained_models/sim2real/',
                                      training=False)
while True:
    start_time = time.time()

    # obs = cv2.resize(obs, (80, 60), interpolation=cv2.INTER_NEAREST)

    obs = np.flipud(obs)
    # cv2.imwrite('output.jpeg', obs)
    # obs = obs.transpose(2, 0, 1)

    vels = tf_controller.predict(observation=obs)
    print(vels)

    if vels[1] < .5:
        print('here: {}'.format(vels[1]))
    obs, reward, done, info = env.step(vels[0])
    #print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))


    env.render()


    end_time = time.time()
    frame_time = 1000 * (end_time - start_time)
    avg_frame_time = avg_frame_time * 0.95 + frame_time * 0.05
    max_frame_time = 0.99 * max(max_frame_time, frame_time) + 0.01 * frame_time
    fps = 1 / (frame_time / 1000)

    print('avg frame time: %d' % int(avg_frame_time))
    print('max frame time: %d' % int(max_frame_time))
    print('fps: %.1f' % fps)
