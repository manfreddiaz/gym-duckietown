#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a model trained with imitation
learning, and visualize the result.
"""

# pass: 6ol4mv8&3


import time
import numpy as np

from learning_iil.demos.movidius.bot_differential_env import DifferentialDuckiebotEnv
from learning_iil.learners import UncertaintyAwareNNController
from learning_iil.learners.models.tf.uncertainty import MonteCarloDropoutResnetOneRegression

# declare remote Duckiebot control environment
env = DifferentialDuckiebotEnv()
env.max_steps = np.inf

# policy parametrization and agent
model = MonteCarloDropoutResnetOneRegression()
controller = UncertaintyAwareNNController(env=env,
                                          learner=model,
                                          storage_location='trained_models/sim2real/',
                                          training=False)

observation = env.reset()
env.render()

if __name__ == '__main__':
    print('Environment running...')
    while True:
        start_time = time.time()

        observation = np.flipud(observation)

        action = controller.predict(observation=observation)

        observation, reward, done, info = env.step(action[0])

        env.render()
