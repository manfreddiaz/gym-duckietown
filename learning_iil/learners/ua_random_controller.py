import math
import numpy as np
from controllers import Controller


class UncertaintyAwareRandomController(Controller):

    def _do_update(self, dt):
        return self.predict(dt)

    def __init__(self, env):
        Controller.__init__(self, env)

    def predict(self, observation):
        return np.random.uniform(0, 1, 2), 0

    def learn(self, observations, actions):
        pass

    def save(self):
        print('I didnt learn a thing...')
