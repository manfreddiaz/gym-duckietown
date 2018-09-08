import math
import numpy as np
from controllers import Controller


class UARandomExploration:

    def __init__(self, uncertainty=math.inf):
        self.uncertainty = uncertainty

    def _do_update(self, dt):
        return self.predict(dt)

    def predict(self, observation):
        v = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, math.pi)
        return  np.array([v, theta]), self.uncertainty

    def learn(self, observations, actions):
        pass

    def save(self):
        print('I didn\'t learn a thing...')

    def reset(self):
        pass
