# experiments configurations for submissions, define behavior of the environment after each episode
from ._settings import simulation

class Icra2019Driver:

    def __init__(self, env, at, routine):
        self.env = env
        self.at = at

        self.routine = routine
        self.routine.on_step_done(self)

    def restart(self):
        simulation(self.at, self.env, reset=False)

    def reset(self):
        simulation(self.at, self.env, reset=True)

    def step_done(self, observation, action, reward, done, info):
        if done:
            self.restart()
