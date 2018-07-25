import numpy as np
from controllers import SharedController


class DAggerLearning(SharedController):
    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.99):
        SharedController.__init__(self, env, teacher, learner)

        # from IIL
        self.horizon = horizon
        self.alpha = alpha
        self.episodes = episodes

        # internal count
        self.horizon_count = 0
        self.episodes_count = 0
        self.alpha_episode = self.alpha

        self.dataset = []

    # execute current control policy
    def _do_update(self, dt):
        control_policy = self._mix_policy()
        # TODO: Some performance/clarity may be added below
        self._aggregate(self.env.unwrapped.render_obs(), self.primary._do_update(dt))
        return control_policy._do_update(dt)

    def step(self, action):
        next_observation, reward, done, info = SharedController.step(self, action)
        self._update_mixture_policy()

        if done:
            self._on_episode_done()
            self.reset()

        return next_observation, reward, done, info

    def _mix_policy(self):
        random_control_policy = np.random.choice([self.primary, self.secondary], p=[self.alpha, 1. - self.alpha])
        return random_control_policy

    def _update_mixture_policy(self):
        self.horizon_count += 1
        if self.horizon_count == self.horizon:
            self._on_episode_done()

        if self.episodes_count == self.episodes:
            self._on_training_done()

    def _aggregate(self, observation, action):
        self.dataset.append([observation, action])

    def _on_episode_done(self):
        self.horizon_count = 0
        self.episodes_count += 1
        self.alpha_episode = self.alpha_episode ** self.episodes_count
        self.secondary.learn(self.dataset)
        print('episode: {}/{}, alpha: {}'.format(self.episodes_count, self.episodes, self.alpha_episode))
        self._on_learning_done()

    def _on_learning_done(self):
        pass

    def _on_training_done(self):
        self.enabled = False
        print('[DONE] episode: {}/{}, alpha: {}'.format(self.episodes_count, self.episodes, self.alpha_episode))
        self.exit()
