from learning_iil.algorithms import DAggerLearning


class SupervisedLearning(DAggerLearning):
    def __init__(self, env, teacher, learner, horizon, episodes):
        DAggerLearning.__init__(self, env, teacher, learner, horizon, episodes)

    def _mix_policy(self):
        return self.primary

    def _on_episode_done(self):
        self.horizon_count = 0
        self.episodes_count += 1
        print('episode: {}/{}'.format(self.episodes_count, self.episodes))

    def _on_training_done(self):
        self.secondary.learn(self.dataset)
        DAggerLearning._on_training_done(self)

