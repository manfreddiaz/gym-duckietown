
class InteractiveImitationLearning:
    def __init__(self, env, teacher, learner, horizon, episodes):

        self.environment = env
        self.teacher = teacher
        self.learner = learner

        # from IIL
        self._horizon = horizon
        self._episodes = episodes

        # data
        self._observations = []
        self._expert_actions = []

        # statistics
        self._expert_interventions = 0

        # internal count
        self._current_horizon = 0
        self._current_episode = 0

    def train(self, samples=1):
        for episode in range(self._episodes):
            self._current_episode = episode
            self._sampling(samples)
            self._optimize() # episodic learning
            self._on_episode_done()
        self._on_process_done()

    def _sampling(self, samples):
        observation = self.environment.render_obs()
        for sample in range(samples):  # number of T-step trajectories
            for horizon in range(self._horizon):
                self._current_horizon = horizon
                action = self._act(observation)
                observation, reward, done, info = self.environment.step(action)
        self._on_sampling_done()

    # execute current control policy
    def _act(self, observation):
        if self._current_episode == 0:  # initial policy equals expert's
            control_policy = self.teacher
        else:
            control_policy = self._mix()

        control_action = control_policy.predict(observation)

        self._query_expert(control_policy, control_action, observation)

        return control_action

    def _query_expert(self, control_policy, control_action, observation):
        if control_policy == self.teacher:
            expert_action = control_action
        else:
            expert_action = self.teacher.predict(self._current_episode)

        if isinstance(expert_action, tuple):
            expert_action, _ = expert_action # if we have uncertainty as input, we do not record it

        if expert_action is not None:
            self._aggregate(observation, expert_action)
            self._expert_interventions += 1

    def _mix(self):
        raise NotImplementedError()

    def _aggregate(self, observation, action):
        self._observations.append(observation)
        self._expert_actions.append(action)

    def _optimize(self):
        self.learner.optimize(self._observations, self._expert_actions)
        self.learner.save()
        self._on_learning_done()

    # triggered after an episode of learning is done
    def _on_episode_done(self):
        raise NotImplementedError()

    # triggered when the learning step after an episode is done
    def _on_learning_done(self):
        raise NotImplementedError()

    # triggered when the whole training is done
    def _on_process_done(self):
        raise NotImplementedError()

    def _on_sampling_done(self):
        raise NotImplementedError()
