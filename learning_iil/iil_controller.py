from controllers import SharedController


class InteractiveImitationLearning(SharedController):
    def __init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle, alpha=0.99):
        SharedController.__init__(self, env, teacher, learner)

        # from IIL
        self.horizon = horizon
        self.alpha = alpha
        self.episodes = episodes
        self.observations = []
        self.expert_actions = []

        # internal count
        self.horizon_count = 0
        self.episodes_count = 0
        self.alpha_episode = self.alpha

        # starting pose
        self.starting_position = starting_position
        self.starting_angle = starting_angle

    # execute current control policy
    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()

        control_policy = self._select_policy()
        control_action = control_policy._do_update(observation)

        self._record(control_policy, control_action, observation)

        return control_action

    def _record(self, control_policy, control_action, observation):
        if control_policy == self.primary:
            if control_action is not None:
                self._aggregate(observation, control_action)
        else:
            expert_action = self.primary._do_update(observation)
            if expert_action is not None:
                self._aggregate(observation, expert_action)
            else:
                print('give me input you idiot :)')

    def step(self, action):
        next_observation, reward, done, info = SharedController.step(self, action)

        if done:
            # self._on_episode_done()
            self.enabled = False
            self.reset()
            self.enabled = True

        self._update_horizon_boundaries()

        return next_observation, reward, done, info

    def reset(self):
        unwrapped_env = self.env.unwrapped
        unwrapped_env.cur_pos = self.starting_position
        unwrapped_env.cur_angle = self.starting_angle
        self.env.render()

    def _select_policy(self):
        raise NotImplementedError()

    def _update_horizon_boundaries(self):
        self.horizon_count += 1

        if self.horizon_count == self.horizon:
            self.horizon_count = 0
            self.episodes_count += 1
            self._on_episode_done()
            self._learn()

        if self.episodes_count >= self.episodes:
            self._on_training_done()

    def _aggregate(self, observation, action):
        self.observations.append(observation)
        self.expert_actions.append(action)

    def _on_episode_done(self):
        print('[FINISHED] Episode: {}/{}'.format(self.episodes_count, self.episodes))

    def _learn(self):
        self.enabled = False
        try:
            print('[START] Learning....')
            self.secondary.learn(self.observations, self.expert_actions)
            self.secondary.save()
            self._on_episode_learning_done()
            print('[FINISHED] Learning')
        finally:
            self.enabled = True

    def _on_episode_learning_done(self):
        pass

    def _on_training_done(self):
        self.enabled = False
        print('[DONE] episode: {}/{}, alpha: {}'.format(self.episodes_count, self.episodes, self.alpha_episode))
        self.exit()
