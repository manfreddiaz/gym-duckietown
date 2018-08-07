from controllers import SharedController


class InteractiveImitationLearning(SharedController):
    def __init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle, alpha=0.99):
        SharedController.__init__(self, env, teacher, learner)

        # from IIL
        self._horizon = horizon
        self._alpha = alpha
        self._episodes = episodes

        # data
        self._observations = []
        self._expert_actions = []

        # statistics
        self._expert_interventions = 0
        self._expert_disengagement = 0

        # internal count
        self._current_horizon = 0
        self._current_episode = 0
        self._current_alpha = self._alpha

        # starting pose
        self._starting_position = starting_position
        self._starting_angle = starting_angle

    # execute current control policy
    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()

        control_policy = self._active_policy()
        control_action = control_policy._do_update(observation)

        self._on_expert_input(control_policy, control_action, observation)

        return control_action

    def _on_expert_input(self, control_policy, control_action, observation):
        if control_policy == self.primary:
            expert_action = control_action
        else:
            expert_action = self.primary._do_update(observation)
        if expert_action is not None:
            self._aggregate(observation, expert_action)
        else:
            self._expert_disengagement += 1
            print('give me input you idiot :)')

    def step(self, action):
        if action is not None:
            next_observation, reward, done, info = SharedController.step(self, action)

            if done:
                # self._on_episode_done()
                self.enabled = False
                self.reset()
                self.enabled = True

            self._update_task_boundaries()

            return next_observation, reward, done, info

    def reset(self):
        unwrapped_env = self.env.unwrapped
        unwrapped_env.cur_pos = self._starting_position
        unwrapped_env.cur_angle = self._starting_angle
        self.env.render()

    def _active_policy(self):
        raise NotImplementedError()

    def _update_task_boundaries(self):
        self._current_horizon += 1

        if self._current_horizon > self._horizon:
            self._current_horizon = 0
            self._current_episode += 1
            self._on_episode_done()

        if self._current_episode > self._episodes:
            self._on_process_done()

    def _aggregate(self, observation, action):
        self._observations.append(observation)
        self._expert_actions.append(action)
        self._expert_interventions += 1

    def _learn(self):
        self.enabled = False
        try:
            print('[START] Learning....')
            self.secondary.learn(self._observations, self._expert_actions)
            self.secondary.save()
            self._on_learning_done()
            print('[FINISHED] Learning')
        finally:
            self.enabled = True

    # triggered after an episode of learning is done
    def _on_episode_done(self):
        self._learn()
        print('[FINISHED] Episode: {}/{}'.format(self._current_episode, self._episodes))

    # triggered when the learning step after an episode is done
    def _on_learning_done(self):
        pass

    # triggered when the whole training is done
    def _on_process_done(self):
        self.enabled = False
        print('[DONE] episode: {}/{}, alpha: {}'.format(self._current_episode, self._episodes, self._current_alpha))
