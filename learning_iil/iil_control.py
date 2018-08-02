import numpy as np
from controllers import SharedController


class InteractiveControl(SharedController):
    def __init__(self, env, teacher, learner, horizon, episodes, respawn_positions, alpha=0.99):
        SharedController.__init__(self, env, teacher, learner)

        # from IIL
        self.horizon = horizon
        self.alpha = alpha
        self.episodes = episodes

        # internal count
        self.horizon_count = 0
        self.episodes_count = 0

        # starting pose
        self.respawn_positions = np.array(respawn_positions)

    def _select_respawn(self):
        respawn_index = np.random.randint(0, self.respawn_positions.shape[0])
        respawn_entry = self.respawn_positions[respawn_index]
        self.starting_position = respawn_entry[0:3]
        self.starting_angle = respawn_entry[3]

    # execute current control policy
    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()

        control_policy = self._select_policy()
        control_action = control_policy._do_update(observation)

        return control_action

    def step(self, action):
        if action is not None:
            next_observation, reward, done, info = SharedController.step(self, action)

            if done:
                # self._on_episode_done()
                # self.horizon_count = self.horizon  # done with episode FIXME: when the simulator detects out of lane
                # otherwise is unfair to algorithms that go outside of the lane in the right.
                self.enabled = False
                self.reset()
                self.enabled = True

            self._update_horizon_boundaries()

            return next_observation, reward, done, info

    def reset(self):
        self._select_respawn()
        unwrapped_env = self.env.unwrapped
        unwrapped_env.cur_pos = self.starting_position
        unwrapped_env.cur_angle = self.starting_angle
        self.env.render()

    def _select_policy(self):
        return self.secondary

    def _update_horizon_boundaries(self):
        self.horizon_count += 1

        if self.horizon_count >= self.horizon:
            self.horizon_count = 0
            self.episodes_count += 1
            self._on_episode_done()

    def _on_episode_done(self):
        self.reset()
        print('[FINISHED] Episode: {}/{}'.format(self.episodes_count, self.episodes))

    def _on_episode_learning_done(self):
        pass
