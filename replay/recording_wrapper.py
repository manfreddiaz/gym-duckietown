import pickle
import numpy as np
import gym


class RecordingWrapper(gym.Wrapper):

    def __init__(self, env, record_file, stop_on_done=True, stop_on_reset=True):
        gym.Wrapper.__init__(self, env)
        # default
        self.stop_on_done = stop_on_done  # makes the recording stop upon reception of done
        self.stop_on_reset = stop_on_reset  # makes the recording stop when the environment is reset

        # recording configuration
        self._recording = False
        self._episodes_counter = 0
        self._episode_current_dataset = []
        self._initialize_(record_file)

    def _initialize_(self, record_file):
        self._record_file = open(record_file, 'wb')


    # here we save global metadata about the environment
    def _store_metadata(self):
        pass

    # auto stop recording on reset
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.stop_on_reset:
            self._stop()

    def step(self, action):
        observation = self.env.unwrapped.render_obs()
        result = self.env.step(action)
        self._record(observation, action, result)
        return result

    # could be parallel at some point
    def _record(self, observation, action, result):
        next_observation, reward, done, info = result

        if not self._recording:
            self._episodes_counter += 1
            self._recording = True

        unwrapped_env = self.env.unwrapped
        self._episode_current_dataset.append({
            'external': np.array([observation, action, next_observation, reward, done, info]),
            'internal': np.array([unwrapped_env.cur_pos, unwrapped_env.cur_angle])
        })

        # auto stop at done
        if done and self.stop_on_done:
            self._stop()

    def _stop(self):
        pickle.dump(self._episode_current_dataset, self._record_file)
        self._episode_current_dataset.clear()
        self._recording = False

    def replay(self, record_file):
        pass

    def close(self):
        self._record_file.close()

    # add a transparent wrapper
    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self.env.unwrapped
