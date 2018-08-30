import numpy as np
from controllers import Controller
from replay import RecordingController


class IILTrainingLogger():

    def __init__(self, env, algorithm, log_file, horizon, episodes):
        RecordingController.__init__(self, env, algorithm, record_file=log_file,
                                     stop_on_done=False, stop_on_reset=False)
        self.horizon = horizon
        self.episodes = episodes
        self.count = 0

    def configure(self):
        capabilities = {
            'record': self.record,
            'stop': self.stop
        }
        Controller.extend_capabilities(self, self._controller.primary, capabilities)
        self._controller.configure()

    def reset(self, **kwargs):
        self._controller.reset(**kwargs)

    def step(self, action):
        if action is not None:
            RecordingController.step(self, action)
            self.count += 1
            if self.count == self.horizon * self.episodes:
                print('[DONE] Recording')
                self.stop(None)
                self.exit()

    def _record(self, observation, action, result):
        if self._recording:
            next_observation, reward, done, info = result
            unwrapped_env = self.env.unwrapped
            self._episodes_current.append({
                'agent': np.array([
                    observation,
                    action
                ]),
                'env': np.array([
                    next_observation,
                    reward,
                    done,
                    info
                ]),
                'hidden': np.array([
                    unwrapped_env.cur_pos,
                    unwrapped_env.cur_angle,
                    unwrapped_env.step_count,
                    unwrapped_env._proximity_penalty(),
                    self._controller._expert_interventions,
                    self._controller._expert_disengagement
                ])
            })