import numpy as np
from controllers import Controller
from replay import RecordingController


class ImitationLearningRecorder(RecordingController):

    def __init__(self, env, iil_controller, record_file, horizon, iterations):
        RecordingController.__init__(self, env, iil_controller, record_file=record_file,
                                     stop_on_done=False, stop_on_reset=False)
        self.horizon = horizon
        self.iterations = iterations
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
        RecordingController.step(self, action)
        self.count += 1
        if self.count == self.horizon * self.iterations:
            print('finishing')
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
                    self._controller.secondary.seen_samples
                ])
            })