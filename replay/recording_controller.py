import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from controllers import Controller


class RecordingController(Controller):

    def __init__(self, env, controller, record_file, stop_on_done=True, stop_on_reset=True):
        # recording configuration
        self._recording = False
        self._controller = controller
        self._record_file = open(record_file, 'wb')
        self._multithreaded_recording = ThreadPoolExecutor(4)

        self._episodes_counter = 0
        self._episodes_current = []

        # flags
        self.stop_on_done = stop_on_done  # makes the recording stop upon reception of done
        self.stop_on_reset = stop_on_reset  # makes the recording stop when the environment is reset

        Controller.__init__(self, env)

    def _do_update(self, dt):
        return self._controller._do_update(dt)

    def configure(self):
        capabilities = {
            'record': self.record,
            'stop': self.stop
        }
        Controller.extend_capabilities(self, self._controller, capabilities)
        self._controller.configure()

    # here we save global metadata about the environment
    def _store_metadata(self):
        pass

    # auto stop recording on reset
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.stop_on_reset:
            self._stop()

    def step(self, action):
        if action is not None:
            observation = self.env.unwrapped.render_obs()
            result = self._controller.step(action)
            self._multithreaded_recording.submit(self._record, observation, action, result)
            return result

        return None

    def _on_data_added(self, what):
        print('data')

    # extended capability
    def record(self, _):
        if not self._recording:
            self._episodes_counter += 1
            self._recording = True
            print('[RECORDING] Episode {}'.format(self._episodes_counter))

    # extended capability
    def stop(self, _):
        self._stop()

    def _record(self, observation, action, result):

        if self._recording:
            next_observation, reward, done, info = result

            unwrapped_env = self.env.unwrapped
            # record for each timestep
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
                    unwrapped_env.step_count
                ])
            })
            # auto stop at done
            if done and self.stop_on_done:
                self._stop()

    def _stop(self):
        print(len(self._episodes_current))
        if len(self._episodes_current) > 0:
            pickle.dump(self._episodes_current, self._record_file)
            print('[RECORDED] Episode {}, Horizon: {}'.format(self._episodes_counter, len(self._episodes_current)))
            self._episodes_current.clear()
        self._recording = False

    def close(self):
        self._record_file.close()
        self._controller.close()
        self._multithreaded_recording.shutdown()
