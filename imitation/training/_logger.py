from concurrent.futures import ThreadPoolExecutor

import os
import pickle

# Stats for ICRA submission

class IILTrainingLogger:

    def __init__(self, env, algorithm, log_file, horizon, episodes, data_file):
        self.env= env
        self.algorithm = algorithm

        self.horizon = horizon
        self.episodes = episodes

        self._log_file = open(log_file, 'wb')
        self._dataset_file = open(data_file, 'wb')
        self._multithreaded_recording = ThreadPoolExecutor(4)
        self._recording = []

        self._configure()

    def _configure(self):
        self.algorithm.on_step_done(self)
        self.algorithm.on_episode_done(self)
        self.algorithm.on_optimization_done(self)
        self.algorithm.on_process_done(self)

    # event handlers
    def step_done(self, observation, action, reward, done, info):
        self._recording.append({
            'state': [
                (self.env.cur_pos, self.env.cur_angle),
                action,
                reward,
                done,
                info,
                self.algorithm.active_uncertainty,
            ],
            'metadata': [
                self.algorithm.expert_queried,
                self.algorithm.active_policy
            ]
        })

    def episode_done(self, episode):
        self._multithreaded_recording.submit(self._dump_recording)
        print('Episode {} completed.'.format(episode))

    def _dump_recording(self):
        pickle.dump(self._recording, self._log_file)
        self._log_file.flush()
        self._recording.clear()

    def optimization_done(self, loss):
        self._multithreaded_recording.submit(self._dump_dataset, loss)

    def _dump_dataset(self, loss):
        pickle.dump([self.algorithm._observations, self.algorithm._expert_actions, loss], self._dataset_file)
        self._dataset_file.flush()

    def process_done(self):
        self._log_file.close()
        self._dataset_file.close()
        os.chmod(self._log_file.name, 0o444) # make file read-only after finishing
        os.chmod(self._dataset_file.name, 0o444) # make file read-only after finishing
        self._multithreaded_recording.shutdown()