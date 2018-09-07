from concurrent.futures import ThreadPoolExecutor

import os
import pickle

# Stats for ICRA submission

class Logger:
    def __init__(self, env, routine, horizon, episodes, log_file):
        self.env = env
        self.routine = routine

        self.horizon = horizon
        self.episodes = episodes

        self._log_file = open(log_file, 'wb')
        self._multithreaded_recording = ThreadPoolExecutor(4)
        self._recording = []
        self._configure()

    def _configure(self):
        self.routine.on_step_done(self)
        self.routine.on_episode_done(self)
        self.routine.on_process_done(self)

    def step_done(self, observation, action, reward, done, info):
        self._recording.append({
            'state': [
                (self.env.cur_pos, self.env.cur_angle),
                action,
                reward,
                done,
                info
            ],
            'metadata': [
                self.routine.teacher_queried,  # 0
                self.routine.teacher_action,  # 1
                self.routine.teacher_uncertainty,  # 2
                self.routine.learner_action,
                self.routine.learner_uncertainty,
                self.routine.active_policy,  # 3
            ]
        })

    def episode_done(self, episode):
        self._multithreaded_recording.submit(self._commit)
        # print('Episode {} completed.'.format(episode))

    def _commit(self):
        pickle.dump(self._recording, self._log_file)
        self._log_file.flush()
        self._recording.clear()

    def process_done(self):
        self._log_file.close()
        os.chmod(self._log_file.name, 0o444)  # make file read-only after finishing
        self._multithreaded_recording.shutdown()

class IILTrainingLogger(Logger):

    def __init__(self, env, routine, horizon, episodes, log_file, data_file):
        Logger.__init__(self, env, routine, horizon, episodes, log_file)
        self._dataset_file = open(data_file, 'wb')

    def _configure(self):
        self.routine.on_optimization_done(self)
        Logger._configure(self)

    def optimization_done(self, loss):
        self._multithreaded_recording.submit(self._dump_dataset, loss)

    def _dump_dataset(self, loss):
        pickle.dump([self.routine._observations, self.routine._expert_actions, loss], self._dataset_file)
        self._dataset_file.flush()

    def process_done(self):
        self._dataset_file.close()
        os.chmod(self._dataset_file.name, 0o444) # make file read-only after finishing
        Logger.process_done(self)
