import pickle

from controllers import Controller


class ReplayController(Controller):
    def __init__(self, env, record_file):
        self.episode_counter = 0
        self.current_episode = None
        self.current_episode_step = 0
        self.current_episode_size = 0

        self.recording_file = None
        self.recording_file_name = record_file

        Controller.__init__(self, env)

    def configure(self):
        self.recording_file = open(self.recording_file_name, 'rb')

    def _do_update(self, dt):
        if self.current_episode is None:
            try:
                self.current_episode = pickle.load(self.recording_file)
                self.current_episode_step = 0
                self.current_episode_size = len(self.current_episode)
                self.episode_counter += 1
                print('[RUNNING] episode {}, horizon: {}'.format(self.episode_counter, self.current_episode_size))
            except EOFError:
                self.enabled = False
                print('[FINISHED] {} episodes shown'.format(self.episode_counter))

        if self.current_episode_step < self.current_episode_size:

            sample = self.current_episode[self.current_episode_step]
            unwrapped_env = self.env.unwrapped

            unwrapped_env.cur_pos = sample['hidden'][0]
            unwrapped_env.cur_angle = sample['hidden'][1]

            self.current_episode_step += 1

            self.env.render()
        else:
            self.current_episode = None
