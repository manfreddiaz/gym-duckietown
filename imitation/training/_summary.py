import numpy as np

class Summary:
    def __init__(self, label):
        self.label = label
        self._reward = 0.0
        self._penalties = 0.0
        self._queries = 0
        self._out_bounds = 0
        self._delta_v = 0
        self._delta_theta = 0

class EpisodeSummary(Summary):
    def __init__(self, label):
        Summary.__init__(self, label)

    def process(self, entry):
        state = entry['state']
        metadata = entry['metadata']
        # reward
        reward = state[2]
        self._reward += reward
        # penalties
        if reward < 0.0:
            self._penalties += reward

        if state[3] and reward == -1000:
            self._out_bounds += 1

        if metadata[0]:
            self._queries += 1

        if metadata[1] is not None:
            # print(state[1], metadata[1])
            self._delta_v += state[1][0] - metadata[1][0]
            self._delta_theta += state[1][1] - metadata[1][1]
            # print(self._delta_theta)
        # print(self._delta_v)


class IterationSummary(Summary):
    def __init__(self, label):
        Summary.__init__(self, label)
        self._episodes = []

    def add_episode_summary(self, episode_summary : EpisodeSummary):
        self._episodes.append(episode_summary)

    def reward(self):
        if self._reward == 0.0:
            for episode_summary in self._episodes:
                self._reward += episode_summary._reward

        return self._reward

    def reward_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._reward)

        return np.array(history)

    def penalties(self):
        if self._penalties == 0.0:
            for episode_summary in self._episodes:
                self._penalties += episode_summary._penalties

        return self._penalties

    def penalties_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._penalties)

        return np.array(history)

    def queries(self):
        if self._queries == 0:
            for episode_summary in self._episodes:
                self._queries += episode_summary._queries
        return self._queries

    def queries_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._queries)

        return np.array(history)

    def out_bounds(self):
        if self._out_bounds == 0:
            for episode_summary in self._episodes:
                self._out_bounds += episode_summary._out_bounds
        return self._out_bounds

    def out_bounds_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._out_bounds)

        return np.array(history)

    def delta_v(self):
        if self._delta_v == 0:
            for episode_summary in self._episodes:
                self._delta_v += episode_summary._delta_v
        return self._delta_v

    def delta_v_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._delta_v)

        return np.array(history)

    def delta_theta(self):
        if self._delta_theta == 0:
            for episode_summary in self._episodes:
                self._delta_theta += episode_summary._delta_theta
        return self._delta_theta

    def delta_theta_history(self):
        history = []
        for episode_summary in self._episodes:
            history.append(episode_summary._delta_theta)

        return np.array(history)

    def episodes(self):
        return len(self._episodes)