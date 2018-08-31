import numpy as np

class Summary:
    def __init__(self):
        self._reward = 0.0
        self._penalties = 0.0
        self._queries = 0
        self._out_bounds = 0

class EpisodeSummary(Summary):
    def __init__(self):
        Summary.__init__(self)

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


class IterationSummary(Summary):
    def __init__(self):
        Summary.__init__(self)
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

    def episodes(self):
        return len(self._episodes)