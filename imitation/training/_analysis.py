import pickle
import matplotlib.pyplot as plt

from imitation.training._settings import *
from imitation.training._summary import *

def summarize_iteration(disk_entry):
    reading = True
    data_file = open(disk_entry, mode='rb')

    iteration_summary = IterationSummary()
    while reading:
        try:
            episode_data = pickle.load(data_file)
            summary = EpisodeSummary()
            for data in episode_data:
                summary.process(data)
            iteration_summary.add_episode_summary(summary)
        except EOFError:
            reading = False

    return iteration_summary

def plot_iteration_per_episode(iteration_summary, disk_entry):
    episode_range = np.arange(0, iteration_summary.episodes())
    plt.plot(episode_range, iteration_summary.out_bounds_history())
    plt.show()

if __name__ == '__main__':
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 0
    optimization_iteration = 2
    algorithm = 'supervised'

    disk_entry = experimental_entry(
        algorithm=algorithm,
        experiment_iteration=iteration,
        selected_parametrization=parametrization_iteration,
        selected_horizon=horizon_iteration,
        selected_episode=horizon_iteration,
        selected_optimization=optimization_iteration
    )

    summary = summarize_iteration(disk_entry + 'training.log')
    plot_iteration_per_episode(summary, None)
