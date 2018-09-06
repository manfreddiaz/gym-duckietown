import pickle


from imitation.training._settings import *
from imitation.training._summary import *
from imitation.training._parametrization import PARAMETRIZATIONS_NAMES
from imitation.training._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES

def summarize_iteration(disk_entry):
    import matplotlib.pyplot as plt

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

def plot_iteration_per_episode(iteration_summary):
    import matplotlib.pyplot as plt
    episode_range = np.arange(0, iteration_summary.episodes())
    plt.plot(episode_range, iteration_summary.reward_history())
    plt.show()


if __name__ == '__main__':
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 0
    optimization_iteration = 3
    learning_rate_iteration = 0

    algorithm = 'supervised'

    disk_entry = experimental_entry(
        algorithm=algorithm,
        experiment_iteration=iteration,
        parametrization_name=PARAMETRIZATIONS_NAMES[parametrization_iteration],
        horizon=HORIZONS[horizon_iteration],
        episodes=EPISODES[horizon_iteration],
        optimization_name=OPTIMIZATION_METHODS_NAMES[optimization_iteration],
        learning_rate=LEARNING_RATES[learning_rate_iteration]
    )

    summary = summarize_iteration(disk_entry + 'testing.log')

    plot_iteration_per_episode(summary)

    # v_histograms, theta_histogram = summarize_dataset(disk_entry + 'dataset_evolution.pkl')

    # plot_histograms(theta_histogram)