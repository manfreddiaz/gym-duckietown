import pickle

recording_file_name = 'trained_models/supervised/1/rom_adag/training.pkl'

HORIZON = 512
EPISODES = 10


def empty_statistics():
    return {
        'risk': 0,
        'reward': 0,
        'interventions': 0
    }


if __name__ == '__main__':
    recording_file = open(recording_file_name, 'rb')

    iterations_statistics = []
    try:
        episode_data = pickle.load(recording_file)
        total_samples = len(episode_data)
        statistics = empty_statistics()
        for index, sample in enumerate(episode_data):
            env = sample['env']
            hidden = sample['hidden']
            statistics['reward'] += env[1]
            if env[2] and index < total_samples:
                statistics['risk'] += 1
            if index % (HORIZON - 1) == 0 and index / (HORIZON - 1) > 0:
                statistics['interventions'] = hidden[4]
                iterations_statistics.append(statistics)
                statistics = empty_statistics()
    except EOFError:
        print('Finishing analysis...')

    # print('cummulative reward: {}, negative reward: {}, expert intervention: {}, risk: {}'.format(cumulative_reward,
    #                                                                                               negative_reward,
    #                                                                                               expert_intervention,
    #                                                                                               risk))

    print(len(iterations_statistics))