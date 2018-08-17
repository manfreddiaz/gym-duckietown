import pickle

recording_file_name = 'trained_models/upms_da_ne/2/ror_64_32_adag/training.pkl'
# statistics_file = open('trained_models/aggravate/experiments.pkl', mode='xab')
HORIZON = 512
EPISODES = 10


def empty_statistics():
    return {
        'risk': 0,
        'reward': 0,
        'interventions': 0,
        'disengagements': 0,
        'negative_reward': 0,
    }


if __name__ == '__main__':
    recording_file = open(recording_file_name, 'rb')

    episodes_stats = []
    try:
        episode_data = pickle.load(recording_file)
        total_samples = len(episode_data)
        statistics = empty_statistics()
        sum_interventions = 0
        sum_disengagements = 0
        for index, sample in enumerate(episode_data):
            env = sample['env']
            hidden = sample['hidden']
            statistics['reward'] += env[1]
            if -1000 < env[1] < 0:
                statistics['negative_reward'] += env[1]
            if env[2] and index < total_samples:
                statistics['risk'] += 1
            if index % (HORIZON - 1) == 0 and index / (HORIZON - 1) > 0:
                statistics['interventions'] = hidden[4] - sum_interventions
                statistics['disengagements'] = hidden[5] - sum_disengagements
                sum_interventions += statistics['interventions']
                sum_disengagements += statistics['disengagements']
                episodes_stats.append(statistics)
                statistics = empty_statistics()
        print(episodes_stats)
    except EOFError:
        print('Finishing analysis...')
    except:
        print('?')

    # print('cummulative reward: {}, negative reward: {}, expert intervention: {}, risk: {}'.format(cumulative_reward,
    #                                                                                               negative_reward,
    #                                                                                               expert_intervention,
    #                                                                                               risk))

    # pickle.dump(final_stats, statistics_file)
