import pickle

recording_file_name = 'trained_models/upms/1/ror_64_32_fo_1e-1_adag/training.pkl'

cumulative_reward = 0
negative_reward = 0
expert_intervention = 0
risk = 0  # times stepped outside of the environment


if __name__ == '__main__':
    recording_file = open(recording_file_name, 'rb')

    try:
        episode = pickle.load(recording_file)
        total_samples = len(episode)
        for index, sample in enumerate(episode):
            env = sample['env']
            hidden = sample['hidden']
            cumulative_reward += env[1]
            if env[2] and index < total_samples:
                risk += 1
            negative_reward += hidden[3]
        expert_intervention = episode[total_samples - 1]['hidden'][4] / total_samples
    except EOFError:
        print('Finishing analysis...')

    print('cummulative reward: {}, negative reward: {}, expert intervention: {}, risk: {}'.format(cumulative_reward,
                                                                                                  negative_reward,
                                                                                                  expert_intervention,
                                                                                                  risk))
