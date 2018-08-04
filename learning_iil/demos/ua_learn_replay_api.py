import pickle
import tensorflow as tf
from learning_iil.learners import NeuralNetworkController
from learning_iil.learners.models.tf.baselines import ResnetOneRegression, ResnetOneMixture
from learning_iil.learners.models.tf.uncertainty import FortifiedResnetOneMixture, MonteCarloDropoutResnetOneMixture, FortifiedResnetOneRegression

tf.set_random_seed(1234)

recording_file_name = 'trained_models/supervised/1/ror_adag_64/training.pkl'

iteration = 1
base_directory = 'trained_models/supervised/{}/fo_rom_adag_64_32/'.format(iteration)

tf_model = FortifiedResnetOneMixture()
tf_learner = NeuralNetworkController(env=None,
                                     learner=tf_model,
                                     storage_location=base_directory)

if __name__ == '__main__':
    recording_file = open(recording_file_name, 'rb')
    observations = []
    actions = []
    try:
        episode = pickle.load(recording_file)
        total_samples = len(episode)
        for index, sample in enumerate(episode):
            agent = sample['agent']
            observations.append(agent[0])
            actions.append(agent[1])
    except EOFError:
        print('Finishing analysis...')

    print('LEARNING...')
    tf_learner.learn(observations, actions)
    tf_learner.save()
    print('DONE')
