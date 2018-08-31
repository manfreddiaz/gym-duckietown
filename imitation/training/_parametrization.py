
# learner's parametrization
from imitation.learners.parametrizations.tf import MonteCarloDropoutResnetOneRegression, MonteCarloDropoutResnetOneMixture


PARAMETRIZATIONS_NAMES = ['resnet_one_regression', 'resnet_one_mixture']
PARAMETRIZATIONS = [MonteCarloDropoutResnetOneRegression, MonteCarloDropoutResnetOneMixture]


def parametrization(iteration, extra_parameters=None):
    return PARAMETRIZATIONS[iteration](**extra_parameters)