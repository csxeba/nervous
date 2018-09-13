from collections import namedtuple

import numpy as np
from keras.layers import Conv2D, Dense


def calculate_synaptic_probability(parameter, Z, C):
    """
    :param parameter: filters, kernels or biases
    :param Z: normalizing constant
    :param C: environmental constraint
    :return: vector of synaptic probabilities regarding groups (aka. units, neurons)
    """
    probabilities = np.exp(np.abs(parameter)) / Z - 1
    if parameter.ndim > 1:
        probabilities = np.prod(probabilities, axis=1)
    return probabilities * C


class ProbabilityModel:

    def __init__(self, config):
        self.Z = config.normalization_constant
        self.C = config.environmental_constraint
        if 1 < self.Z or 0 >= self.Z:
            raise RuntimeError("evnironmental constraint is a scalar > 0 and <= 1")
        self.synaptic_probabilities = None
        self.base_params = None

    def update_probabilities(self, base_filters, base_biases=None):
        num_filters = base_filters.shape[-1]
        if base_filters.shape[-1] != len(self.synaptic_probabilities):
            raise RuntimeError("Wrong number of filters! Expected {}, got {}"
                               .format(num_filters, len(self.synaptic_probabilities)))
        probabilities = calculate_synaptic_probability(base_biases, self.Z, self.C)
        self.base_params = [base_filters]
        if base_biases is not None:
            if len(base_biases) != num_filters:
                raise RuntimeError("Wrong number of biases! Expected {}, got {}"
                                   .format(num_filters, len(base_biases)))
            bias_probabilities = calculate_synaptic_probability(base_biases, self.Z, self.C)
            probabilities *= bias_probabilities
            self.base_params.append(base_biases)
        probabilities *= self.C
        self.synaptic_probabilities = probabilities
        return self

    def sample_new_number_of_units(self):
        selection_mask = np.random.uniform(size=self.synaptic_probabilities[0].shape) > self.synaptic_probabilities[0]
        return sum(selection_mask)


class StressedPopulation:
    Config = namedtuple("Config", ["normalization_constant", "environmental_constraint"])
    layer_type_map = {1: Conv2D, 2: Dense}
    layer_unit_param = {1: "filters", 2: "units"}

    def __init__(self, layers, config):
        self.layers = layers
        self.probability_models = [ProbabilityModel(config).update_probabilities(*layer.get_weights())
                                   for layer in self.iter_evolved_layers()]

    def iter_evolved_layers(self):
        return (l for l in self.layers if type(l) in (Conv2D, Dense))

    def update_probabilities(self):
        probability_model_stream = iter(self.probability_models)
        for layer in self.iter_evolved_layers():
            probability_model = next(probability_model_stream)
            probability_model.update_probabilities(*layer.get_weights())
