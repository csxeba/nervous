from collections import OrderedDict

import numpy as np

# from keras.layers.convolutional import Conv2D
# from keras.layers import Dense


_layer_unit_name = {"Dense": "units", "Conv2D": "filters"}


def calculate_synaptic_probabilities(kernel, norm, group_probabilities):
    synaptic_probs = np.exp(np.abs(kernel)/norm) - 1.
    if kernel.ndim == 2:
        synaptic_probs *= group_probabilities[:, None]
    elif kernel.ndim == 4:
        synaptic_probs *= group_probabilities[None, None, None, :]
    return synaptic_probs


def calculate_group_probabilities(kernel, norm, bias=None):
    filter_probs = np.exp(np.sum(np.trunc(np.abs(kernel)), axis=-1)/norm - 1.)
    if bias is not None:
        bias_probs = np.exp(np.sum(np.trunc(np.abs(kernel)))/norm - 1.)
        filter_probs *= bias_probs
    return filter_probs


class SynapticProbabilityModel:

    def __init__(self,
                 layers,
                 synaptic_environmental_constraint,
                 group_environmental_constraint):
        """
        Holds and evolves the synaptic probability model.
        :param synaptic_environmental_constraint: 0-1, ratio of connections to let into the next generation
        :param group_environmental_constraint: 0-1, ratio of groups to let into the next generation
        """
        self.environmental_constraint = synaptic_environmental_constraint * group_environmental_constraint
        self.synaptic_probabilities = OrderedDict({layer.name: None for layer in layers})
        self.update_probabilities(layers)

    def update_probabilities(self, layers):
        error = RuntimeError("This model is not for the layers you passed to it!")
        layer_names = set(self.synaptic_probabilities)
        if len(layer_names) != len(layers):
            raise error
        for layer in layers:
            parameters = layer.get_weights()
            z = np.max(parameters[0])
            group_probs = calculate_group_probabilities(parameters[0], z, parameters[1] if layer.use_bias else None)
            synaptic_probs = calculate_synaptic_probabilities(parameters[0], z, group_probs)
            self.synaptic_probabilities[layer.name] = synaptic_probs * self.environmental_constraint
            try:
                layer_names.remove(layer.name)
            except Exception:
                raise error

    def sample_weight_masks(self):
        return {name: np.random.uniform(size=prob.shape) < prob
                for name, prob in self.synaptic_probabilities.items()}
