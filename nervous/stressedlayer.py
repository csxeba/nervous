import numpy as np
from keras.layers import Layer

from .probability_model import calculate_group_probabilities, calculate_synaptic_probabilities


class StressedLayer:

    def __init__(self, layer: Layer, layer_config):
        self.config = layer_config
        self.nodes = layer._outbound_nodes
        self.lname = layer.name
        self.ltype = layer.__class__.__name__
        self.has_bias = layer.use_bias if hasattr(layer, "use_bias") else False
        self.synaptic_probabilities = None
        self.group_probabilities = None

    def step(self, layer):
        parameters = layer.get_weights()
        wnorm = parameters[0].max()
        self.group_probabilities = calculate_group_probabilities(
            parameters[0], wnorm, parameters[1] if self.has_bias else None)
        self.synaptic_probabilities = calculate_synaptic_probabilities(
            parameters[0], wnorm, self.group_probabilities)

        survivor_mask = np.random.uniform(size=parameters[0].size) > self.synaptic_probabilities
        pruned_filters, = np.where(np.all(survivor_mask, axis=-1))
