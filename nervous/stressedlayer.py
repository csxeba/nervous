from keras.layers import Layer

from .probability_model import calculate_group_probabilities, calculate_synaptic_probabilities


class StressedLayer:

    def __init__(self, layer: Layer, layer_config):
        self.config = layer_config
        self.nodes = layer._outbound_nodes
        self.lname = layer.name
        self.ltype = layer.__class__.__name__
        self.synaptic_probabilities = None
        self.group_probabilities = None

    def step(self, layer):
