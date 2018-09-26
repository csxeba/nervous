from collections import OrderedDict

from keras.layers import Layer
import numpy as np

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
        self.group_probabilities = OrderedDict({layer.name: None for layer in layers})
        self.update_probabilities(layers)

    def update_probabilities(self, layers):
        error = RuntimeError("This model is not for the layers you passed to it!")
        layer_names = set(self.synaptic_probabilities)
        if len(layer_names) != len(layers):
            raise error
        for layer in layers:
            parameters = layer.get_weights()
            z = np.max(parameters[0])
            self.group_probabilities[layer.name] = calculate_group_probabilities(
                parameters[0], z, parameters[1] if layer.use_bias else None)
            self.synaptic_probabilities[layer.name] = calculate_synaptic_probabilities(
                parameters[0], z, self.group_probabilities[layer.name]) * self.environmental_constraint
            self.group_probabilities[layer.name] *= self.environmental_constraint
            try:
                layer_names.remove(layer.name)
            except Exception:
                raise error

    def sample_unit_masks(self, negate=False):
        if negate:
            return {
                name: np.random.uniform(size=prob.shape) > prob
                for name, prob in self.group_probabilities.items()}
        return {
            name: np.random.uniform(size=prob.shape) < prob
            for name, prob in self.group_probabilities.items()}

    def sample_synaptic_masks(self, negate=False):
        if negate:
            return {
                name: np.random.uniform(size=prob.shape) > prob
                for name, prob in self.synaptic_probabilities.items()}
        return {
            name: np.random.uniform(size=prob.shape) < prob
            for name, prob in self.synaptic_probabilities.items()}


class LayerProbabilityModel:

    layers_modelled = {"Dense", "Conv2D"}

    def __init__(self, layer: Layer, layer_cfg, synaptic_constraint, environmental_constraint):
        self.layer_name = layer.name
        self.layer_type = layer.__class__.__name__
        self.layer_cfg = layer_cfg
        self.outbound_layers = [node.outbound_layer.name for node in layer._outbound_nodes]
        self.synaptic_constraint = synaptic_constraint
        self.environmental_constraint = environmental_constraint
        self.synaptic_probability = None

    def update_model(self, layer):
        if layer.name != self.layer_name:
            raise ValueError("This object is not modelling the layer passed to it!")
        if self.layer_type not in self.layers_modelled:
            return
        parameters = layer.get_weights()
        z = np.max(parameters[0])
        group_probs = calculate_group_probabilities(parameters[0], z, parameters[1] if layer.use_bias else None)
        synaptic_probs = calculate_synaptic_probabilities(parameters[0], z, group_probs)
        self.synaptic_probability = synaptic_probs * self.environmental_constraint

    def prune(self, prune_info):
        prune_mask = np.random.uniform(size=self.synaptic_probability) > self.synaptic_probability
        pruned_units = np.where(np.all(prune_mask, axis=-1))[0]
        self.layer_cfg[_layer_unit_name[self.layer_type]] -= pruned_units