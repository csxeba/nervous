import numpy as np

from keras.callbacks import Callback

from ..probability_model import SynapticProbabilityModel


class StressedCallback(Callback):

    def __init__(self, stress_factor, probability_model: SynapticProbabilityModel, layers_of_interest):
        super().__init__()
        self.stress_factor = stress_factor
        self.probability_model = probability_model
        self.layers_of_interest = layers_of_interest

    def on_epoch_end(self, epoch, logs=None):
        masks = self.probability_model.sample_synaptic_masks(negate=True)
        for layer in self.model.layers:
            if layer.name not in self.layers_of_interest:
                continue
            weights = layer.get_weights()
            stress_mask = masks[layer.name] * self.stress_factor
            weights[0] *= stress_mask
            if layer.use_bias:
                bias_mask = np.all(stress_mask, axis=0) * self.stress_factor
                weights[1] *= bias_mask
            layer.set_weights(weights)
