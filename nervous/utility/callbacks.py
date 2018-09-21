from keras.callbacks import Callback
from nervous.probability_model import SynapticProbabilityModel


class StressedCallback(Callback):

    def __init__(self, stress_factor, probability_model: SynapticProbabilityModel):
        super().__init__()
        self.stress_factor = stress_factor
        self.probability_model = probability_model

    def on_epoch_end(self, epoch, logs=None):
        layer_probs = self.probability_model.synaptic_probabilities
        for layer in self.model.layers:
            if layer.__class__.__name__ not in ("Dense", "Conv2D"):
                pass
