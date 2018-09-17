import os
import copy
from collections import namedtuple, OrderedDict

import numpy as np
from keras import Model
from keras.models import model_from_config

from .probability_model import SynapticProbabilityModel, _layer_unit_name
from .logger import NervousLogger


class StressedNet:

    Config = namedtuple("Config", ["synaptic_normalizing_term", "group_normalizing_term",
                                   "synaptic_environmental_constraint",
                                   "group_environmental_constraint",
                                   "save_folder"])
    Offspring = namedtuple("Offspring", ["file_path", "history"])

    def __init__(self, model: Model, stressed_net_config: Config):
        if not model._built:
            raise RuntimeError("Please build model before wrapping with StressedNet")
        self.ancestor_config_template = copy.copy(model.get_config())
        self._model_inputs = model.inputs
        self._model_losses = model.losses
        self._all_layer_configs = OrderedDict({layer_cfg["config"]["name"]: copy.copy(layer_cfg) for layer_cfg
                                               in self.ancestor_config_template})
        self._layers_of_interest = [name for name, layer in self._all_layer_configs.items() if
                                    layer["class_name"] in ("Dense", "Conv2D")]
        self.probability_model = SynapticProbabilityModel(model.layers, *stressed_net_config[:4])
        self.save_folder = stressed_net_config.save_folder
        self._generation_counter = 1
        self._offspring_counter = 1
        self._logger = NervousLogger()

    def sample_new_model(self):
        pruning_masks = self.probability_model.sample_weight_masks()
        pruning_masks_child = {name: None for name in self._layers_of_interest}
        for layer_name in self._layers_of_interest:
            layer_cfg = self._all_layer_configs[layer_name]
            layer_type = layer_cfg["class_name"]
            prune = pruning_masks[layer_name]
            pruned_filters = [filter_no for filter_no in range(prune.shape[-1]) if prune[..., filter_no].sum() == 0]
            pruning_masks_child[layer_name] = np.delete(prune, pruned_filters)
            layer_cfg[_layer_unit_name[layer_type]] -= len(pruned_filters)
            self._all_layer_configs[layer_name] = copy.copy(layer_cfg)
        new_model_config = copy.copy(self.ancestor_config_template)
        new_model_config["name"] += "_stressed_gen_{}_offspring_{}".format(
            self._generation_counter, self._offspring_counter)
        new_model_config["layers"] = copy.copy(self._all_layer_configs)
        new_model = model_from_config(new_model_config)  # type: Model
        for layer in new_model.layers:
            weights = layer.get_weights()
            weights[0] *= pruning_masks_child[layer.name]
            layer.set_weights(weights)
        return new_model

    def fit_generator(self,
                      generator,
                      generations,
                      num_offsprings,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):
        run_log = []
        for num_generation in range(generations):
            self._logger.info("*" * 50)
            self._logger.info("Starting Generation {}/{}".format(generations, num_generation))
            self._logger.info("*" * 50)
            offsprings = []
            for num_offspring in range(num_offsprings):
                self._logger.info("Training offspring {}/{}".format(num_offsprings, num_offspring))
                model = self.sample_new_model()
                model.compile(optimizer="sgd", loss=self._model_losses)
                history = model.fit_generator(
                    generator=generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=callbacks,
                    validation_data=validation_data,
                    validation_steps=validation_steps,
                    class_weight=class_weight,
                    max_queue_size=max_queue_size,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle,
                    initial_epoch=initial_epoch)
                offsprings.append(
                    self.Offspring(file_path=os.path.join(self.save_folder, model.name) + ".h5",
                                   history=history)
                )
                model.save(offsprings[-1].file_path)
                self._offspring_counter += 1
                run_log.append(offsprings)
            self._generation_counter += 1
            self._offspring_counter = 1
        self._logger.info("*"*50)
        print("Finished run!")
        return run_log
