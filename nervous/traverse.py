"""Model-related utilities.
"""

from keras import backend as K
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.engine.input_layer import Input
from keras.engine.input_layer import InputLayer
from keras.engine.training import Model
from keras.engine.sequential import Sequential
from keras.layers import Layer


class LayerMemory:

    def __init__(self, model: Model, input_tensors):
        self.model = model
        self.input_tensors = to_list(input_tensors)
        self.layer_map = {}
        self.tensor_map = {}
        for x, y in zip(model.inputs, input_tensors):
            self.tensor_map[x] = (y, None)  # tensor, mask

    def _extract_layer_config(self):
        # Iterated over every node in the reference model, in depth order.
        depth_keys = list(self.model._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self.model._nodes_by_depth[depth]
            self.layer_map.update({node.outbound_layer.name: node.outbound_layer.get_config() for node in nodes})


def _traverse_functional_model(model, input_tensors):
    layer_map = {}  # Cache for created layers.
    tensor_map = {}  # Map {reference_tensor: (corresponding_tensor, mask)}

    # Make sure that all input tensors come from a Keras layer.
    # If tensor comes from an input layer: cache the input layer.
    input_tensors = to_list(input_tensors)
    _input_tensors = []
    for i, x in enumerate(input_tensors):
        if not K.is_keras_tensor(x):
            name = model._input_layers[i].name
            input_tensor = Input(tensor=x,
                                 name='input_wrapper_for_' + name)
            _input_tensors.append(input_tensor)
            # Cache newly created input layer.
            original_input_layer = x._keras_history[0]
            newly_created_input_layer = input_tensor._keras_history[0]
            layer_map[original_input_layer] = newly_created_input_layer
        else:
            _input_tensors.append(x)

    input_tensors = _input_tensors

    for x, y in zip(model.inputs, input_tensors):
        tensor_map[x] = (y, None)  # tensor, mask

    # Iterated over every node in the reference model, in depth order.
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            # Recover the corresponding layer.
            layer = node.outbound_layer

            # Get or create layer.
            if layer not in layer_map:
                # Clone layer.
                new_layer = layer.__class__.from_config(layer.get_config())
                layer_map[layer] = new_layer
                layer = new_layer
            else:
                # Reuse previously cloned layer.
                layer = layer_map[layer]
                # Don't call InputLayer multiple times.
                if isinstance(layer, InputLayer):
                    continue

            # Gather inputs to call the new layer.
            reference_input_tensors = node.input_tensors
            reference_output_tensors = node.output_tensors

            # If all previous input tensors are available in tensor_map,
            # then call node.inbound_layer on them.
            computed_data = []  # List of tuples (input, mask).
            for x in reference_input_tensors:
                if x in tensor_map:
                    computed_data.append(tensor_map[x])

            if len(computed_data) == len(reference_input_tensors):
                # Call layer.
                if node.arguments:
                    kwargs = node.arguments
                else:
                    kwargs = {}
                if len(computed_data) == 1:
                    computed_tensor, computed_mask = computed_data[0]
                    if has_arg(layer.call, 'mask'):
                        if 'mask' not in kwargs:
                            kwargs['mask'] = computed_mask
                    output_tensors = to_list(
                        layer(computed_tensor, **kwargs))
                    output_masks = to_list(
                        layer.compute_mask(computed_tensor,
                                           computed_mask))
                    computed_tensors = [computed_tensor]
                    computed_masks = [computed_mask]
                else:
                    computed_tensors = [x[0] for x in computed_data]
                    computed_masks = [x[1] for x in computed_data]
                    if has_arg(layer.call, 'mask'):
                        if 'mask' not in kwargs:
                            kwargs['mask'] = computed_masks
                    output_tensors = to_list(
                        layer(computed_tensors, **kwargs))
                    output_masks = to_list(
                        layer.compute_mask(computed_tensors,
                                           computed_masks))
                # Update tensor_map.
                for x, y, mask in zip(reference_output_tensors,
                                      output_tensors,
                                      output_masks):
                    tensor_map[x] = (y, mask)

    # Check that we did compute the model outputs,
    # then instantiate a new model from inputs and outputs.
    output_tensors = []
    for x in model.outputs:
        assert x in tensor_map, 'Could not compute output ' + str(x)
        tensor, _ = tensor_map[x]
        output_tensors.append(tensor)
    return Model(input_tensors, output_tensors, name=model.name)


def _traverse_sequential_model(model, input_tensors=None):

    def clone(layer):
        return layer.__class__.from_config(layer.get_config())

    layers = [clone(layer) for layer in model.layers]
    if len(to_list(input_tensors)) != 1:
        raise ValueError('To clone a `Sequential` model, we expect '
                         ' at most one tensor '
                         'as part of `input_tensors`.')
    x = to_list(input_tensors)[0]
    if K.is_keras_tensor(x):
        origin_layer = x._keras_history[0]
        if isinstance(origin_layer, InputLayer):
            return Sequential(layers=[origin_layer] + layers,
                              name=model.name)
        else:
            raise ValueError('Cannot clone a `Sequential` model on top '
                             'of a tensor that comes from a Keras layer '
                             'other than an `InputLayer`. '
                             'Use the functional API instead.')
    input_tensor = Input(tensor=x,
                         name='input_wrapper_for_' + str(x.name))
    input_layer = input_tensor._keras_history[0]
    return Sequential(layers=[input_layer] + layers, name=model.name)


def traverse_model(model, input_tensors):
    if isinstance(model, Sequential):
        return _traverse_sequential_model(model, input_tensors=input_tensors)
    else:
        return _traverse_functional_model(model, input_tensors=input_tensors)
