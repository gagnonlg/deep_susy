import keras
import numpy as np

from deep_susy import custom_layers

def build_model(model, x_dset, y_dset, *args, **kwargs):

    input_node = keras.layers.Input((x_dset.shape[1],))
    k_model = custom_layers.ScaleOffset(scale=100, offset=-3)(input_node)
    k_model = custom_layers.ScaleOffset(scale=np.full(x_dset.shape[1], -0.5))(k_model)
    model['name'] = 'test_scale_offset'
    model['keras_model'] = keras.models.Model(inputs=input_node, outputs=k_model)
    model['optimizer'] = 'sgd'
    model['loss'] = 'mse'
    model['max_epochs'] = 10
    model['batch_size'] = 32
