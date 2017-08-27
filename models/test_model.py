""" Implementation of model from 1402.4735 """

import keras


def build_model(model, x_dset, y_dset, *args, **kwargs):
    """ build the model """
    # pylint: disable=unused-argument
    input_node = keras.layers.Input((x_dset.shape[1],))
    k_model = keras.layers.Dense(10, activation='sigmoid')(input_node)
    output_node = keras.layers.Dense(
        y_dset.shape[1],
        activation='softmax'
    )(input_node)
    k_model = keras.models.Model(inputs=input_node, outputs=output_node)

    model['name'] = 'TEST'
    model['keras_model'] = k_model
    model['optimizer'] = keras.optimizers.SGD()
    model['loss'] = 'categorical_crossentropy'
    model['callbacks'] = []
    model['batch_size'] = 32
    model['max_epochs'] = 10
