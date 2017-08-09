import uuid

import keras
import numpy as np

from deep_susy import preprocess, utils

# optimized parameters
L2 = np.random.uniform(0, 1e-3)
NLAYERS = np.random.randint(1, 6)
NUNITS = utils.draw_geometrically(100, 1000)
NORMALIZATION = np.random.choice([
    None,
    preprocess.normalization,
    preprocess.standardize
])

def build_model(model, x_dset, y_dset, *args, **kwargs):

    input_node = keras.layers.Input((x_dset.shape[1],))

    if NORMALIZATION is None:
        k_model = input_node
    else:
        k_model = NORMALIZATION(x_dset)(input_node)

    for _ in range(NLAYERS):
        k_model = keras.layers.Dense(
            NUNITS,
            activation='relu',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=keras.regularizers.l2(L2)
        )(k_model)

    output_node = keras.layers.Dense(
        y_dset.shape[1],
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(L2),
        kernel_initializer='glorot_uniform',
    )(k_model)

    k_model = keras.models.Model(inputs=input_node, outputs=output_node)

    model['name'] = 'gen_001.' + str(uuid.uuid4())
    model['keras_model'] = k_model
    model['optimizer'] = keras.optimizers.Adam()
    model['loss'] = 'categorical_crossentropy'
    model['batch_size'] = 32
    model['max_epochs'] = 400

    model['callbacks'] = [
        keras.callbacks.EarlyStopping(
            patience=10,
            verbose=1
        )
    ]
