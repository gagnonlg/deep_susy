import logging
import uuid

import keras
import numpy as np
import theano

from deep_susy import preprocess, utils

# optimized parameters
hyperparameters = {}
hyperparameters['HIDDEN_L1'] = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
hyperparameters['HIDDEN_L2'] = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
hyperparameters['OUTPUT_L1'] = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
hyperparameters['OUTPUT_L2'] = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
hyperparameters['NLAYERS'] = np.random.randint(1, 6)
hyperparameters['NUNITS'] = np.random.randint(100, 1001)
hyperparameters['LEARNING_RATE'] = utils.draw_exponentially(1e-4, 1e-2)
hyperparameters['BATCH_NORM'] = 0 if hyperparameters['NLAYERS'] == 1 else np.random.randint(0, 2)
hyperparameters['DROPOUT_INPUT'] = np.random.choice([0.0, 0.8])
hyperparameters['DROPOUT_HIDDEN'] = np.random.choice([0.0, 0.5])

def build_model(model, x_dset, y_dset, *args, **kwargs):

    input_node = keras.layers.Input((x_dset.shape[1],))

    k_model = preprocess.standardize(x_dset)(input_node)

    if hyperparameters['DROPOUT_INPUT'] > 0:
        k_model = keras.layers.Dropout(float(hyperparameters['DROPOUT_INPUT']))(k_model)

    for _ in range(hyperparameters['NLAYERS']):
        k_model = keras.layers.Dense(
            hyperparameters['NUNITS'],
            kernel_initializer='glorot_uniform',
            kernel_regularizer=(
                keras.regularizers.l1_l2(
                    l1=hyperparameters['HIDDEN_L1'],
                    l2=hyperparameters['HIDDEN_L2']
                )
            ),
            use_bias=(not hyperparameters['BATCH_NORM'])
        )(k_model)

        if hyperparameters['BATCH_NORM']:
            k_model = keras.layers.BatchNormalization()(k_model)

        k_model = keras.layers.Activation('relu')(k_model)

        if hyperparameters['DROPOUT_HIDDEN'] > 0:
            k_model = keras.layers.Dropout(float(hyperparameters['DROPOUT_HIDDEN']))(k_model)

    k_model = keras.layers.Dense(
        y_dset.shape[1],
        activation='softmax',
        kernel_regularizer=(
            keras.regularizers.l1_l2(
                l1=hyperparameters['OUTPUT_L1'],
                l2=hyperparameters['OUTPUT_L2']
            )
        ),
        kernel_initializer='glorot_uniform',
    )(k_model)

    output_node = keras.layers.Activation('softmax')(k_model)

    k_model = keras.models.Model(inputs=input_node, outputs=output_node)

    model['name'] = 'gen_002.' + str(uuid.uuid4())
    model['keras_model'] = k_model
    model['optimizer'] = keras.optimizers.Adam(lr=hyperparameters['LEARNING_RATE'])
    model['loss'] = 'categorical_crossentropy'
    model['batch_size'] = 32
    model['max_epochs'] = 500

    model['callbacks'] = [
        keras.callbacks.EarlyStopping(
            patience=10,
            verbose=1
        )
    ]

    model['hyperparameters'] = hyperparameters
