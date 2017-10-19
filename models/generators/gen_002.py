import logging
import uuid

import keras
import numpy as np
import theano

from deep_susy import preprocess, utils

# optimized parameters
HIDDEN_L1 = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
HIDDEN_L2 = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
OUTPUT_L1 = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
OUTPUT_L2 = np.random.choice([0, utils.draw_geometrically(1e-7, 1e-4)])
NLAYERS = np.random.randint(1, 6)
NUNITS = np.random.randint(100, 1001)
LEARNING_RATE = utils.draw_exponentially(1e-4, 1e-2)
BATCH_NORM = 0 if NLAYERS == 1 else np.random.randint(0, 2)
DROPOUT_INPUT = np.random.choice([0.0, 0.8])
DROPOUT_HIDDEN = np.random.choice([0.0, 0.5])

logging.debug("HIDDEN_L1: %s", HIDDEN_L1)
logging.debug("HIDDEN_L2: %s", HIDDEN_L2)
logging.debug("OUTPUT_L1: %s", OUTPUT_L1)
logging.debug("OUTPUT_L2: %s", OUTPUT_L2)
logging.debug("NLAYERS: %s", NLAYERS)
logging.debug("NUNITS: %s", NUNITS)
logging.debug("LEARNING_RATE: %s", LEARNING_RATE)
logging.debug("BATCH_NORM: %s", BATCH_NORM)
logging.debug("DROPOUT_INPUT: %s", DROPOUT_INPUT)
logging.debug("DROPOUT_HIDDEN: %s", DROPOUT_HIDDEN)

def build_model(model, x_dset, y_dset, *args, **kwargs):

    input_node = keras.layers.Input((x_dset.shape[1],))

    k_model = preprocess.standardize(x_dset)(input_node)

    if DROPOUT_INPUT > 0:
        k_model = keras.layers.Dropout(float(DROPOUT_INPUT))(k_model)

    for _ in range(NLAYERS):
        k_model = keras.layers.Dense(
            NUNITS,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=(
                keras.regularizers.l1_l2(
                    l1=HIDDEN_L1,
                    l2=HIDDEN_L2
                )
            ),
            use_bias=(not BATCH_NORM)
        )(k_model)

        if BATCH_NORM:
            k_model = keras.layers.BatchNormalization()(k_model)

        k_model = keras.layers.Activation('relu')(k_model)

        if DROPOUT_HIDDEN > 0:
            k_model = keras.layers.Dropout(float(DROPOUT_HIDDEN))(k_model)

    k_model = keras.layers.Dense(
        y_dset.shape[1],
        activation='softmax',
        kernel_regularizer=(
            keras.regularizers.l1_l2(
                l1=OUTPUT_L1,
                l2=OUTPUT_L2
            )
        ),
        kernel_initializer='glorot_uniform',
    )(k_model)

    output_node = keras.layers.Activation('softmax')(k_model)

    k_model = keras.models.Model(inputs=input_node, outputs=output_node)

    model['name'] = 'gen_002.' + str(uuid.uuid4())
    model['keras_model'] = k_model
    model['optimizer'] = keras.optimizers.Adam(lr=LEARNING_RATE)
    model['loss'] = 'categorical_crossentropy'
    model['batch_size'] = 32
    model['max_epochs'] = 500

    model['callbacks'] = [
        keras.callbacks.EarlyStopping(
            patience=10,
            verbose=1
        )
    ]
