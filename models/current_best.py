import argparse
import logging
import os
import uuid

import keras
import numpy as np
import theano

from deep_susy import preprocess, utils

# optimized parameters
hyperparameters = {}
hyperparameters['PARAMETRIZATION'] = 'pxpypze'
hyperparameters['HIDDEN_L1'] = 7.2789510821323538e-05
hyperparameters['HIDDEN_L2'] = 0.0
hyperparameters['OUTPUT_L1'] = 2.68436754794579e-06
hyperparameters['OUTPUT_L2'] = 0.0
hyperparameters['NLAYERS'] = 2
hyperparameters['NUNITS'] = 996
hyperparameters['LEARNING_RATE'] = 0.00010713827964378717
hyperparameters['BATCH_NORM'] = 1
hyperparameters['DROPOUT_INPUT'] = 0.2
hyperparameters['DROPOUT_HIDDEN'] = 0.5
hyperparameters['BATCH_SIZE'] = 256
hyperparameters['NORMALIZATION'] =  preprocess.standardize

def build_model(model, x_dset, y_dset, x_dtype, *args, **kwargs):

    input_node = keras.layers.Input((x_dset.shape[1],))

    k_model = hyperparameters['NORMALIZATION'](x_dset, x_dtype)(input_node)

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

    model['name'] = 'current_best'
    model['keras_model'] = k_model
    model['optimizer'] = keras.optimizers.Adam(lr=hyperparameters['LEARNING_RATE'])
    model['loss'] = 'categorical_crossentropy'
    model['batch_size'] = hyperparameters['BATCH_SIZE']
    model['max_epochs'] = 500

    model['callbacks'] = [
        keras.callbacks.EarlyStopping(
            patience=10,
            verbose=1
        )
    ]

    model['hyperparameters'] = hyperparameters
