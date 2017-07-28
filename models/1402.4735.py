""" Implementation of model from 1402.4735 """

import math

import keras
import numpy as np


def build_model(model, x_dset, y_dset, *args, **kwargs):
    """ build the model """
    # pylint: disable=unused-argument
    input_node = keras.layers.Input((x_dset.shape[1],))

    k_model = standardize(x_dset)(input_node)

    k_model = keras.layers.Dense(
        300,
        activation='tanh',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.1),
        kernel_regularizer=keras.regularizers.l2(1e-5),
    )(k_model)

    k_model = keras.layers.Dense(
        300,
        activation='tanh',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
        kernel_regularizer=keras.regularizers.l2(1e-5),
    )(k_model)

    k_model = keras.layers.Dense(
        300,
        activation='tanh',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
        kernel_regularizer=keras.regularizers.l2(1e-5),
    )(k_model)

    k_model = keras.layers.Dense(
        300,
        activation='tanh',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
        kernel_regularizer=keras.regularizers.l2(1e-5),
    )(k_model)

    k_model = keras.layers.Dense(
        300,
        activation='tanh',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.05),
        kernel_regularizer=keras.regularizers.l2(1e-5),
    )(k_model)

    output_node = keras.layers.Dense(
        y_dset.shape[1],
        activation='softmax',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.001),
        kernel_regularizer=keras.regularizers.l2(1e-5),
    )(k_model)

    k_model = keras.models.Model(inputs=input_node, outputs=output_node)

    model.name = '1402.4735'
    model.keras_model = k_model
    model.optimizer = keras.optimizers.SGD(lr=0.05)
    model.loss = 'categorical_crossentropy'
    model.callbacks = [
        keras.callbacks.LearningRateScheduler(schedule_lr),
        MomentumScheduler(
            start=0.9,
            end=0.99,
            nepochs=200
        ),
        EarlyStopping(
            min_epochs=200,
            threshold=0.00001,
            patience=10
        )
    ]
    model.batch_size = 100
    model.max_epochs = 1000


def standardize(dset):
    """ Compute the standardization constants """
    branches = [n for n, _ in dset.dtype.descr]
    scales = np.ones(len(branches))
    offsets = np.zeros(len(branches))
    means = np.mean(dset, axis=0)
    stds = np.std(dset, ddof=1, axis=0)
    scales = 1.0 / stds
    offsets = - means / stds

    for i, pos in enumerate(np.all(np.greater_equal(dset, 0), axis=0)):
        if pos:
            offsets[i] += 1

    return keras.layers.Lambda(
        lambda x: x * scales + offsets
    )


class EarlyStopping(keras.callbacks.EarlyStopping):
    """ Early stopping """
    def __init__(self, min_epochs, threshold, *args, **kwargs):
        self.min_epochs = min_epochs
        self.threshold = threshold
        super(EarlyStopping, self).__init__(*args, **kwargs)
        self.monitor_op = lambda x, y: (y - x) / y > self.threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 200:
            return super(EarlyStopping, self).on_epoch_end(epoch, logs)


class MomentumScheduler(keras.callbacks.Callback):
    """ Momentum schedule """
    def __init__(self, start, end, nepochs):
        self.start = start
        self.end = end
        self.nepochs = nepochs
        super(MomentumScheduler, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):

        if epoch >= self.nepochs:
            mom = self.end
        else:
            mom = epoch * (self.end - self.start) / self.nepochs + self.start

        keras.backend.set_value(self.model.optimizer.momentum, mom)


def schedule_lr(epoch):
    """ Learning rate schedule """
    initial = 0.05
    freduce = 1.0000002
    minrate = 1e-6
    minepoch = math.log(initial / minrate, freduce)

    if epoch >= minepoch:
        return 1e-6

    return initial / (freduce ** epoch)
