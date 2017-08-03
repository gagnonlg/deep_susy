""" Code to run training and evaluation of models """
import imp
import logging

import h5py as h5
import keras
import numpy as np

from deep_susy import dataset, utils

LOG = logging.getLogger(__name__)

keras.backend.set_floatx('float32')


def load(path):
    """ Load a model from a .py source """
    LOG.info('Loading model from %s', path)
    defm = imp.load_source('model_def', path)
    if 'build_model' not in dir(defm):
        raise RuntimeError("build_model() function not defined in '%s'", path)
    return defm.build_model


def build(buildf, x_data, y_data):
    """ Build a model given input data and a build function """
    LOG.info('Building model structure')
    model_def = _default_def()
    buildf(model_def, x_data, y_data)
    LOG.info('Model name: %s', model_def['name'])
    LOG.info('Compiling model')
    model_def['keras_model'].compile(
        optimizer=model_def['optimizer'],
        loss=model_def['loss'],
        metrics=['accuracy'],
    )
    return model_def


def train(model_def, x_data, y_data):
    """ Train a compiled model """
    LOG.info('Training model')
    callbacks = model_def['callbacks'] + [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            filepath=utils.unique_path(model_def['name'] + '_trained.h5'),
            save_best_only=True,
            verbose=1
        )
    ]
    model_def['keras_model'].fit(
        x_data,
        y_data,
        callbacks=callbacks,
        epochs=model_def['max_epochs'],
        validation_split=0.01,
        verbose=2
    )
    return model_def


def train_from_file(path, x_data, y_data):
    """ load/compile/train a model """
    return train(build(load(path), x_data, y_data), x_data, y_data)


def train_on_NNinput(model_path, data_path):
    """ load/compile/train a model on NNinput data """
    # pylint: disable=invalid-name,no-member
    LOG.info('loading data from %s', data_path)
    dset = h5.File(data_path, 'r')
    xdata = dataset.unpack(dset, 'training', 'input')
    ydata = dataset.unpack(dset, 'training', 'target')
    ishuf = range(xdata.shape[0])
    np.random.shuffle(ishuf)
    xdata = xdata[ishuf]
    ydata = ydata[ishuf]
    train_from_file(model_path, xdata, ydata)


def _default_def():
    return {
        'name': 'DEFAULT',
        'keras_model': None,
        'optimizer': None,
        'loss': None,
        'callbacks': [],
        'max_epochs': 0,
        'batch_size': 0,
    }
