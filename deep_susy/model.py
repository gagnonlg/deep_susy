""" Code to run training and evaluation of models """
import collections
import imp
import logging

import h5py as h5
import keras
import numpy as np

from deep_susy import custom_layers, dataset, utils

LOG = logging.getLogger(__name__)

keras.backend.set_floatx('float32')


def get_hyperparameters(k_model):
    """ Get hyperparameters in dictionnary form """

    hps = {}

    layers = k_model.get_config()['layers']

    # NLAYERS
    denses = [l for l in layers if l['class_name'] == 'Dense']
    hps['n_hidden_layers'] = len(denses) - 1

    # NUNITS
    hps['n_hidden_units'] = denses[0]['config']['units']

    # L2
    hps['l2'] = denses[0]['config']['kernel_regularizer']['config']['l2']

    norm = '???'
    # logging.debug(layers[1]['config']['scale'])
    if not layers[1]['name'].startswith('scale_offset_'):
        norm = '"None"'
    else:
        # HACK!!!
        # check if the offset == 0 for first 4 jets
        noff_0 = np.all(layers[1]['config']['offset'][0:4] == 0)
        noff_1 = np.all(layers[1]['config']['offset'][5:9] == 0)
        noff_2 = np.all(layers[1]['config']['offset'][10:14] == 0)
        noff_3 = np.all(layers[1]['config']['offset'][15:19] == 0)
        if noff_0 and noff_1 and noff_2 and noff_3:
            norm = '"4vec"'
        else:
            norm = '1402.4735'
    hps['normalization'] = norm

    return hps

def load(path):
    """ Load a model from a .py source """
    LOG.info('Loading model from %s', path)
    defm = imp.load_source('model_def', path)
    if 'build_model' not in dir(defm):
        raise RuntimeError("build_model() function not defined in '%s'", path)
    return defm.build_model


def load_keras(path, compile=True):
    """ Load a keras model """
    LOG.info("Loading keras model from %s", path)
    custom = {
        name: getattr(custom_layers, name) for name in dir(custom_layers)
        if name[0].isupper()
    }
    return keras.models.load_model(
        path,
        custom_objects=custom,
        compile=compile
    )


def build(buildf, x_data, y_data, x_dtype=None):
    """ Build a model given input data and a build function """
    LOG.info('Building model structure')
    model_def = _default_def()
    buildf(model_def, x_data, y_data, x_dtype)
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
    model_def['path'] = utils.unique_path(model_def['name'] + '_trained.h5')
    callbacks = model_def['callbacks'] + [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            filepath=model_def['path'],
            save_best_only=True,
            verbose=1
        )
    ]

    if 'hyperparameters' in model_def:
        with open(
            utils.unique_path(model_def['name'] + '.hyperparameters'),
            'w') as hpf:
            hpf.write(str(model_def['hyperparameters']))

    hist = model_def['keras_model'].fit(
        x_data,
        y_data,
        callbacks=callbacks,
        epochs=model_def['max_epochs'],
        validation_split=0.01,
        verbose=2
    )
    model_def['keras_model'].load_weights(model_def['path'])
    _save_history(
        hist.history,
        utils.unique_path(model_def['name'] + '_history.h5')
    )
    return model_def


def train_from_file(path, x_data, y_data, x_dtype=None):
    """ load/compile/train a model """
    return train(build(load(path), x_data, y_data, x_dtype), x_data, y_data)


def train_on_NNinput(model_path, data_path):
    """ load/compile/train a model on NNinput data """
    # pylint: disable=invalid-name,no-member
    LOG.info('loading data from %s', data_path)
    dset = h5.File(data_path, 'r')
    xdtype = dataset.get_dtype(dset, 'training', 'input')
    xdata = dataset.unpack(dset, 'training', 'input')
    ydata = dataset.unpack(dset, 'training', 'target')
    ishuf = range(xdata.shape[0])
    np.random.shuffle(ishuf)
    xdata = xdata[ishuf]
    ydata = ydata[ishuf]
    return train_from_file(model_path, xdata, ydata, xdtype)


def _default_def():
    return collections.OrderedDict([
        ('name', 'DEFAULT'),
        ('keras_model', None),
        ('optimizer', None),
        ('loss', None),
        ('callbacks', []),
        ('max_epochs', 0),
        ('batch_size', 0),
    ])


def _save_history(hist, path):
    with h5.File(path, 'x') as hfile:
        for key, val in hist.iteritems():
            hfile.create_dataset(key, data=np.array(val))
    logging.info('Fit history saved to %s', path)
