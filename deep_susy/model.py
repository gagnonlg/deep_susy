""" Code to run training and evaluation of models """
import imp
import logging

import h5py as h5
import keras
import numpy as np
import ROOT
import root_numpy

from root_graph_utils import atlas_utils
from deep_susy import custom_layers, dataset, utils

LOG = logging.getLogger(__name__)

keras.backend.set_floatx('float32')


def threshold(k_model, data, key, masses):
    logging.info('Computing decision threshold for %s', masses)
    bkey = 'background/' + key

    b_input = dataset.destructure_array(data[bkey + '/input'].value)
    b_input[:, -2] = masses[0]
    b_input[:, -1] = masses[1]

    outs = k_model.predict(b_input)

    buf = np.empty((outs.shape[0], 2))
    buf[:, 0] = np.log(outs[:, 0]) - np.log(np.sum(outs[:, 1:], axis=1))
    buf[:, 1] = data[bkey + '/metadata'].value['M_weight']

    # low score == high SUSY probability
    # high score == low SUSY probability
    # sort(score): ordered from high SUSY probability to low
    buf.sort(axis=0)

    # after taking the cumulative sum, the i-th element is the 1 ifb
    # yield when retaining events for which the score is equal or
    # lower than the i-th ratio.
    uncert = np.sqrt(np.cumsum(np.power(buf[:, 1], 2))) / np.cumsum(buf[:, 1])

    # The chosen threshold is the lowest one for which the uncertainty
    # doesn't exceed 0.3. np.argmax returns the first occurance
    i_min = np.argmax(np.logical_and(uncert > 0, uncert <= 0.3))
    LOG.debug(
        'masses=%s, i_min=%d, threshold=%f, yield=%f, uncert=%f',
        masses,
        i_min,
        buf[i_min, 0],
        np.sum(buf[:i_min+1, 1]),
        uncert[i_min]
    )
    # still need to check that at least one point has the correct
    # uncertainty
    if uncert[i_min] > 0.3:
        raise RuntimeError("Cannot find suitable threshold")
    return buf[i_min, 0]


def evaluate(k_model, history, data):
    ROOT.gROOT.SetBatch(True)
    atlas_utils.set_atlas_style()
    _fit_history_curves(history)


def _fit_history_curves(history):
    LOG.info('Creating the fit history curves')
    keys = ['acc', 'loss']

    def _enumerate(key):
        data = history[key].value
        buf = np.empty((data.shape[0], 2))
        buf[:, 0] = np.arange(data.shape[0])
        buf[:, 1] = data
        return buf

    for key in ['acc', 'loss']:
        graphs = ROOT.TMultiGraph('mg_' + key, '')
        data = _enumerate(key)
        val_data = _enumerate('val_' + key)
        graph = ROOT.TGraph(data.shape[0])
        val_graph = ROOT.TGraph(val_data.shape[0])
        root_numpy.fill_graph(graph, data)
        root_numpy.fill_graph(val_graph, val_data)
        val_graph.SetLineColor(ROOT.kRed)
        graphs.Add(graph)
        graphs.Add(val_graph)

        graph.SetLineWidth(2)
        val_graph.SetLineWidth(2)

        canvas = ROOT.TCanvas('fit_history', '', 0, 0, 800, 600)
        graphs.SetTitle(';Epoch;' + key)
        graphs.Draw('AL')
        canvas.SaveAs('fit_history_{}.pdf'.format(key))


def load(path):
    """ Load a model from a .py source """
    LOG.info('Loading model from %s', path)
    defm = imp.load_source('model_def', path)
    if 'build_model' not in dir(defm):
        raise RuntimeError("build_model() function not defined in '%s'", path)
    return defm.build_model


def load_keras(path):
    """ Load a keras model """
    LOG.info("Loading keras model from %s", path)
    custom = {
        name: getattr(custom_layers, name) for name in dir(custom_layers)
        if name[0].isupper()
    }
    return keras.models.load_model(
        path,
        custom_objects=custom
    )


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
    model_def['path'] = utils.unique_path(model_def['name'] + '_trained.h5')
    callbacks = model_def['callbacks'] + [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            filepath=model_def['path'],
            save_best_only=True,
            verbose=1
        )
    ]
    hist = model_def['keras_model'].fit(
        x_data,
        y_data,
        callbacks=callbacks,
        epochs=model_def['max_epochs'],
        validation_split=0.01,
        verbose=2
    )
    _save_history(
        hist.history,
        utils.unique_path(model_def['name'] + '_history.h5')
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
    return train_from_file(model_path, xdata, ydata)


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


def _save_history(hist, path):
    with h5.File(path, 'x') as hfile:
        for key, val in hist.iteritems():
            hfile.create_dataset(key, data=np.array(val))
    logging.info('Fit history saved to %s', path)
