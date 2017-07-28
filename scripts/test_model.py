""" test a model on xor dataset """

import argparse
import imp
import logging

import keras
import numpy as np

from deep_susy import utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    return args.parse_args()


def _gen_data():
    # pylint: disable=no-member
    xdat = np.random.normal(size=(10000, 2)).astype(np.float32)
    ydat = np.logical_xor(
        xdat[:, 0] >= 0,
        xdat[:, 1] >= 0
    ).astype(np.float32)[:, np.newaxis]
    return xdat, keras.utils.np_utils.to_categorical(ydat)


def _empty_def():
    class _Def(object):
        # pylint: disable=too-few-public-methods
        def __init__(self):
            self.name = '<default>'
            self.keras_model = None
            self.optimizer = 'sgd'
            self.loss = 'categorical_crossentropy'
            self.callbacks = []
            self.batch_size = 32
            self.max_epochs = 1000
    return _Def()


def _main():
    args = _get_args()
    log = logging.getLogger('test_model')

    log.info('generating data')
    xdat, ydat = _gen_data()

    log.info('building model structure')
    model_def = _empty_def()
    model = imp.load_source('model_def', args.model)
    model.build_model(model_def, xdat, ydat)

    log.info('compiling model')
    model_def.keras_model.compile(
        optimizer=model_def.optimizer,
        loss=model_def.loss,
        metrics=['accuracy']
    )

    callbacks = model_def.callbacks + [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            filepath=model_def.name + '_trained.h5',
            save_best_only=True
        )
    ]

    log.info('fitting model')
    model_def.keras_model.fit(
        xdat,
        ydat,
        callbacks=callbacks,
        batch_size=model_def.batch_size,
        epochs=model_def.max_epochs,
        validation_split=0.1,
        verbose=2,
    )


if __name__ == '__main__':
    utils.main(_main, 'test_model')
