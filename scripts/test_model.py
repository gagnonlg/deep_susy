""" test a model on xor dataset """

import argparse
import logging

import keras
import numpy as np

from deep_susy import custom_layers, model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('--max-epochs', type=int)
    return args.parse_args()


def _gen_data():
    # pylint: disable=no-member
    xdat = np.random.normal(size=(10000, 2)).astype(np.float32)
    ydat = np.logical_xor(
        xdat[:, 0] >= 0,
        xdat[:, 1] >= 0
    ).astype(np.float32)[:, np.newaxis]
    return xdat, keras.utils.np_utils.to_categorical(ydat)


def _main():
    args = _get_args()
    log = logging.getLogger('test_model')

    log.info('generating data')
    xdat, ydat = _gen_data()

    mdef = model.build(model.load(args.model), xdat, ydat)
    if args.max_epochs is not None:
        mdef['max_epochs'] = args.max_epochs
    mdef = model.train(mdef, xdat, ydat)

    custom = {
        l: getattr(custom_layers, l)
        for l in dir(custom_layers) if l[0].isupper()
    }

    loaded = keras.models.load_model(
        mdef['path'],
        custom_objects=custom
    )

    xtest = _gen_data()[0]
    np.testing.assert_equal(
        mdef['keras_model'].predict(xtest),
        loaded.predict(xtest)
    )


if __name__ == '__main__':
    utils.main(_main, 'test_model')
