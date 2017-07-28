""" test a model on xor dataset """

import argparse
import logging

import keras
import numpy as np

from deep_susy import model, utils


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


def _main():
    args = _get_args()
    log = logging.getLogger('test_model')

    log.info('generating data')
    xdat, ydat = _gen_data()

    model.train_from_file(args.model, xdat, ydat)


if __name__ == '__main__':
    utils.main(_main, 'test_model')
