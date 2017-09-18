""" Print model config """
import argparse
import logging

import numpy as np

from deep_susy import model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    return args.parse_args()


def _main():
    args = _get_args()
    k_model = model.load_keras(args.model)
    hps = model.get_hyperparameters(k_model)
    logging.info('NLAYERS: %d', hps['n_hidden_layers'])
    logging.info('NUNITS: %d', hps['n_hidden_units'])
    logging.info('L2: %f', hps['l2'])
    logging.info('NORMALIZATION: %s', hps['normalization'])


if __name__ == '__main__':
    utils.main(_main, '')
