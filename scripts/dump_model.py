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
    # k_model.summary()

    layers = k_model.get_config()['layers']

    # NLAYERS
    denses = [l for l in layers if l['class_name'] == 'Dense']
    logging.info('NLAYERS: %d', len(denses) - 1)
    # NUNITS
    logging.info('NUNITS: %d', denses[0]['config']['units'])
    # L2
    logging.info(
        'L2: %f',
        denses[0]['config']['kernel_regularizer']['config']['l2']
    )
    # NORMALIZATION
    norm = '???'
    # logging.debug(layers[1]['config']['scale'])
    if np.all(layers[1]['config']['scale'] == 1):
        norm = 'None'
    elif layers[1]['config']['offset'][3] == 0:  # HACK!!!!
        norm = '4vec'
    else:
        norm = '1402.4735'
    logging.info('NORMALIZATION: %s', norm)

if __name__ == '__main__':
    utils.main(_main, '')
