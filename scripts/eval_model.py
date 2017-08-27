import argparse

import h5py as h5
import keras

from deep_susy import model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('history')
    args.add_argument('data')
    return args.parse_args()


def _main():
    args = _get_args()

    model.evaluate(
        k_model=model.load_keras(args.model),
        history=h5.File(args.history, 'r'),
        data=h5.File(args.data, 'r')['test']
    )

if __name__ == '__main__':
    utils.main(_main, '')
