""" Launch a training on NNinput data """
# pylint: disable=invalid-name

import argparse

from deep_susy import model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('dataset')
    return args.parse_args()


def _main():
    args = _get_args()
    model.train_on_NNinput(args.model, args.dataset)


if __name__ == '__main__':
    utils.main(_main, 'train_on_NNinput')
