""" Program to compute the decision thresholds """
import argparse
import collections
import cPickle
import functools
import logging
import multiprocessing

import h5py as h5

from deep_susy import model, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('model')
    args.add_argument('data')
    args.add_argument('set')
    args.add_argument('key')
    args.add_argument('output')
    args.add_argument('-j', '--jobs', type=int, default=1)
    return args.parse_args()


def _worker(k_model, data_path, setname, key, masses):
    with h5.File(data_path, 'r') as dfile:
        return model.threshold(k_model, dfile[setname], key, masses)


def _get_masses(data_path, setname):
    masses = []
    with h5.File(data_path, 'r') as dfile:
        for key in dfile[setname + '/signal'].keys():
            fields = key.split('_')
            masses.append((int(fields[1]), int(fields[3])))
    return masses


def _main():
    args = _get_args()
    masses = _get_masses(args.data, args.set)
    k_model = model.load_keras(args.model)

    pool = multiprocessing.Pool(args.jobs)
    thresholds = pool.map(
        functools.partial(_worker, k_model, args.data, args.set, args.key),
        masses
    )

    result = collections.OrderedDict(zip(masses, thresholds))
    with open(args.output, 'w') as rfile:
        cPickle.dump(
            result,
            rfile
        )

    for (m_g, m_l), thr in result.iteritems():
        logging.debug('mg=%d, ml=%d: %f', m_g, m_l, thr)


if __name__ == '__main__':
    utils.main(_main, '')
