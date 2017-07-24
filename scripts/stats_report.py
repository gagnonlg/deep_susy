""" count available statistics """

from __future__ import print_function

import argparse

import numpy as np
import h5py as h5

from deep_susy import signal_regions, utils


def _count_statistics(masterh5, signal):
    print('==> Initial statistics, background')
    b_dset = masterh5['background']
    print('{:>14} {:^7} {:^4}'.format('component', 'number', 'weight'))
    for key in b_dset:
        n_evts = b_dset['{}/input'.format(key)].shape[0]
        w_evts = np.sum(b_dset['{}/metadata'.format(key)]['M_weight'])
        print("{:>14} {:^6,} {:^4.3f}".format(key, n_evts, w_evts))

    if signal:
        print('==> Initial statistics, signal')
        s_dset = masterh5['signal']
        print('{:>18} {:^7} {:^4}'.format('component', 'number', 'weight'))
        for key in s_dset:
            n_evts = s_dset['{}/input'.format(key)].shape[0]
            w_evts = np.sum(s_dset['{}/metadata'.format(key)]['M_weight'])
            print("{:>18} {:^6,} {:^4.3f}".format(key, n_evts, w_evts))

    print('==> Signal regions, background')
    for srname, srfunc in signal_regions.SR_dict.items():
        print('  -> {}'.format(srname))
        print('{:>14} {:^4} {:^4}'.format('component', 'weight', 'rel.uncert'))
        for key in b_dset:
            weight, unc = srfunc(b_dset[key])
            print("{:>14} {:.3f} {:.3f}".format(key, weight, unc / weight))


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--signal', action='store_true')
    args.add_argument('input')
    return args.parse_args()


def _main():
    args = _get_args()
    masterh5 = h5.File(args.input, 'r')
    _count_statistics(masterh5, args.signal)


if __name__ == '__main__':
    utils.main(_main, 'stats_report')
