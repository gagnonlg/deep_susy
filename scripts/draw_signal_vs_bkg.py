""" Show signal vs background comparisons """
import argparse

import h5py as h5

from deep_susy import histograms, utils

BKGND_KEYS = ['WZjets', 'topEW', 'singletop', 'ttbar']
SIGNAL_KEYS = ['Gtt_2100_5000_1', 'Gtt_2100_5000_800', 'Gtt_2100_5000_1600']


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('masterh5')
    args.add_argument('--unweighted', action='store_true')
    args.add_argument('--output', default='.')
    args.add_argument('--variables', nargs='+')
    args.add_argument(
        '--ttbar',
        choices=['MGPy8', 'PhHpp', 'PhPy6'],
        default='PhPy6'
    )
    return args.parse_args()


def _main():
    args = _get_args()
    h5file = h5.File(args.masterh5)

    if args.ttbar != 'PhPy6':
        BKGND_KEYS.remove('ttbar')
        BKGND_KEYS.append(args.ttbar + '_ttbar')

    s_dsets = [h5file['signal/' + k] for k in SIGNAL_KEYS]
    b_dsets = [h5file['background/' + k] for k in BKGND_KEYS]

    histograms.signal_vs_background(
        signals=s_dsets,
        backgrounds=b_dsets,
        weighted=not args.unweighted,
        variables='all' if args.variables is None else args.variables,
        directory=args.output
    )


if __name__ == '__main__':
    utils.main(_main, 'draw_signal_vs_bkg')
