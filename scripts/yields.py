""" Program to compute expected Z across the Gtt grid """
# pylint: disable=no-member
import argparse
import array
import itertools
import logging
import os

import h5py as h5
import numpy as np
import ROOT
import root_numpy

from deep_susy import evaluation, utils
import mbj_contour
from root_graph_utils import atlas_utils


def _draw_atlas_label(preliminary):
    atlas_utils.atlas_label(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(
        0.33,
        0.88,
        'Simulation {}'.format(
            'Preliminary' if preliminary else 'Internal'
        )
    )


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('evaluated')

    # to get the weights
    args.add_argument('data')
    args.add_argument('setname')
    args.add_argument('output')

    args.add_argument('--lumi', default=36.1, type=float)
    args.add_argument('--uncert', default=0.3, type=float)
    args.add_argument('--cache', action='store_true')

    return args.parse_args()


def _main():
    args = _get_args()
    dfile = h5.File(args.evaluated, 'r')
    data = h5.File(args.data, 'r')[args.setname]

    if args.cache and os.path.exists('CONTOUR_CACHE.txt'):
        logging.info('Loading cached data')
        results = np.loadtxt(
                'CONTOUR_CACHE.txt',
                dtype=[
                    ('mg', 'i4'),
                    ('ml', 'i4'),
                    ('z', 'f4'),
                    ('s', 'f4'),
                    ('b', 'f4')
                ]
        )
    else:
        results = evaluation.compute_significance_grid(
            dfile,
            data,
            args.lumi,
            args.uncert
        )

        if args.cache:
            np.savetxt('CONTOUR_CACHE.txt', results)


    mbj_results = mbj_contour._get_yields_and_exp_Z(data, args.lumi, args.uncert)
    logging.debug(mbj_results.keys())

    with open(args.output, 'w') as ofile:
        header = '#mg,ml,MBJ_s,MBJ_b,MBJ_z,NN_s,NN_b,NN_z'
        ofile.write(header + '\n')
        logging.info(header)
        for i in range(results.shape[0]):
            mg, ml, z, s, b = results[['mg', 'ml', 'z', 's', 'b']][i]
            mbj_s, mbj_b, mbj_z = mbj_results[(int(mg), int(ml))]
            data = '{},{},{},{},{},{},{},{}'.format(
                mg,
                ml,
                mbj_s,
                mbj_b,
                mbj_z,
                s,
                b,
                z
            )
            logging.info(data)
            ofile.write(data + '\n')

if __name__ == '__main__':
    utils.main(_main, '')
