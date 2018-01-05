import argparse
import os

import h5py as h5
import numpy as np
import ROOT

from deep_susy import evaluation, utils

def main():

    args = argparse.ArgumentParser()
    args.add_argument('data')
    args.add_argument('setname')
    args.add_argument('evaluated_model')
    args.add_argument('MBJ_exclusion')
    args.add_argument('--cache', action='store_true')
    args = args.parse_args()

    input_data = h5.File(args.data, 'r')[args.setname]
    model_data = h5.File(args.evaluated_model, 'r')
    mbj_data_file = ROOT.TFile(args.MBJ_exclusion, 'READ')
    mbj_data = mbj_data_file.Get('contour')

    if args.cache and os.path.exists('TEST_EVALUATION_CACHE.h5'):
        ifile = h5.File('TEST_EVALUATION_CACHE.h5', 'r')
        model_grid = ifile['grid'].value
    else:
        model_grid = evaluation.compute_significance_grid(
            model_data,
            input_data,
            36.1,
            0.3
        )
        if args.cache:
            with h5.File('TEST_EVALUATION_CACHE.h5', 'w') as of:
                of.create_dataset('grid', data=model_grid)

    n_above = evaluation.compute_exclusion_above_mbj(mbj_data, model_grid)
    print(n_above)


if __name__ == '__main__':
    utils.main(main, '')
