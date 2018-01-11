""" Compute the MBJ signal region expected Z contour """
import array
import argparse
import collections
import itertools
import logging

import h5py as h5
import numpy as np
import ROOT
import root_numpy

from deep_susy import signal_regions, utils


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('data')
    args.add_argument('setname')
    args.add_argument('--lumi', default=36.1, type=float)
    args.add_argument('--uncert', default=0.3, type=float)
    return args.parse_args()


def _bkg_keys(dset):
    return ['background/' + k for k in dset['background'].keys()]


def _sig_keys(dset):
    return ['signal/' + k for k in dset['signal'].keys()]


def _calc_yield(dset, srfun, key):
    return srfun(dset[key])[0]

def _get_yields_and_exp_Z(dset, lumi, uncert):
    byields = {}
    logging.info('Computing background yields')
    for srname, srfun in signal_regions.SR_dict.iteritems():
        logging.info('  %s', srname)
        byields[srname] = sum(
            [srfun(dset[key])[0] for key in _bkg_keys(dset)]
        )
    logging.info('Computing signal yields')
    syields = collections.defaultdict(dict)
    for skey in _sig_keys(dset):
        logging.info('  %s', skey)
        for srname, srfun in signal_regions.SR_dict.iteritems():
            logging.debug('    %s', srname)
            syields[skey][srname] = srfun(dset[skey])[0]
    logging.info('Computing significances')
    exp_zs = {}
    for i, skey in enumerate(_sig_keys(dset)):
        maxz = -float('inf')
        for srname in signal_regions.SR_dict:
            s_yield = syields[skey][srname]
            b_yield = byields[srname]
            exp_z = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(
                lumi * s_yield,
                lumi * b_yield,
                uncert
            )
            if exp_z > maxz:
                maxz = exp_z
        fields = skey.split('_')
        s = lumi * s_yield
        b = lumi * b_yield
        exp_zs[(int(fields[1]), int(fields[3]))] = (s,b,exp_z)

    return exp_zs

def _main():
    # pylint: disable=too-many-locals
    args = _get_args()

    with h5.File(args.data, 'r') as dfile:
        dset = dfile[args.setname]

        exp_zs = _get_yields_and_exp_Z(dset, args.lumi, args.uncert)

        logging.info('Extracting contour')
        bins_x = [
            900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
            1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500
        ]
        bins_y = [
            1, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800,
            2000
        ]
        bin_data = np.zeros((len(bins_x) * len(bins_y), 3))
        for i, (m_g, m_l) in enumerate(itertools.product(bins_x, bins_y)):
            bin_data[i, 0] = m_g
            bin_data[i, 1] = m_l
            try:
                bin_data[i, 2] = max(exp_zs[(m_g, m_l)][-1], 0)
            except KeyError:
                logging.warning('no data for mg=%d, ml=%d', m_g, m_l)
                bin_data[i, 2] = 0
        hist = ROOT.TH2F(
            'contour',
            '',
            len(bins_x) - 1,
            array.array('f', bins_x),
            len(bins_y) - 1,
            array.array('f', bins_y)
        )
        root_numpy.fill_hist(hist, bin_data[:, :2], bin_data[:, 2])

        logging.info('Saving to "MBJ_contour.root"')
        ofile = ROOT.TFile('MBJ_contour.root', 'RECREATE')
        ofile.cd()
        hist.Write()
        ofile.Close()


if __name__ == '__main__':
    utils.main(_main, '')
