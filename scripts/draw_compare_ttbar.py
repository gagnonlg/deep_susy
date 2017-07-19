import argparse
import logging

import h5py as h5
import numpy as np
import ROOT
import root_numpy as rnp

from root_graph_utils import atlas_utils
import utils


KEYS = ['ttbar', 'MGPy8EG_ttbar', 'PhHppEG_ttbar']
LOG = logging.getLogger(__name__)
WEIGHTS = {}

def _key2name(key):
    return {
        'ttbar': 'Powheg+Pythia6',
        'MGPy8EG_ttbar': 'MadGraph+Pythia8',
        'PhHppEG_ttbar': 'Powheg+Herwigpp'
    }[key]

def _key2col(key):
    return {
        'ttbar': ROOT.kBlack,
        'MGPy8EG_ttbar': ROOT.kBlue,
        'PhHppEG_ttbar': ROOT.kRed
    }[key]


def _make_figure(dset, var):
    cnv = ROOT.TCanvas('c_' + var, '', 0, 0, 800, 600)
    cnv.SetLogy()
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    leg.SetBorderSize(0)
    for i, key in enumerate(KEYS):
        LOG.debug((i,key))
        data = dset[key][var]
        if key not in WEIGHTS:
            WEIGHTS[key] = dset[key]['M_weight']
        hist = ROOT.TH1F(
            'h_' + key + '_' + var,
            '',
            100,
            np.min(data),
            np.max(data)
        )
        ROOT.SetOwnership(hist, False)
        rnp.fill_hist(hist, data, weights=WEIGHTS[key])
        hist.SetLineColor(_key2col(key))
        hist.SetTitle(';' + var + ';Event density')
        hist.DrawNormalized('hist' + ('' if i == 0 else 'same'))
        leg.AddEntry(hist, _key2name(key), 'L')

    atlas_utils.atlas_label(0.2, 0.88)
    txt = ROOT.TLatex()
    txt.SetNDC()
    txt.DrawText(0.33, 0.88, 'Simulation')
    txt.DrawText(0.2, 0.83, 'Internal')
    leg.Draw()
    cnv.SaveAs('compare_ttbar_' + var + '.pdf')


def _variables(dset):
    return [name for name, _ in dset['ttbar'].dtype.descr]


def _get_args():
    args = argparse.ArgumentParser()
    args.add_argument('input')
    return args.parse_args()


def _main():
    ROOT.gROOT.SetBatch()
    atlas_utils.set_atlas_style()
    args = _get_args()
    LOG.info('input: %s', args.input)
    dset = h5.File(args.input, 'r')
    for v in _variables(dset):
        LOG.debug(v)
        _make_figure(dset, v)

    return 0


if __name__ == '__main__':
    utils.main(_main, 'compare_ttbar')
