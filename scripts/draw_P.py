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

from deep_susy import dataset, evaluation, utils
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
    args.add_argument('--prefix')

    args.add_argument('--lumi', default=36.1, type=float)

    return args.parse_args()


def _main():
    ROOT.gROOT.SetBatch()
    ROOT.gStyle.SetPalette(ROOT.kBlueGreenYellow)
    atlas_utils.set_atlas_style()
    args = _get_args()
    dfile = h5.File(args.evaluated, 'r')
    data = h5.File(args.data, 'r')[args.setname]

    if args.prefix is None:
        prefix = os.path.basename(args.evaluated).replace('.h5', '').split(
            '_evaluated'
        )[0].split('.')[1].split('-')[0]
    else:
        prefix = args.prefix

    for tgt in dataset.TARGETS:
        logging.info(tgt)
        _draw_P(
            dfile,
            data,
            args.lumi,
            prefix + '-' + args.setname,
            target=tgt
        )


def _bkg_keys(dfile, sigkey):

    for key in dfile[sigkey + '/background']:
        if dfile[sigkey + '/background/' + key].shape[0] > 0:
            yield key


def _draw_P(evaluated, data, lumi, prefix, target='Gtt'):

    p_bin = [i for i, n in enumerate(dataset.TARGETS) if n == target][0]

    for sigkey in evaluated.keys():
        logging.info(sigkey)

        xmin = 0.0
        xmax = 1.0

        s_p = evaluated[sigkey + '/signal/' + sigkey].value[:, p_bin]
        s_w = data['signal/' + sigkey + '/metadata'].value['M_weight']
        s_h = ROOT.TH1D('h_s_{}_{}'.format(sigkey, target), '', 100, xmin, xmax)
        root_numpy.fill_hist(s_h, s_p, s_w)
        s_h.SetLineColor(ROOT.kRed)
        s_h.Scale(lumi)

        leg = ROOT.TLegend(0.63, 0.8, 0.83, 0.92)
        leg.AddEntry(s_h, sigkey, 'L')

        b_stk = ROOT.THStack('stk_'+sigkey+target, '')
        icols = [0, 90, 125, 200]
        for i,bkgkey in enumerate(_bkg_keys(evaluated, sigkey)):
            b_p = evaluated[sigkey + '/background/' + bkgkey].value[:, p_bin]
            b_w = data['background/' + bkgkey + '/metadata'].value['M_weight']
            b_h = ROOT.TH1D('h_{}_{}_{}'.format(bkgkey, sigkey, target), '', 100, xmin, xmax)
            root_numpy.fill_hist(b_h, b_p, b_w)
            b_h.Scale(lumi)
            b_h.SetLineWidth(0)
            col = ROOT.gStyle.GetColorPalette(icols[i])
            b_h.SetFillColor(col)
            leg.AddEntry(b_h, bkgkey, 'F')
            b_stk.Add(b_h)

        cnv = ROOT.TCanvas('c'+sigkey+target, '', 0, 0, 800, 600)
        cnv.SetLogy()
        b_stk.SetTitle(';P_{%s};Events'%target)
        b_stk.SetMaximum(b_stk.GetMaximum() * 10)
        b_stk.Draw('HIST')
        s_h.Draw('HIST SAME')

        leg.SetBorderSize(0)
        leg.Draw()

        txt = ROOT.TLatex()
        txt.SetNDC()
        txt.SetTextSize(txt.GetTextSize() * 0.65)
        txt.DrawLatex(
            0.2,
            0.82,
            '#tilde{g}#rightarrow t #bar{t} + #tilde{#chi}^{0}_{1}'
        )
        name = prefix
        txt.DrawText(
            0.2,
            0.77,
            'model: {}'.format(name)
        )

        _draw_atlas_label(preliminary=False)

        cnv.SaveAs('{}_{}_{}.pdf'.format(name, sigkey, target))


if __name__ == '__main__':
    utils.main(_main, '')
